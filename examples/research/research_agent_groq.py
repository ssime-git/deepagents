import os
import time
import logging
from typing import Literal
from functools import wraps

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tavily import TavilyClient
from groq import RateLimitError

from deepagents import create_deep_agent, SubAgent
from prompts import RESEARCH_AGENT_PROMPT, SUB_RESEARCH_PROMPT, SUB_CRITIQUE_PROMPT

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rate limiting configuration
RATE_LIMIT_DELAY = 60  # seconds to wait when rate limited
MAX_RETRIES = 3
BASE_DELAY = 1  # base delay between retries


# Search tool to use to do research
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search with logging and error handling"""
    logger.info(f"Starting web search for query: '{query}' (max_results={max_results}, topic={topic})")
    
    try:
        tavily_async_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        search_docs = tavily_async_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        
        # Log search results summary
        if hasattr(search_docs, 'get') and 'results' in search_docs:
            results_count = len(search_docs['results'])
        else:
            results_count = len(search_docs) if search_docs else 0
            
        logger.info(f"Web search completed: {results_count} results found for '{query}'")
        return search_docs
        
    except Exception as e:
        logger.error(f"Error during web search for '{query}': {str(e)}")
        # Return empty results to allow agent to continue
        return {"results": []}


# Define sub-agents using imported prompts
research_sub_agent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question.",
    "prompt": SUB_RESEARCH_PROMPT,
    "tools": ["internet_search"]
}

critique_sub_agent = {
    "name": "critique-agent",
    "description": "Used to critique the final report. Give this agent some information about how you want it to critique the report.",
    "prompt": SUB_CRITIQUE_PROMPT,
}

# 1️⃣ Create a Groq model instance with Kimi
# Note: Environment variables will be loaded from .env file automatically
kimi_model = ChatGroq(
    model="moonshotai/kimi-k2-instruct",  # Using Kimi K2 Instruct model from Moonshot AI via Groq
    temperature=0.1,
    max_tokens=16384,  # Maximum tokens for Kimi model (must be <= 16384)
    streaming=True,    # Enable streaming if you need it
)

# 2️⃣ Create the Deep Agent with the Groq model and sub-agents
agent = create_deep_agent(
    [internet_search],
    RESEARCH_AGENT_PROMPT,
    model=kimi_model,  # Pass the custom Groq model here instead of default Claude
    subagents=[critique_sub_agent, research_sub_agent],
).with_config({"recursion_limit": 1000})

# 3️⃣ Example usage
if __name__ == "__main__":
    # Environment variables will be loaded from .env file automatically
    # Make sure your .env file contains:
    # GROQ_API_KEY=your_groq_api_key_here
    # TAVILY_API_KEY=your_tavily_api_key_here
    
    # Test the agent with a research question
    test_question = "What are the latest developments in quantum computing in 2024?"
    
    print(f"🔍 Starting research on: {test_question}")
    print("=" * 60)
    

    def handle_rate_limit(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = BASE_DELAY
            while retries <= MAX_RETRIES:
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    if retries == MAX_RETRIES:
                        logger.error("Max retries reached. Exiting.")
                        raise
                    wait = delay + retries * RATE_LIMIT_DELAY
                    logger.warning(f"Rate limited: {str(e)}. Retrying in {wait} seconds...")
                    time.sleep(wait)
                    retries += 1
                    delay *= 2
        return wrapper

    @handle_rate_limit
    def run_agent_with_retry(messages):
        logger.info("Invoking agent with messages: %s", messages)
        return agent.invoke({"messages": messages})

    logger.info("Starting research question: %s", test_question)
    
    try:
        start_time = time.time()
        result = run_agent_with_retry([{"role": "user", "content": test_question}])
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"Research completed successfully in {duration:.2f} seconds")
        
        # Print the final response
        print("\n📊 Research Complete!")
        print("=" * 60)
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print("\nFinal Response:")
        print(result["messages"][-1].content)
        
        # Show any files that were created
        if "files" in result and result["files"]:
            logger.info(f"Generated {len(result['files'])} files")
            print("\n📁 Files Created:")
            print("=" * 60)
            for filename, content in result["files"].items():
                logger.debug(f"File created: {filename} ({len(content)} chars)")
                print(f"\n📄 {filename}:")
                print("-" * 40)
                print(content[:500] + "..." if len(content) > 500 else content)
        else:
            logger.info("No files were generated")
            
    except RateLimitError as e:
        logger.error(f"Rate limit error after all retries: {str(e)}")
        print("\n❌ Rate Limit Error!")
        print("The Groq API rate limit has been exceeded.")
        print("Please wait a few minutes before trying again.")
        print("Consider upgrading your Groq plan for higher rate limits.")
        
    except Exception as e:
        logger.error(f"Unexpected error during research: {str(e)}", exc_info=True)
        print(f"\n❌ Error occurred: {str(e)}")
        print("Check the log file 'research_agent.log' for more details.")

import os
import time
import logging
import json
from typing import Literal, Dict, Any, List
from functools import wraps

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tavily import TavilyClient
from groq import RateLimitError

from deepagents import create_deep_agent, SubAgent
from prompts import RESEARCH_AGENT_PROMPT, SUB_RESEARCH_PROMPT, SUB_CRITIQUE_PROMPT, CONTEXT_OPTIMIZATION_TEMPLATE
from context_optimizer import ContextOptimizer, TaskPriority

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

# Initialize context optimizer globally
context_optimizer = ContextOptimizer(max_context_size=12000, max_active_items=20)
logger.info("Initialized context optimizer for research agent")

# Search tool to use to do research with context optimization
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search with logging, error handling, and context optimization"""
    logger.info(f"Starting web search for query: '{query}' (max_results={max_results}, topic={topic})")
    
    # Check context utilization and potentially delegate to sub-agent
    context_stats = context_optimizer.get_context_stats()
    if len(query) > 100 or context_stats["utilization"] > 0.8:
        logger.info(f"Context utilization at {context_stats['utilization']:.1%}, delegating search to sub-agent")
        delegation_id = context_optimizer.delegate_to_subagent(
            f"Search for: {query}",
            "research-agent"
        )
        return {"delegated": True, "delegation_id": delegation_id, "query": query}
    
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
        
        # Add search results to context with appropriate priority
        result_content = json.dumps(search_docs, indent=2)
        priority = TaskPriority.HIGH if "error" in query.lower() else TaskPriority.MEDIUM
        
        result_id = context_optimizer.add_context_item(
            result_content,
            "result",
            priority
        )
        
        logger.info(f"Added search results to context: {result_id}")
        return search_docs
        
    except Exception as e:
        logger.error(f"Error during web search for '{query}': {str(e)}")
        
        # Add error to context with high priority
        context_optimizer.add_context_item(
            f"Search error for '{query}': {str(e)}",
            "error",
            TaskPriority.HIGH
        )
        
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

# 1️⃣ Create specialized Groq model instances for different agents
# Note: Environment variables will be loaded from .env file automatically

# Main agent: Premium model for complex reasoning and synthesis
main_model = ChatGroq(
    model="moonshotai/kimi-k2-instruct",  # Premium model for main research
    temperature=0.1,
    max_tokens=12288,  # Reduced to manage rate limits
    streaming=True,
)

# Research sub-agent: Balanced model for information gathering
research_model = ChatGroq(
    model="llama-3.1-70b-versatile",  # Good balance of speed and capability
    temperature=0.1,
    max_tokens=8192,  # Moderate token limit for research tasks
    streaming=True,
)

# Critique sub-agent: Fast model for reviews and feedback
critique_model = ChatGroq(
    model="llama-3.1-8b-instant",  # Fast model for quick feedback
    temperature=0.1,
    max_tokens=4096,  # Lower token limit for critique tasks
    streaming=True,
)

logger.info("Initialized multi-model setup:")
logger.info(f"  - Main agent: {main_model.model}")
logger.info(f"  - Research agent: {research_model.model}")
logger.info(f"  - Critique agent: {critique_model.model}")

# Function to get optimized research prompt with context awareness
def get_optimized_research_prompt():
    """Get research prompt enhanced with current context optimization status"""
    context_stats = context_optimizer.get_context_stats()
    optimized_context = context_optimizer.get_optimized_context()
    
    # Format the context optimization template with current stats
    optimization_instructions = CONTEXT_OPTIMIZATION_TEMPLATE.format(
        context_summary=context_stats['context_summary'],
        utilization=context_stats['utilization'],
        high_priority_count=len(optimized_context.get('high_priority_items', [])),
        medium_priority_count=len(optimized_context.get('medium_priority_items', []))
    )
    
    return RESEARCH_AGENT_PROMPT + optimization_instructions

# 2️⃣ Model fallback system for rate limit management
model_fallback_chain = [
    main_model,      # Try Kimi first (premium model)
    research_model,  # Fallback to Llama-3.1-70B (balanced)
    critique_model,  # Final fallback to Llama-3.1-8B (fast)
]

def get_available_model():
    """Get the first available model from fallback chain"""
    # For simplicity, we'll start with the main model and implement 
    # fallback logic in the retry mechanism
    return main_model

# Enhanced sub-agents with specialized models and rate limit awareness
enhanced_research_sub_agent = {
    "name": "research-agent",
    "description": """Used to research specific questions efficiently. 
                   This sub-agent uses Llama-3.1-70B for balanced speed and capability.
                   Only give this researcher one focused topic at a time.""",
    "prompt": SUB_RESEARCH_PROMPT + "\n\nOPTIMIZATION: Provide concise, focused research. Use bullet points for key findings.",
    "tools": ["internet_search"],
    "model": research_model  # Uses Llama-3.1-70B for efficient research
}

enhanced_critique_sub_agent = {
    "name": "critique-agent",
    "description": """Used to provide quick feedback on reports.
                   This sub-agent uses Llama-3.1-8B for fast critique tasks.""",
    "prompt": SUB_CRITIQUE_PROMPT + "\n\nOPTIMIZATION: Provide concise, bullet-pointed feedback. Focus on critical issues only.",
    "model": critique_model  # Uses Llama-3.1-8B for fast feedback
}

# 3️⃣ Create the Deep Agent with rate limit optimization
def create_optimized_agent():
    """Create agent with context optimization and rate limit management"""
    
    # Use reduced token limits to avoid rate limits
    current_model = ChatGroq(
        model=main_model.model,
        temperature=0.1,
        max_tokens=8192,  # Significantly reduced from 12288
        streaming=True,
    )
    
    logger.info("Creating rate-limit optimized deep agent:")
    logger.info(f"  - Main model: {current_model.model} (max_tokens: {current_model.max_tokens})")
    logger.info(f"  - Fallback models available: {len(model_fallback_chain)-1}")
    
    return create_deep_agent(
        [internet_search],
        get_optimized_research_prompt(),
        model=current_model,  # Use token-optimized model
        subagents=[enhanced_critique_sub_agent, enhanced_research_sub_agent],
    ).with_config({"recursion_limit": 1000})

# Create the initial agent
agent = create_optimized_agent()

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
    
    # Add research question to context with critical priority
    question_id = context_optimizer.add_context_item(
        test_question,
        "task",
        TaskPriority.CRITICAL
    )
    
    try:
        start_time = time.time()
        
        # Check context status before execution
        initial_context_stats = context_optimizer.get_context_stats()
        logger.info(f"Initial context utilization: {initial_context_stats['utilization']:.1%}")
        
        result = run_agent_with_retry([{"role": "user", "content": test_question}])
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Update context with completion
        context_optimizer.update_context_item(
            question_id,
            f"COMPLETED: {test_question}\nResult: Research finished successfully"
        )
        
        # Archive completed research task
        context_optimizer.archive_completed_tasks([question_id])
        
        # Get final context stats
        final_context_stats = context_optimizer.get_context_stats()
        
        logger.info(f"Research completed successfully in {duration:.2f} seconds")
        logger.info(f"Final context utilization: {final_context_stats['utilization']:.1%}")
        
        # Print the final response with context optimization info
        print("\n📊 Research Complete!")
        print("=" * 70)
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"🧠 Context Utilization: {final_context_stats['utilization']:.1%}")
        print(f"📁 Active Context Items: {final_context_stats['active_items']}")
        print(f"🗄️  Archived Items: {final_context_stats['archived_items']}")
        print(f"📊 Context Summary: {final_context_stats['context_summary']}")
        
        print("\n📝 Final Response:")
        print("-" * 50)
        print(result["messages"][-1].content)
        
        # Show any files that were created
        if "files" in result and result["files"]:
            logger.info(f"Generated {len(result['files'])} files")
            print("\n📄 Files Created:")
            print("=" * 60)
            for filename, content in result["files"].items():
                logger.debug(f"File created: {filename} ({len(content)} chars)")
                print(f"\n📋 {filename}:")
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

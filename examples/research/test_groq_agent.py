import os
from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tavily import TavilyClient

from deepagents import create_deep_agent
from prompts import RESEARCH_AGENT_PROMPT

# Load environment variables from .env file
load_dotenv()


# Simple search tool for testing
def internet_search(
    query: str,
    max_results: int = 3,  # Reduced to limit API calls
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    tavily_async_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    search_docs = tavily_async_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    return search_docs


# Create a Groq model instance with Kimi (with lower temperature to be more conservative)
kimi_model = ChatGroq(
    model="moonshotai/kimi-k2-instruct",
    temperature=0.0,  # Lower temperature for more consistent responses
    max_tokens=8192,  # Reduced token limit to avoid hitting limits
    streaming=False,  # Disable streaming for simpler testing
)

# Create a simple agent without sub-agents for testing
simple_agent = create_deep_agent(
    [internet_search],
    "You are a helpful research assistant. Answer questions concisely and provide sources when possible.",
    model=kimi_model,
)

if __name__ == "__main__":
    # Simple test question
    test_question = "What is LangGraph?"
    
    print(f"🧪 Testing Groq agent with: {test_question}")
    print("=" * 50)
    
    try:
        result = simple_agent.invoke({
            "messages": [{"role": "user", "content": test_question}]
        })
        
        print("✅ Success! Agent Response:")
        print("-" * 30)
        print(result["messages"][-1].content)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nThis might be due to:")
        print("1. Rate limits on Groq API (try again in a few minutes)")
        print("2. API key issues")
        print("3. Network connectivity")

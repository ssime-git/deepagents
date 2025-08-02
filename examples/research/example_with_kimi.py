import os
from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tavily import TavilyClient
from deepagents import create_deep_agent

# Load environment variables from .env file
load_dotenv()


# Search tool to use to do research
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    tavily_async_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return tavily_async_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# Prompt prefix to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
"""

# 1️⃣ Create a Groq model instance with Kimi
# Note: You'll need to set your GROQ_API_KEY environment variable
kimi_model = ChatGroq(
    model="moonshotai/kimi-k2-instruct",  # Using Kimi K2 Instruct model from Moonshot AI via Groq
    temperature=0.1,
    max_tokens=16384,  # Maximum tokens for Kimi model (must be <= 16384)
    streaming=True,    # Enable streaming if you need it
)

# 2️⃣ Create the Deep Agent with the Groq model
agent = create_deep_agent(
    [internet_search],
    research_instructions,
    model=kimi_model  # Pass the custom Groq model here instead of default Claude
)

# 3️⃣ Invoke the agent
if __name__ == "__main__":
    # Environment variables will be loaded from .env file automatically
    # Make sure your .env file contains:
    # GROQ_API_KEY=your_groq_api_key_here
    # TAVILY_API_KEY=your_tavily_api_key_here
    
    result = agent.invoke({
        "messages": [{"role": "user", "content": "what is langgraph?"}]
    })
    
    # Print the final response
    print("Agent Response:")
    print(result["messages"][-1].content)

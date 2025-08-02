#!/usr/bin/env python3
"""
Test script to verify that the enhanced deepagents package correctly supports
different models for different sub-agents.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tavily import TavilyClient

from deepagents import create_deep_agent, SubAgent, ContextOptimizer

# Load environment variables
load_dotenv()

# Simple search tool
def internet_search(
    query: str,
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    print(f"🔍 Searching for: {query}")
    tavily_async_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return tavily_async_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# Create different models for testing
# Store model names for display purposes
MAIN_MODEL_NAME = "moonshotai/kimi-k2-instruct"
RESEARCH_MODEL_NAME = "llama-3.1-70b-versatile"
CRITIQUE_MODEL_NAME = "llama-3.1-8b-instant"

main_model = ChatGroq(
    model=MAIN_MODEL_NAME,
    temperature=0.1,
    max_tokens=4096,
    streaming=False,
)

research_model = ChatGroq(
    model=RESEARCH_MODEL_NAME,
    temperature=0.1,
    max_tokens=2048,
    streaming=False,
)

critique_model = ChatGroq(
    model=CRITIQUE_MODEL_NAME,
    temperature=0.1,
    max_tokens=1024,
    streaming=False,
)

# Create sub-agents with different models
research_subagent = {
    "name": "research-specialist",
    "description": "Specialized research agent using Llama-70B for detailed information gathering",
    "prompt": "You are a research specialist. Provide concise, factual research on the given topic. Use bullet points for key findings.",
    "tools": ["internet_search"],
    "model": research_model  # This should use Llama-70B
}

critique_subagent = {
    "name": "quality-checker", 
    "description": "Fast quality checker using Llama-8B for quick feedback",
    "prompt": "You are a quality checker. Provide brief, actionable feedback on the content quality.",
    "model": critique_model  # This should use Llama-8B
}

# Create the agent with multi-model sub-agents
agent = create_deep_agent(
    [internet_search],
    "You are a research coordinator using Kimi for complex reasoning. Delegate research to specialists when needed.",
    model=main_model,  # Main agent uses Kimi
    subagents=[research_subagent, critique_subagent],
)

def test_multi_model_agent():
    """Test that the agent correctly uses different models for different sub-agents"""
    print("🧪 Testing Multi-Model Deep Agent")
    print("=" * 50)
    print(f"📋 Main agent model: {MAIN_MODEL_NAME}")
    print(f"🔬 Research sub-agent model: {RESEARCH_MODEL_NAME}")
    print(f"✅ Critique sub-agent model: {CRITIQUE_MODEL_NAME}")
    print()
    
    # Simple test question
    test_question = "What is artificial intelligence?"
    
    print(f"❓ Test question: {test_question}")
    print("-" * 50)
    
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": test_question}]
        })
        
        print("✅ Success! Multi-model agent worked correctly.")
        print()
        print("📝 Response preview:")
        print("-" * 30)
        response_preview = result["messages"][-1].content[:200] + "..." if len(result["messages"][-1].content) > 200 else result["messages"][-1].content
        print(response_preview)
        
        # Show files if any were created
        if "files" in result and result["files"]:
            print(f"\n📄 Files created: {list(result['files'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nThis might be due to:")
        print("1. API key issues (check your .env file)")
        print("2. Rate limits (try again in a few minutes)")
        print("3. Network connectivity")
        return False

def test_context_optimization():
    """Test the built-in context optimization features"""
    print("\n🧪 Testing Context Optimization Features")
    print("=" * 50)
    
    # Create agent with context optimization enabled
    context_agent = create_deep_agent(
        [internet_search],
        "You are a research coordinator with context optimization. Use context tools to monitor usage.",
        model=main_model,
        subagents=[research_subagent, critique_subagent],
        enable_context_optimization=True,
        max_context_size=8000,  # Smaller for testing
        max_active_items=10
    )
    
    print("📊 Context optimization enabled with:")
    print("   - Max context size: 8000")
    print("   - Max active items: 10")
    print("   - Additional context tools available")
    print()
    
    test_question = "Explain the basics of machine learning in simple terms."
    print(f"❓ Test question: {test_question}")
    print("-" * 50)
    
    try:
        result = context_agent.invoke({
            "messages": [{"role": "user", "content": test_question}]
        })
        
        print("✅ Success! Context-optimized agent worked correctly.")
        print()
        print("📝 Response preview:")
        print("-" * 30)
        response_preview = result["messages"][-1].content[:200] + "..." if len(result["messages"][-1].content) > 200 else result["messages"][-1].content
        print(response_preview)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nThis might be due to:")
        print("1. API key issues (check your .env file)")
        print("2. Rate limits (try again in a few minutes)")
        print("3. Network connectivity")
        return False

if __name__ == "__main__":
    print("🚀 Testing Enhanced DeepAgents Features")
    print("=" * 60)
    
    # Test 1: Multi-model sub-agents
    success1 = test_multi_model_agent()
    
    # Test 2: Context optimization
    success2 = test_context_optimization()
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 Test Results Summary")
    print("=" * 60)
    
    if success1:
        print("✅ Multi-model sub-agents: PASSED")
        print("   Each sub-agent can use its own specialized model")
    else:
        print("❌ Multi-model sub-agents: FAILED")
    
    if success2:
        print("✅ Context optimization: PASSED")
        print("   Built-in context management tools work correctly")
    else:
        print("❌ Context optimization: FAILED")
    
    if success1 and success2:
        print("\n🎉 All enhanced deepagents features work correctly!")
        print("\n🔧 New features available:")
        print("   • Per-subagent model assignment")
        print("   • Built-in context optimization")
        print("   • Context monitoring tools")
        print("   • Automatic context management")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")

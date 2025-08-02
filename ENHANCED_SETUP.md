# Enhanced DeepAgents Setup Guide

This guide helps you set up and use the enhanced deepagents package with the new multi-model and context optimization features using uv.

## 🆕 New Features in v0.1.0

- **Per-SubAgent Model Assignment**: Each sub-agent can use a different LLM model
- **Built-in Context Optimization**: Automatic context window management with priority-based task handling
- **Context Monitoring Tools**: Built-in tools for tracking and managing context utilization
- **Enhanced Rate Limit Management**: Better handling of API rate limits across multiple models

## 🚀 Quick Start with uv

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Build Enhanced DeepAgents from Source

```bash
# Navigate to the enhanced deepagents repository
cd /Users/sebastiensime/Documents/deepagents

# Build and install the enhanced package using uv
uv sync
uv build

# Install in development mode for easy updates
uv pip install -e .
```

### 3. Install Research Example Dependencies

```bash
cd examples/research
uv add langchain-groq python-dotenv tavily-python langgraph-cli groq
```

### 4. Environment Setup

Create a `.env` file in the research directory:

```bash
# examples/research/.env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## 🧪 Test the Enhanced Features

```bash
# Test the multi-model and context optimization features
cd /Users/sebastiensime/Documents/deepagents
uv run test_multi_model.py
```

## 🔧 Usage Examples

### Basic Multi-Model Agent

```python
from deepagents import create_deep_agent
from langchain_groq import ChatGroq

# Define different models for different tasks
main_model = ChatGroq(model="moonshotai/kimi-k2-instruct")
research_model = ChatGroq(model="llama-3.1-70b-versatile") 
critique_model = ChatGroq(model="llama-3.1-8b-instant")

# Create sub-agents with specialized models
research_subagent = {
    "name": "researcher",
    "description": "Specialized research agent",
    "prompt": "You are a research specialist...",
    "model": research_model  # Uses Llama-70B
}

critique_subagent = {
    "name": "critic", 
    "description": "Quality checker",
    "prompt": "You are a quality checker...",
    "model": critique_model  # Uses Llama-8B
}

# Create agent with different models per sub-agent
agent = create_deep_agent(
    tools=[your_tools],
    instructions="You are a coordinator...",
    model=main_model,  # Main agent uses Kimi
    subagents=[research_subagent, critique_subagent]
)
```

### Context-Optimized Agent

```python
from deepagents import create_deep_agent, ContextOptimizer

# Option 1: Enable built-in context optimization
agent = create_deep_agent(
    tools=[your_tools],
    instructions="Your instructions...",
    enable_context_optimization=True,
    max_context_size=15000,
    max_active_items=25
)

# Option 2: Use custom context optimizer
custom_optimizer = ContextOptimizer(
    max_context_size=20000, 
    max_active_items=30
)

agent = create_deep_agent(
    tools=[your_tools],
    instructions="Your instructions...",
    context_optimizer=custom_optimizer
)
```

## 🏗️ Development with uv

### Building and Testing

```bash
# Sync dependencies
uv sync

# Build the package
uv build

# Run tests
uv run python test_multi_model.py

# Run research examples
cd examples/research
uv run python research_agent_groq.py
```

### Adding Dependencies

```bash
# Add core dependencies to the base package
uv add langchain langgraph langchain-anthropic

# Add example-specific dependencies
cd examples/research
uv add langchain-groq python-dotenv tavily-python groq
```

## 📁 Enhanced Package Structure

```
deepagents/
├── src/deepagents/
│   ├── __init__.py              # Exports enhanced features
│   ├── context_optimizer.py     # Context management (NEW)
│   ├── context_tools.py         # Context monitoring tools (NEW)
│   ├── graph.py                 # Enhanced create_deep_agent (UPDATED)
│   ├── sub_agent.py             # Multi-model support (UPDATED)
│   └── ...                      # Other core files
├── examples/research/
│   ├── requirements.txt         # Research-specific deps
│   ├── research_agent_groq.py   # Full-featured example
│   └── test_multi_model.py      # Comprehensive tests
├── pyproject.toml               # Clean base dependencies
└── uv.lock                      # Dependency lock file
```

## 🔍 Key Differences from Base Package

### Core Package (lightweight)
- Basic deepagents functionality + new features
- Standard dependencies (langchain, langgraph, langchain-anthropic)
- Context optimization available but optional

### Research Examples (enhanced)
- Multi-model demonstrations (Groq/Kimi/Llama)
- Context optimization in action
- Rate limit management examples
- Research-specific prompts and tools

## 🚨 Requirements

- Python >= 3.11
- uv package manager
- API keys for Groq and Tavily (for research examples)

## 🛠️ uv Commands Cheat Sheet

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync project dependencies
uv sync

# Build the package
uv build

# Install in development mode
uv pip install -e .

# Add a dependency
uv add package-name

# Run a script
uv run script.py

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # or uv shell
```

---

🎉 **Ready to build powerful multi-model deep agents with intelligent context management using uv!**

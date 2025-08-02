from deepagents.sub_agent import _create_task_tool, SubAgent
from deepagents.model import get_default_model
from deepagents.tools import write_todos, write_file, read_file, ls, edit_file
from deepagents.state import DeepAgentState
from deepagents.context_optimizer import ContextOptimizer
from deepagents.context_tools import create_context_tools
from typing import Sequence, Union, Callable, Any, TypeVar, Type, Optional
from langchain_core.tools import BaseTool
from langchain_core.language_models import LanguageModelLike

from langgraph.prebuilt import create_react_agent

StateSchema = TypeVar("StateSchema", bound=DeepAgentState)
StateSchemaType = Type[StateSchema]

base_prompt = """You have access to a number of standard tools

## `write_todos`

You have access to the `write_todos` tools to help you manage and plan tasks. Use these tools VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.
These tools are also EXTREMELY helpful for planning tasks, and for breaking down larger complex tasks into smaller steps. If you do not use this tool when planning, you may forget to do important tasks - and that is unacceptable.

It is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.
## `task`

- When doing web search, prefer to use the `task` tool in order to reduce context usage."""


def create_deep_agent(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]],
    instructions: str,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: list[SubAgent] = None,
    state_schema: Optional[StateSchemaType] = None,
    context_optimizer: Optional[ContextOptimizer] = None,
    enable_context_optimization: bool = False,
    max_context_size: int = 12000,
    max_active_items: int = 20,
):
    """Create a deep agent.

    This agent will by default have access to a tool to write todos (write_todos),
    and then four file editing tools: write_file, ls, read_file, edit_file.

    Args:
        tools: The additional tools the agent should have access to.
        instructions: The additional instructions the agent should have. Will go in
            the system prompt.
        model: The model to use.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
                - (optional) `model` (model for this specific sub-agent)
        state_schema: The schema of the deep agent. Should subclass from DeepAgentState
        context_optimizer: Optional pre-configured context optimizer instance.
        enable_context_optimization: Whether to enable built-in context optimization.
        max_context_size: Maximum context size for optimization (used if enable_context_optimization=True).
        max_active_items: Maximum active items for optimization (used if enable_context_optimization=True).
    """
    # Initialize context optimizer if requested
    if enable_context_optimization and context_optimizer is None:
        context_optimizer = ContextOptimizer(
            max_context_size=max_context_size,
            max_active_items=max_active_items
        )
    
    # Add context optimization instructions to prompt if enabled
    if context_optimizer:
        context_optimization_prompt = """

CONTEXT OPTIMIZATION ENABLED:
- Monitor context utilization and delegate to sub-agents when context > 80% full
- Prioritize tasks: CRITICAL > HIGH > MEDIUM > LOW
- Archive completed tasks to preserve context space
- Use incremental updates for large tasks
- Focus on synthesis rather than new research when context is full
"""
        prompt = instructions + context_optimization_prompt + base_prompt
    else:
        prompt = instructions + base_prompt
    
    built_in_tools = [write_todos, write_file, read_file, ls, edit_file]
    
    # Add context optimization tools if enabled
    context_tools = create_context_tools(context_optimizer)
    if context_optimizer:
        built_in_tools.extend(context_tools)
    
    if model is None:
        model = get_default_model()
    state_schema = state_schema or DeepAgentState
    
    # Pass context optimizer to task tool if available
    task_tool = _create_task_tool(
        list(tools) + built_in_tools,
        instructions,
        subagents or [],
        model,
        state_schema,
        context_optimizer
    )
    all_tools = built_in_tools + list(tools) + [task_tool]
    return create_react_agent(
        model,
        prompt=prompt,
        tools=all_tools,
        state_schema=state_schema,
    )

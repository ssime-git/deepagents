"""
Context-aware tools for deepagents that integrate with the ContextOptimizer
"""

from langchain_core.tools import tool
from deepagents.context_optimizer import ContextOptimizer, TaskPriority
from typing import Literal, Optional

def create_context_tools(context_optimizer: Optional[ContextOptimizer] = None):
    """Create context optimization tools for use within deepagents"""
    
    if context_optimizer is None:
        # Create dummy tools that do nothing if no context optimizer is provided
        @tool
        def get_context_stats() -> str:
            """Get current context utilization statistics (context optimization disabled)"""
            return "Context optimization is not enabled for this agent."
        
        @tool 
        def add_context_item(
            content: str,
            item_type: str = "general",
            priority: Literal["critical", "high", "medium", "low"] = "medium"
        ) -> str:
            """Add an item to context tracking (context optimization disabled)"""
            return "Context optimization is not enabled for this agent."
        
        @tool
        def archive_completed_tasks(task_ids: str = "") -> str:
            """Archive completed tasks to free up context space (context optimization disabled)"""
            return "Context optimization is not enabled for this agent."
            
        return [get_context_stats, add_context_item, archive_completed_tasks]
    
    # Create functional context tools
    @tool
    def get_context_stats() -> str:
        """Get current context utilization statistics and optimization recommendations"""
        stats = context_optimizer.get_context_stats()
        return f"""Context Statistics:
- Utilization: {stats['utilization']:.1%}
- Active items: {stats['active_items']}
- Archived items: {stats['archived_items']}
- Summary: {stats['context_summary']}

Recommendations:
{_get_context_recommendations(stats['utilization'])}"""

    @tool
    def add_context_item(
        content: str,
        item_type: str = "general", 
        priority: Literal["critical", "high", "medium", "low"] = "medium"
    ) -> str:
        """Add an item to context tracking with specified priority"""
        priority_map = {
            "critical": TaskPriority.CRITICAL,
            "high": TaskPriority.HIGH,
            "medium": TaskPriority.MEDIUM,
            "low": TaskPriority.LOW
        }
        
        item_id = context_optimizer.add_context_item(
            content=content,
            item_type=item_type,
            priority=priority_map[priority]
        )
        
        stats = context_optimizer.get_context_stats()
        return f"Added context item {item_id} with {priority} priority. Context utilization: {stats['utilization']:.1%}"

    @tool
    def archive_completed_tasks(task_ids: str = "") -> str:
        """Archive completed tasks to free up context space. Provide comma-separated task IDs or leave empty to archive all completed tasks."""
        if task_ids.strip():
            # Parse comma-separated task IDs
            ids = [tid.strip() for tid in task_ids.split(",")]
            archived_count = context_optimizer.archive_completed_tasks(ids)
        else:
            # Archive all completed tasks
            optimized_context = context_optimizer.get_optimized_context()
            completed_items = [
                item.id for item in optimized_context.get("completed_items", [])
            ]
            archived_count = context_optimizer.archive_completed_tasks(completed_items)
        
        stats = context_optimizer.get_context_stats()
        return f"Archived {archived_count} completed tasks. Context utilization: {stats['utilization']:.1%}"

    return [get_context_stats, add_context_item, archive_completed_tasks]

def _get_context_recommendations(utilization: float) -> str:
    """Get context optimization recommendations based on utilization"""
    if utilization < 0.5:
        return "- Context usage is low. Continue with current tasks."
    elif utilization < 0.8:
        return "- Context usage is moderate. Consider delegating complex sub-tasks to sub-agents."
    else:
        return """- Context usage is high! Recommended actions:
  1. Archive completed tasks immediately
  2. Delegate new complex tasks to sub-agents
  3. Focus on synthesis rather than new research
  4. Use incremental updates for large tasks"""

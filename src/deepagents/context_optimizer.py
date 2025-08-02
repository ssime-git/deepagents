"""
Context Window Optimization for Deep Agents

This module implements strategies to optimize context window usage:
1. Task Prioritization
2. State Management Enhancements (Segmentation & Serialization)
4. Sub-Agent Delegation
5. Incremental State Updates
6. Archived Context Handling
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class TaskPriority:
    """Task priority levels for context optimization"""
    CRITICAL = 1    # Must be in active context
    HIGH = 2       # Should be in active context
    MEDIUM = 3     # Can be deferred or summarized
    LOW = 4        # Can be archived or delegated

@dataclass
class ContextItem:
    """Represents an item in the context with metadata"""
    id: str
    content: str
    priority: int
    timestamp: float
    type: Literal["task", "state", "result", "error"]
    size_estimate: int
    last_accessed: float
    access_count: int = 0
    
    def __post_init__(self):
        if not self.size_estimate:
            self.size_estimate = len(str(self.content))

@dataclass
class ContextSummary:
    """Summary of archived context"""
    id: str
    summary: str
    original_items: List[str]  # IDs of original items
    created_at: float
    context_type: str

class ContextOptimizer:
    """Optimizes context window usage for deep agents"""
    
    def __init__(self, max_context_size: int = 12000, max_active_items: int = 20):
        self.max_context_size = max_context_size
        self.max_active_items = max_active_items
        
        # Active context (kept in memory)
        self.active_context: Dict[str, ContextItem] = {}
        
        # Archived context (serialized to files)
        self.archived_context: Dict[str, ContextSummary] = {}
        
        # Delta tracking for incremental updates
        self.context_deltas: List[Dict[str, Any]] = []
        
        # Priority weights for different item types
        self.type_priority_weights = {
            "task": 1.0,      # Tasks are important
            "state": 0.8,     # State is somewhat important
            "result": 0.6,    # Results can be summarized
            "error": 1.2      # Errors need immediate attention
        }
    
    def add_context_item(self, content: str, item_type: str, priority: int = TaskPriority.MEDIUM) -> str:
        """Add a new item to the context with optimization"""
        # Generate unique ID
        item_id = hashlib.md5(f"{content}_{time.time()}".encode()).hexdigest()[:8]
        
        # Create context item
        item = ContextItem(
            id=item_id,
            content=content,
            priority=priority,
            timestamp=time.time(),
            type=item_type,
            size_estimate=len(content),
            last_accessed=time.time()
        )
        
        # Add to active context
        self.active_context[item_id] = item
        
        # Record delta
        self._record_delta("add", item_id, content)
        
        # Optimize context if needed
        self._optimize_context()
        
        logger.info(f"Added context item {item_id} (type: {item_type}, priority: {priority}, size: {len(content)})")
        return item_id
    
    def update_context_item(self, item_id: str, new_content: str) -> bool:
        """Update existing context item (incremental update)"""
        if item_id not in self.active_context:
            # Try to restore from archive if needed
            if not self._restore_from_archive(item_id):
                return False
        
        old_content = self.active_context[item_id].content
        self.active_context[item_id].content = new_content
        self.active_context[item_id].size_estimate = len(new_content)
        self.active_context[item_id].last_accessed = time.time()
        self.active_context[item_id].access_count += 1
        
        # Record delta (only the change)
        self._record_delta("update", item_id, new_content, old_content)
        
        logger.info(f"Updated context item {item_id}")
        return True
    
    def access_context_item(self, item_id: str) -> Optional[str]:
        """Access a context item, updating access metrics"""
        if item_id in self.active_context:
            item = self.active_context[item_id]
            item.last_accessed = time.time()
            item.access_count += 1
            return item.content
        
        # Try to restore from archive
        if self._restore_from_archive(item_id):
            return self.active_context[item_id].content
            
        return None
    
    def get_optimized_context(self) -> Dict[str, Any]:
        """Get the current optimized context for the agent"""
        # Sort by priority and recency
        sorted_items = sorted(
            self.active_context.values(),
            key=lambda x: (x.priority, -x.last_accessed, -x.access_count)
        )
        
        # Build optimized context
        context = {
            "high_priority_items": [],
            "medium_priority_items": [],
            "recent_deltas": self.context_deltas[-10:],  # Last 10 deltas
            "context_summary": self._generate_context_summary()
        }
        
        current_size = 0
        for item in sorted_items:
            if current_size + item.size_estimate > self.max_context_size:
                break
            
            if item.priority <= TaskPriority.HIGH:
                context["high_priority_items"].append({
                    "id": item.id,
                    "content": item.content,
                    "type": item.type
                })
            else:
                context["medium_priority_items"].append({
                    "id": item.id,
                    "content": item.content[:200] + "..." if len(item.content) > 200 else item.content,
                    "type": item.type
                })
            
            current_size += item.size_estimate
        
        logger.info(f"Generated optimized context: {len(context['high_priority_items'])} high priority, "
                   f"{len(context['medium_priority_items'])} medium priority items")
        
        return context
    
    def delegate_to_subagent(self, task_content: str, agent_type: str = "general-purpose") -> str:
        """Delegate a task to a sub-agent to reduce main context load"""
        delegation_id = hashlib.md5(f"delegation_{task_content}_{time.time()}".encode()).hexdigest()[:8]
        
        # Create delegation record
        delegation_item = {
            "id": delegation_id,
            "type": "delegation",
            "agent_type": agent_type,
            "task": task_content,
            "status": "delegated",
            "created_at": time.time()
        }
        
        # Add to context but with low priority (will be archived quickly)
        self.add_context_item(
            json.dumps(delegation_item),
            "delegation",
            TaskPriority.LOW
        )
        
        logger.info(f"Delegated task to {agent_type} sub-agent: {delegation_id}")
        return delegation_id
    
    def archive_completed_tasks(self, task_ids: List[str]) -> None:
        """Archive completed tasks to free up context space"""
        for task_id in task_ids:
            if task_id in self.active_context:
                item = self.active_context[task_id]
                
                # Create summary
                summary = ContextSummary(
                    id=f"summary_{task_id}",
                    summary=f"Completed {item.type}: {item.content[:100]}...",
                    original_items=[task_id],
                    created_at=time.time(),
                    context_type=item.type
                )
                
                # Move to archive
                self.archived_context[task_id] = summary
                del self.active_context[task_id]
                
                # Record delta
                self._record_delta("archive", task_id, summary.summary)
                
                logger.info(f"Archived completed task: {task_id}")
    
    def prioritize_tasks(self, task_priorities: Dict[str, int]) -> None:
        """Update task priorities for context optimization"""
        for task_id, priority in task_priorities.items():
            if task_id in self.active_context:
                old_priority = self.active_context[task_id].priority
                self.active_context[task_id].priority = priority
                
                if old_priority != priority:
                    logger.info(f"Updated priority for {task_id}: {old_priority} -> {priority}")
    
    def _optimize_context(self) -> None:
        """Internal method to optimize context when it gets too large"""
        total_size = sum(item.size_estimate for item in self.active_context.values())
        
        if total_size <= self.max_context_size and len(self.active_context) <= self.max_active_items:
            return
        
        # Sort items by optimization score (priority, age, access frequency)
        items_to_optimize = []
        for item in self.active_context.values():
            # Calculate optimization score (higher = more likely to be archived)
            age_factor = time.time() - item.last_accessed
            priority_factor = item.priority * self.type_priority_weights.get(item.type, 1.0)
            access_factor = 1.0 / (item.access_count + 1)
            
            score = (age_factor * priority_factor * access_factor) / 1000
            items_to_optimize.append((score, item))
        
        # Sort by score (highest first = most likely to archive)
        items_to_optimize.sort(key=lambda x: x[0], reverse=True)
        
        # Archive items until we're under limits
        items_archived = 0
        for score, item in items_to_optimize:
            if total_size <= self.max_context_size and len(self.active_context) <= self.max_active_items:
                break
            
            # Don't archive critical items
            if item.priority == TaskPriority.CRITICAL:
                continue
            
            # Create summary and archive
            summary = ContextSummary(
                id=f"auto_summary_{item.id}",
                summary=f"Auto-archived {item.type}: {item.content[:100]}...",
                original_items=[item.id],
                created_at=time.time(),
                context_type=item.type
            )
            
            self.archived_context[item.id] = summary
            total_size -= item.size_estimate
            del self.active_context[item.id]
            items_archived += 1
            
            self._record_delta("auto_archive", item.id, summary.summary)
        
        if items_archived > 0:
            logger.info(f"Auto-archived {items_archived} context items to optimize context window")
    
    def _restore_from_archive(self, item_id: str) -> bool:
        """Restore an item from archive to active context"""
        if item_id not in self.archived_context:
            return False
        
        summary = self.archived_context[item_id]
        
        # Create a placeholder item (we can't fully restore without original content)
        restored_item = ContextItem(
            id=item_id,
            content=f"[RESTORED] {summary.summary}",
            priority=TaskPriority.MEDIUM,
            timestamp=summary.created_at,
            type=summary.context_type,
            size_estimate=len(summary.summary),
            last_accessed=time.time()
        )
        
        self.active_context[item_id] = restored_item
        del self.archived_context[item_id]
        
        logger.info(f"Restored item from archive: {item_id}")
        return True
    
    def _record_delta(self, operation: str, item_id: str, content: str, old_content: str = None) -> None:
        """Record a delta change for incremental updates"""
        delta = {
            "operation": operation,
            "item_id": item_id,
            "timestamp": time.time(),
            "content": content
        }
        
        if old_content and operation == "update":
            # Only record the actual change
            delta["change"] = f"Changed from: {old_content[:50]}... to: {content[:50]}..."
        
        self.context_deltas.append(delta)
        
        # Keep only recent deltas
        if len(self.context_deltas) > 50:
            self.context_deltas = self.context_deltas[-30:]
    
    def _generate_context_summary(self) -> str:
        """Generate a high-level summary of the current context"""
        total_items = len(self.active_context)
        archived_items = len(self.archived_context)
        
        type_counts = {}
        priority_counts = {}
        
        for item in self.active_context.values():
            type_counts[item.type] = type_counts.get(item.type, 0) + 1
            priority_counts[item.priority] = priority_counts.get(item.priority, 0) + 1
        
        summary = f"Context: {total_items} active items, {archived_items} archived. "
        summary += f"Types: {dict(type_counts)}. Priorities: {dict(priority_counts)}"
        
        return summary
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about context usage"""
        total_size = sum(item.size_estimate for item in self.active_context.values())
        
        return {
            "active_items": len(self.active_context),
            "archived_items": len(self.archived_context),
            "total_size": total_size,
            "max_size": self.max_context_size,
            "utilization": total_size / self.max_context_size,
            "recent_deltas": len(self.context_deltas),
            "context_summary": self._generate_context_summary()
        }

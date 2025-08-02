import os
import time
import logging
import json
import hashlib
from difflib import SequenceMatcher
from typing import Literal, Dict, Any, List, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tavily import TavilyClient
from groq import RateLimitError

from deepagents import create_deep_agent, SubAgent, ContextOptimizer, TaskPriority
from prompts import RESEARCH_AGENT_PROMPT, SUB_RESEARCH_PROMPT, SUB_CRITIQUE_PROMPT

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_agent_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rate limiting configuration
RATE_LIMIT_DELAY = 60
MAX_RETRIES = 3
BASE_DELAY = 1

class OptimizedResearchAgent:
    """Research agent with context window optimization"""
    
    def __init__(self, max_context_size: int = 12000):
        # Initialize context optimizer
        self.context_optimizer = ContextOptimizer(max_context_size=max_context_size)
        
        # 🆕 Initialize multiple Groq models for different tasks to distribute load
        # Main agent: Premium model for complex reasoning and synthesis
        self.kimi_model = ChatGroq(
            model="moonshotai/kimi-k2-instruct",
            temperature=0.1,
            max_tokens=8192,  # Reduced to manage rate limits
            streaming=True,
        )
        
        # Research sub-agent: Balanced model for information gathering
        self.research_model = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0.1,
            max_tokens=6144,  # Moderate token limit for research tasks
            streaming=True,
        )
        
        # Critique sub-agent: Fast model for reviews and feedback
        self.critique_model = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=4096,  # Lower token limit for critique tasks
            streaming=True,
        )
        
        # Track active delegations
        self.active_delegations: Dict[str, Dict[str, Any]] = {}
        
        # Track research session state
        self.session_state = {
            "current_question": None,
            "research_phase": "planning",  # planning, researching, writing, reviewing
            "completed_subtasks": [],
            "pending_subtasks": [],
            "current_priority_task": None
        }
        
        # 🆕 Initialize search memory system
        self.search_memory = {
            "queries": {},  # query_hash -> {query, results, timestamp}
            "results_cache": {},  # result_hash -> cached_results
            "similarity_threshold": 0.75  # Minimum similarity to consider queries similar
        }
        
        logger.info("Initialized OptimizedResearchAgent with context optimization")
        # Store model names for display (since .model attribute isn't accessible)
        self.main_model_name = "moonshotai/kimi-k2-instruct"
        self.research_model_name = "llama-3.1-70b-versatile"
        self.critique_model_name = "llama-3.1-8b-instant"
        
        logger.info("Multi-model setup:")
        logger.info(f"  - Main agent: {self.main_model_name} (max_tokens: {self.kimi_model.max_tokens})")
        logger.info(f"  - Research sub-agent: {self.research_model_name} (max_tokens: {self.research_model.max_tokens})")
        logger.info(f"  - Critique sub-agent: {self.critique_model_name} (max_tokens: {self.critique_model.max_tokens})")
        logger.info("Search memory system initialized for deduplication")
    
    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two search queries"""
        # Normalize queries
        q1_clean = query1.lower().strip()
        q2_clean = query2.lower().strip()
        
        # Use SequenceMatcher for similarity
        return SequenceMatcher(None, q1_clean, q2_clean).ratio()
    
    def _get_query_hash(self, query: str) -> str:
        """Generate a hash for the query for caching"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()[:8]
    
    def _find_similar_cached_search(self, query: str) -> Optional[Dict[str, Any]]:
        """Find if we have a similar search in cache"""
        query_normalized = query.lower().strip()
        
        for cached_hash, cached_data in self.search_memory["queries"].items():
            cached_query = cached_data["query"]
            similarity = self._calculate_query_similarity(query_normalized, cached_query)
            
            if similarity >= self.search_memory["similarity_threshold"]:
                # Check if cache is still fresh (within 1 hour)
                cache_age = time.time() - cached_data["timestamp"]
                if cache_age < 3600:  # 1 hour cache
                    logger.info(f"🔄 Found similar cached search (similarity: {similarity:.2f}): '{cached_query}'")
                    return cached_data["results"]
                else:
                    logger.info(f"⏰ Cached search expired (age: {cache_age/60:.1f}min): '{cached_query}'")
        
        return None
    
    def _cache_search_result(self, query: str, results: Dict[str, Any]):
        """Cache search results for future deduplication"""
        query_hash = self._get_query_hash(query)
        
        self.search_memory["queries"][query_hash] = {
            "query": query.lower().strip(),
            "results": results,
            "timestamp": time.time()
        }
        
        # Limit cache size to prevent memory issues
        if len(self.search_memory["queries"]) > 50:
            # Remove oldest entries
            oldest_key = min(self.search_memory["queries"].keys(), 
                           key=lambda k: self.search_memory["queries"][k]["timestamp"])
            del self.search_memory["queries"][oldest_key]
            logger.info("🗑️ Removed oldest search cache entry to manage memory")
    
    def internet_search_optimized(
        self,
        query: str,
        max_results: int = 3,  # Reduced for context optimization
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
    ):
        """Optimized internet search with deduplication and context management"""
        logger.info(f"Starting optimized web search for: '{query}'")
        
        # 🆕 STEP 1: Check for similar cached searches first
        cached_result = self._find_similar_cached_search(query)
        if cached_result:
            logger.info(f"💾 Using cached search result - saved API call!")
            
            # Add cached results to context
            result_content = json.dumps(cached_result, indent=2)
            priority = TaskPriority.HIGH if "error" in query.lower() else TaskPriority.MEDIUM
            
            result_id = self.context_optimizer.add_context_item(
                f"CACHED: {result_content}",
                "result",
                priority
            )
            
            return cached_result
        
        # Check if we should delegate this search to a sub-agent
        if len(query) > 100 or self.context_optimizer.get_context_stats()["utilization"] > 0.8:
            logger.info("Delegating complex search to sub-agent to preserve context")
            delegation_id = self.context_optimizer.delegate_to_subagent(
                f"Search for: {query}",
                "research-agent"
            )
            return {"delegated": True, "delegation_id": delegation_id, "query": query}
        
        try:
            # 🆕 STEP 2: Perform new search only if not cached
            logger.info(f"🌐 Performing new search (not in cache)")
            tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            search_docs = tavily_client.search(
                query,
                max_results=max_results,
                include_raw_content=include_raw_content,
                topic=topic,
            )
            
            # 🆕 STEP 3: Cache the new search result
            self._cache_search_result(query, search_docs)
            
            # Add search results to context with appropriate priority
            result_content = json.dumps(search_docs, indent=2)
            priority = TaskPriority.HIGH if "error" in query.lower() else TaskPriority.MEDIUM
            
            result_id = self.context_optimizer.add_context_item(
                result_content,
                "result",
                priority
            )
            
            logger.info(f"Search completed: {len(search_docs.get('results', []))} results, context_id: {result_id}")
            return search_docs
            
        except Exception as e:
            logger.error(f"Search error for '{query}': {str(e)}")
            # Add error to context with high priority
            self.context_optimizer.add_context_item(
                f"Search error for '{query}': {str(e)}",
                "error",
                TaskPriority.HIGH
            )
            return {"results": [], "error": str(e)}
    
    def create_optimized_agent(self):
        """Create the deep agent with optimization enhancements using built-in context optimization"""
        
        # 🚀 Enhanced sub-agents with different models for load distribution
        research_sub_agent = {
            "name": "research-agent",
            "description": """Used for focused research on specific topics using Llama-70B. 
                           This agent works in isolation to prevent context pollution.
                           Use for: complex searches, detailed analysis, fact-checking.""",
            "prompt": SUB_RESEARCH_PROMPT + """
            
            CONTEXT OPTIMIZATION INSTRUCTIONS:
            - Focus only on the specific research task given
            - Provide concise, structured responses
            - Summarize key findings at the end
            - Avoid repeating information unless critical
            """,
            "tools": ["internet_search_optimized"],
            "model": self.research_model  # 🆕 Uses Llama-3.1-70B for research tasks
        }
        
        critique_sub_agent = {
            "name": "critique-agent", 
            "description": """Used to review and critique research reports using Llama-8B.
                           Works independently to provide objective feedback.""",
            "prompt": SUB_CRITIQUE_PROMPT + """
            
            CONTEXT OPTIMIZATION INSTRUCTIONS:
            - Focus on actionable feedback only
            - Prioritize critical issues over minor improvements
            - Provide structured, bullet-pointed critiques
            - Suggest specific improvements with examples
            """,
            "model": self.critique_model  # 🆕 Uses Llama-3.1-8B for fast feedback
        }
        
        # 🆕 NEW: Use built-in context optimization instead of manual management
        agent = create_deep_agent(
            [self.internet_search_optimized],
            self._get_optimized_prompt(),
            model=self.kimi_model,
            subagents=[critique_sub_agent, research_sub_agent],
            context_optimizer=self.context_optimizer,  # 🆕 Pass existing optimizer
            enable_context_optimization=True,  # 🆕 Enable built-in features
            max_context_size=self.context_optimizer.max_context_size,
            max_active_items=self.context_optimizer.max_active_items
        ).with_config({"recursion_limit": 1000})
        
        logger.info("Created agent with built-in context optimization enabled")
        return agent
    
    def _get_optimized_prompt(self) -> str:
        """Get the research prompt enhanced with context optimization instructions"""
        
        context_stats = self.context_optimizer.get_context_stats()
        optimized_context = self.context_optimizer.get_optimized_context()
        
        base_prompt = RESEARCH_AGENT_PROMPT
        
        optimization_instructions = f"""
        
        CONTEXT OPTIMIZATION GUIDELINES:
        
        Current Context Status: {context_stats['context_summary']}
        Context Utilization: {context_stats['utilization']:.1%}
        
        PRIORITY MANAGEMENT:
        - Mark urgent tasks as CRITICAL priority
        - Use HIGH priority for current active tasks
        - Set MEDIUM priority for planned tasks
        - Use LOW priority for completed task summaries
        
        DELEGATION STRATEGY:
        - Delegate detailed research to research-agent when context > 80% full
        - Use critique-agent for report reviews to isolate feedback
        - Delegate routine searches when handling complex reasoning
        
        CONTEXT PRESERVATION:
        - Focus on current high-priority items: {len(optimized_context.get('high_priority_items', []))} items
        - Medium priority items available: {len(optimized_context.get('medium_priority_items', []))} items
        - Archive completed tasks immediately after verification
        
        INCREMENTAL PROGRESS:
        - Update task status frequently (in_progress -> completed)
        - Record key findings incrementally, not in large blocks
        - Summarize previous work when revisiting archived context
        
        When context utilization exceeds 80%, prioritize:
        1. Complete current high-priority tasks
        2. Archive finished work
        3. Delegate new complex tasks to sub-agents
        4. Focus on synthesis rather than new research
        """
        
        return base_prompt + optimization_instructions
    
    def conduct_research(self, question: str) -> Dict[str, Any]:
        """Conduct research with context optimization"""
        logger.info(f"Starting optimized research on: {question}")
        
        # Initialize session
        self.session_state["current_question"] = question
        self.session_state["research_phase"] = "planning"
        
        # Add research question to context with critical priority
        question_id = self.context_optimizer.add_context_item(
            question,
            "task",
            TaskPriority.CRITICAL
        )
        
        # Create optimized agent
        agent = self.create_optimized_agent()
        
        try:
            start_time = time.time()
            
            # Phase 1: Planning with context optimization
            self._update_research_phase("planning")
            planning_result = self._execute_with_context_management(
                agent, 
                f"Plan research approach for: {question}",
                TaskPriority.HIGH
            )
            
            # Phase 2: Research execution  
            self._update_research_phase("researching")
            research_result = self._execute_with_context_management(
                agent,
                question,
                TaskPriority.CRITICAL
            )
            
            # Phase 3: Archive completed work and optimize
            self._cleanup_context()
            
            duration = time.time() - start_time
            
            # Prepare final result
            final_result = {
                "question": question,
                "result": research_result,
                "duration": duration,
                "context_stats": self.context_optimizer.get_context_stats(),
                "files": research_result.get("files", {}),
                "messages": research_result.get("messages", [])
            }
            
            logger.info(f"Research completed in {duration:.2f}s. Context utilization: {final_result['context_stats']['utilization']:.1%}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Research failed: {str(e)}", exc_info=True)
            # Add error to context for potential recovery
            self.context_optimizer.add_context_item(
                f"Research error: {str(e)}",
                "error", 
                TaskPriority.CRITICAL
            )
            raise
    
    def _execute_with_context_management(self, agent, content: str, priority: int) -> Dict[str, Any]:
        """Execute agent with context management and optimization"""
        
        # Check context status before execution
        context_stats = self.context_optimizer.get_context_stats()
        
        if context_stats["utilization"] > 0.9:
            logger.warning(f"Context near capacity ({context_stats['utilization']:.1%}). Optimizing...")
            self._emergency_context_optimization()
        
        # Add task to context
        task_id = self.context_optimizer.add_context_item(
            content,
            "task",
            priority
        )
        
        # Execute with rate limiting
        result = self._execute_with_retry(agent, content)
        
        # Update context with result
        if result:
            self.context_optimizer.update_context_item(
                task_id,
                f"COMPLETED: {content}\nResult: {str(result)[:200]}..."
            )
            
            # Archive if low priority
            if priority >= TaskPriority.MEDIUM:
                self.context_optimizer.archive_completed_tasks([task_id])
        
        return result
    
    def _execute_with_retry(self, agent, content: str) -> Dict[str, Any]:
        """Execute agent with rate limit handling"""
        for attempt in range(MAX_RETRIES + 1):
            try:
                messages = [{"role": "user", "content": content}]
                result = agent.invoke({"messages": messages})
                return result
                
            except RateLimitError as e:
                if attempt == MAX_RETRIES:
                    raise
                
                wait_time = BASE_DELAY * (2 ** attempt) + RATE_LIMIT_DELAY
                logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}")
                time.sleep(wait_time)
            
            except Exception as e:
                logger.error(f"Execution error on attempt {attempt + 1}: {str(e)}")
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(BASE_DELAY * (2 ** attempt))
    
    def _update_research_phase(self, phase: str):
        """Update research phase and optimize context accordingly"""
        old_phase = self.session_state["research_phase"]
        self.session_state["research_phase"] = phase
        
        # Add phase transition to context
        self.context_optimizer.add_context_item(
            f"Research phase: {old_phase} -> {phase}",
            "state",
            TaskPriority.MEDIUM
        )
        
        # Phase-specific optimizations
        if phase == "writing":
            # Archive research results, keep summaries
            self._archive_research_phase_items("researching")
        elif phase == "reviewing":
            # Archive writing process, keep final outputs
            self._archive_research_phase_items("writing")
        
        logger.info(f"Research phase updated: {old_phase} -> {phase}")
    
    def _archive_research_phase_items(self, phase: str):
        """Archive items from a specific research phase"""
        items_to_archive = []
        for item_id, item in self.context_optimizer.active_context.items():
            if phase in item.content.lower() and item.priority >= TaskPriority.MEDIUM:
                items_to_archive.append(item_id)
        
        if items_to_archive:
            self.context_optimizer.archive_completed_tasks(items_to_archive)
            logger.info(f"Archived {len(items_to_archive)} items from {phase} phase")
    
    def _cleanup_context(self):
        """Clean up context after research completion"""
        # Archive all completed tasks
        completed_tasks = []
        for item_id, item in self.context_optimizer.active_context.items():
            if item.type == "task" and "COMPLETED" in item.content:
                completed_tasks.append(item_id)
        
        if completed_tasks:
            self.context_optimizer.archive_completed_tasks(completed_tasks)
        
        # Log final context stats
        stats = self.context_optimizer.get_context_stats()
        logger.info(f"Context cleanup complete. Final utilization: {stats['utilization']:.1%}")
    
    def _emergency_context_optimization(self):
        """Emergency context optimization when near capacity"""
        logger.warning("Performing emergency context optimization")
        
        stats_before = self.context_optimizer.get_context_stats()
        
        # Archive all non-critical items
        items_to_archive = []
        for item_id, item in self.context_optimizer.active_context.items():
            if item.priority > TaskPriority.HIGH:
                items_to_archive.append(item_id)
        
        if items_to_archive:
            self.context_optimizer.archive_completed_tasks(items_to_archive)
        
        # Force context optimization
        self.context_optimizer._optimize_context()
        
        stats_after = self.context_optimizer.get_context_stats()
        
        logger.info(f"Emergency optimization: {stats_before['utilization']:.1%} -> {stats_after['utilization']:.1%}")

# Example usage
if __name__ == "__main__":
    # Initialize optimized research agent
    research_agent = OptimizedResearchAgent(max_context_size=14000)
    
    # Test question
    test_question = "What are the latest developments in quantum computing in 2024?"
    
    # Show multi-model setup
    print(f"🚀 Multi-Model Research Agent Setup:")
    print(f"   📋 Main Agent: {research_agent.main_model_name}")
    print(f"   🔬 Research Sub-Agent: {research_agent.research_model_name}")
    print(f"   ✅ Critique Sub-Agent: {research_agent.critique_model_name}")
    print()
    print(f"🔍 Starting optimized research on: {test_question}")
    print("=" * 70)
    
    try:
        result = research_agent.conduct_research(test_question)
        
        print("\n📊 Research Complete!")
        print("=" * 70)
        print(f"⏱️  Duration: {result['duration']:.2f} seconds")
        print(f"🧠 Context Utilization: {result['context_stats']['utilization']:.1%}")
        print(f"📁 Active Context Items: {result['context_stats']['active_items']}")
        print(f"🗄️  Archived Items: {result['context_stats']['archived_items']}")
        print(f"💾 Search Cache Entries: {len(research_agent.search_memory['queries'])}")
        
        # Show final response
        if result['messages']:
            print("\n📝 Final Response:")
            print("-" * 50)
            print(result['messages'][-1].content)
        
        # Show any files created
        if result.get('files'):
            print("\n📄 Files Generated:")
            print("-" * 50)
            for filename, content in result['files'].items():
                print(f"\n📋 {filename}:")
                print(content[:300] + "..." if len(content) > 300 else content)
    
    except Exception as e:
        print(f"\n❌ Research failed: {str(e)}")
        print("Check the log file for more details.")

"""
Memory Retrieval Agent - A specialized agent for intelligent memory exploration and retrieval.

This agent implements a partitioned memory system with three-layer abstraction levels:
- Methods: Quick scanning and relevance filtering
- Rules: Tactical-level analysis of promising memories  
- Trajectories: Deep analysis and example extraction

Structured Output Contract (for wrapped tools):
----------------------------------------------
All public wrapped tools defined by this agent return JSON-serializable Python dicts with a
stable schema. These include keys such as ok, partition, items, count, total, and a pretty
field for human-readable inspection. See each tool's docstring for exact schemas.

Why: Downstream agent code often treats results as lists/dicts. Returning structured data
prevents type errors when iterating fields and enables programmatic use, while preserving
"pretty" for logging and UI.

The agent maintains exploration state and provides comprehensive memory analysis.
"""

import json
import os
from typing import List, Dict, Any, Optional
from smolagents import CodeAgent, tool
from smolagents import OpenAIServerModel
import yaml


class MemoryRetrieverAgent(CodeAgent):
    """
    A specialized agent for intelligent memory retrieval and exploration.
    
    Implements partitioned memory system (Golden/Warning/Mixed) with multi-level
    abstraction exploration and state management.
    """
    
    def __init__(self, model, memory_path: str = None, system_prompt_path: Optional[str] = None):
        """
        Initialize the Memory Retriever Agent.
        
        Args:
            model: The language model to use for reasoning
            memory_path: Path to the partitioned memory database
        """
        self.memory_path = memory_path or "/home/eric_guoxy/mnt/d/Research/forward learning/test/exps/"
        self.memory_data = {}  # Will store partitioned data
        self.exploration_state = {
            'golden': {'explored_up_to': -1, 'recommended_indices': []},
            'warning': {'explored_up_to': -1, 'recommended_indices': []},
            'mixed': {'explored_up_to': -1, 'recommended_indices': []}
        }
        
        self._load_partitioned_memory()
        self.wrapped_tools = self._create_wrapped_tools()

        # Load the system prompt template (configurable: aime or retro)
        yaml_path = system_prompt_path or "prompts/memory_aime.yaml"
        try:
            with open(yaml_path, "r") as f:
                system_prompt = yaml.safe_load(f)
        except Exception as e:
            # Fallback to old default path for backward compatibility
            try:
                with open("prompts/memory.yaml", "r") as f:
                    system_prompt = yaml.safe_load(f)
            except Exception:
                raise RuntimeError(f"Failed to load memory system prompt YAML from {yaml_path} and legacy prompts/memory.yaml: {e}")
        
        # Initialize parent with wrapped tools
        super().__init__(
            tools=self.wrapped_tools,
            model=model,
            max_steps=30,
            prompt_templates=system_prompt
        )
    
    def _load_partitioned_memory(self):
        """Load partitioned memory data from self.memory_path.

        Supports either a single JSON file or a directory containing one or more
        JSON files. When a directory is provided, all JSON files are scanned and
        merged by partition.
        """
        try:
            if not os.path.exists(self.memory_path):
                print(f"⚠️  Memory path not found at {self.memory_path}, using empty partitions")
                self.memory_data = {'golden': [], 'warning': [], 'mixed': []}
                return

            aggregated = {
                'golden': [],
                'warning': [],
                'mixed': [],
            }

            loaded_files = []

            if os.path.isdir(self.memory_path):
                # Load and merge all JSON files under the directory
                for fname in sorted(os.listdir(self.memory_path)):
                    if not fname.lower().endswith('.json'):
                        continue
                    fpath = os.path.join(self.memory_path, fname)
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            content = json.load(f)
                    except Exception as e:
                        print(f"⚠️  Skipping '{fpath}': {e}")
                        continue

                    if isinstance(content, dict) and 'library' in content:
                        lib = content['library'] or {}
                        aggregated['golden'].extend(lib.get('golden', []) or [])
                        aggregated['warning'].extend(lib.get('warning', []) or [])
                        aggregated['mixed'].extend(lib.get('mixed', []) or [])
                        loaded_files.append(fname)
                    else:
                        # If no 'library' key, treat as a list (or single item) of golden memories
                        if isinstance(content, list):
                            aggregated['golden'].extend(content)
                        else:
                            aggregated['golden'].append(content)
                        loaded_files.append(fname)

                self.memory_data = aggregated
                total_entries = sum(len(v) for v in self.memory_data.values())
                print(f"✅ Loaded memory library from directory: {self.memory_path}")
                print(f"  Files loaded: {', '.join(loaded_files) if loaded_files else 'None'}")
                for part, data in self.memory_data.items():
                    print(f"  - {part}: {len(data)} entries")
                print(f"  Total: {total_entries} entries")
            else:
                # Single file path
                with open(self.memory_path, 'r', encoding='utf-8') as f:
                    memory_file = json.load(f)

                if isinstance(memory_file, dict) and 'library' in memory_file:
                    library = memory_file['library']
                    self.memory_data = {
                        'golden': library.get('golden', []) or [],
                        'warning': library.get('warning', []) or [],
                        'mixed': library.get('mixed', []) or [],
                    }
                    total_entries = sum(len(partition) for partition in self.memory_data.values())
                    print(f"✅ Loaded memory library from {self.memory_path}:")
                    for partition, data in self.memory_data.items():
                        print(f"  - {partition}: {len(data)} entries")
                    print(f"  Total: {total_entries} entries")
                else:
                    # Fallback: treat entire file as golden partition if no library structure
                    self.memory_data = {
                        'golden': memory_file if isinstance(memory_file, list) else [memory_file],
                        'warning': [],
                        'mixed': []
                    }
                    print(f"✅ Loaded {len(self.memory_data['golden'])} entries as golden partition (no library structure)")

        except Exception as e:
            print(f"❌ Error loading memory: {e}")
            self.memory_data = {'golden': [], 'warning': [], 'mixed': []}
    
    
    def _create_wrapped_tools(self):
        """Create tool wrappers for partitioned memory exploration.

        All wrapped tools return JSON-serializable Python dicts with a stable schema,
        rather than human-formatted strings. This aligns with downstream code that
        expects structured data (list/dicts) instead of plain text and prevents
        TypeError when iterating fields.
        """
        @tool
        def get_partition_methods(partition: str, index_range: List[int]) -> Dict[str, Any]:
            """
            Retrieve method-level information for quick scanning.

            Args:
                partition: Memory partition ('golden', 'warning', 'mixed')
                index_range: List of indices to retrieve (0-based)

            Returns:
                Dict[str, Any]: JSON-serializable payload with the following schema:
                    {
                        ok: bool,                     # True if request succeeded
                        partition: str,               # Queried partition name
                        total: int,                   # Total items available in partition
                        count: int,                   # Number of items returned
                        range_requested: List[int],   # Sanitized indices requested
                        items: List[{
                            index: int,
                            method: Any,               # Original method field (str|dict|list|None)
                            method_text: str,          # Stringified method content for scanning
                            metadata: {                # Normalized (never None) strings
                                problem_category: str,   # '' if missing
                                priority: str,           # '' if missing
                                memory_type: str         # 'golden'|'warning'|'mixed'
                            }
                        }],
                        pretty: str                  # Human-readable summary
                    }

            Notes:
                - All metadata string fields are normalized to '' (empty string) if missing.
                - method_text is always a string (JSON-dumped if original is a dict).
            """
            return self._get_partition_methods(partition, index_range)

        @tool  
        def get_partition_rules(partition: str, index_range: List[int]) -> Dict[str, Any]:
            """
            Retrieve detailed rules for selected memories (structured).

            Args:
                partition: Memory partition name ('golden', 'warning', or 'mixed').
                index_range: 0-based indices to fetch from the chosen partition.

            Returns:
                Dict[str, Any]: JSON-serializable payload with keys:
                  - ok (bool): True if request succeeded.
                  - partition (str): The partition queried.
                  - total (int): Total available items in the partition.
                  - count (int): Number of items actually returned.
                  - range_requested (List[int]): Sanitized indices requested.
                  - items (List[Dict]): Each item contains:
                      - index (int)
                      - rules (List[Dict|str]): Either {rule: str, examples: List[str]} or a plain string
                      - num_rules (int)
                  - pretty (str): Optional human-readable summary for logging/UI.
            """
            return self._get_partition_rules(partition, index_range)

        @tool
        def get_partition_trajectories(partition: str, index_range: List[int]) -> Dict[str, Any]:
            """
            Retrieve complete trajectory information with insights (structured).

            Args:
                partition: Memory partition name ('golden', 'warning', or 'mixed').
                index_range: 0-based indices to fetch from the chosen partition.

            Returns:
                Dict[str, Any]: JSON-serializable payload with keys:
                  - ok (bool): True if request succeeded.
                  - partition (str): The partition queried.
                  - total (int): Total available items in the partition.
                  - count (int): Number of items actually returned.
                  - range_requested (List[int]): Sanitized indices requested.
                  - items (List[Dict]): Each item contains:
                      - index (int)
                      - trajectory (str): Revised trajectory if present, otherwise trajectory.
                      - length (int): Character length of trajectory.
                      - insight_comments (int): Count of '<comments>' markers.
                      - preview (str): First ~400 chars for quick inspection.
                  - pretty (str): Optional human-readable summary for logging/UI.
            """
            return self._get_partition_trajectories(partition, index_range)

        @tool
        def get_exploration_state(partition: str = None) -> Dict[str, Any]:
            """
            Get current exploration progress (structured).

            Args:
                partition: Optional specific partition ('golden'|'warning'|'mixed'). If omitted, returns all partitions.

            Returns:
                Dict[str, Any]: Two possible forms:
                    - If partition is provided:
                        {
                            ok: bool,
                            partition: str,
                            state: { explored_up_to: int, recommended_indices: List[int] }
                        }
                    - Else (all partitions):
                        {
                            ok: true,
                            state_by_partition: {
                                <partition>: { explored_up_to: int, recommended_indices: List[int], total_available: int }
                            },
                            pretty: str
                        }
            """
            return self._get_exploration_state(partition)

        @tool
        def update_exploration_state(partition: str, up_to: int, recommended_indices: List[int]) -> Dict[str, Any]:
            """
            Update exploration state (structured).

            Args:
                partition: Partition to update ('golden'|'warning'|'mixed').
                up_to: Last explored index for this session.
                recommended_indices: Indices deemed relevant to record.

            Returns:
                Dict[str, Any]:
                    {
                        ok: bool,
                        partition: str,
                        explored_up_to: int,
                        recommended_indices: List[int],
                        total_recommended: int,
                        pretty: str
                    }
            """
            return self._update_exploration_state(partition, up_to, recommended_indices)

        @tool
        def get_memory_statistics() -> Dict[str, Any]:
            """
            Get memory library statistics (structured).

            Returns:
                Dict[str, Any]:
                    {
                        ok: true,
                        partitions: {
                            <partition>: { total_count: int, last_index: int, available_range: [int, int] }
                        },
                        total_memories: int,
                        pretty: str
                    }
            """
            return self._get_memory_statistics()

        return [
            get_partition_methods,
            get_partition_rules,
            get_partition_trajectories,
            get_exploration_state,
            update_exploration_state,
            get_memory_statistics,
        ]
    
    
    def _get_partition_methods(self, partition: str, index_range: List[int]) -> Dict[str, Any]:
        """Implementation for method-level retrieval returning structured data."""
        if partition not in self.memory_data:
            return {
                "ok": False,
                "error": f"Invalid partition: {partition}",
                "available_partitions": list(self.memory_data.keys()),
            }

        data = self.memory_data[partition]

        # sanitize and de-duplicate requested indices
        requested = sorted({int(i) for i in index_range if isinstance(i, (int, str))})
        items: List[Dict[str, Any]] = []
        for idx in requested:
            if 0 <= idx < len(data):
                memory = data[idx]
                method_val = memory.get('method', 'No method available')
                if isinstance(method_val, dict):
                    try:
                        method_text = json.dumps(method_val, ensure_ascii=False)
                    except Exception:
                        method_text = str(method_val)
                else:
                    method_text = str(method_val)

                # Merge metadata from possible locations
                raw_meta = memory.get('memory_metadata') or {}
                # Top-level fallbacks commonly present in our datasets
                if not raw_meta:
                    raw_meta = {
                        'problem_category': memory.get('problem_category'),
                        'priority': memory.get('priority'),
                        'memory_type': memory.get('memory_type', partition),
                    }

                # Normalize to safe strings to avoid None-related errors downstream
                problem_category = str(raw_meta.get('problem_category') or '')
                priority = str(raw_meta.get('priority') or '')
                memory_type = str(raw_meta.get('memory_type') or partition)
                items.append({
                    "index": idx,
                    "method": method_val,
                    "method_text": method_text,
                    "metadata": {
                        "problem_category": problem_category,
                        "priority": priority,
                        "memory_type": memory_type,
                    },
                })

        pretty_lines = [
            f"📋 METHOD-LEVEL SCAN: {partition.upper()} PARTITION",
            "=" * 50,
            "",
        ]
        for it in items:
            pretty_lines.append(f"Index {it['index']}:")
            pretty_lines.append(f"Method: {it['method_text']}")
            meta = it['metadata']
            pretty_lines.append(f"Category: {meta.get('problem_category') or 'N/A'}")
            pretty_lines.append(f"Priority: {meta.get('priority') or 'N/A'}")
            pretty_lines.append(f"Type: {meta.get('memory_type') or partition}")
            pretty_lines.append("")
        pretty_lines.append(f"📊 Retrieved {len(items)} method summaries from {partition} partition")



        return {
            "ok": True,
            "partition": partition,
            "total": len(data),
            "count": len(items),
            "range_requested": requested,
            "items": items,
            "pretty": "\n".join(pretty_lines),
        }
    
    def _get_partition_rules(self, partition: str, index_range: List[int]) -> Dict[str, Any]:
        """Implementation for rules-level retrieval returning structured data."""
        if partition not in self.memory_data:
            return {
                "ok": False,
                "error": f"Invalid partition: {partition}",
                "available_partitions": list(self.memory_data.keys()),
            }

        data = self.memory_data[partition]

        requested = sorted({int(i) for i in index_range if isinstance(i, (int, str))})
        items: List[Dict[str, Any]] = []
        pretty_lines = [
            f"📚 RULES-LEVEL ANALYSIS: {partition.upper()} PARTITION",
            "=" * 50,
            "",
        ]

        for idx in requested:
            if 0 <= idx < len(data):
                memory = data[idx]
                rules = memory.get('rules', [])
                normalized_rules: List[Any] = []
                for rule in rules:
                    if isinstance(rule, dict):
                        normalized_rules.append({
                            "rule": rule.get('rule', 'No rule text'),
                            "examples": rule.get('examples', []),
                        })
                    else:
                        normalized_rules.append(str(rule))

                items.append({
                    "index": idx,
                    "rules": normalized_rules,
                    "num_rules": len(normalized_rules),
                })

                pretty_lines.append(f"--- Memory {idx} Rules ---")
                pretty_lines.append(f"Number of rules: {len(normalized_rules)}")
                pretty_lines.append("")
                # include up to 2 example lines per rule for readability
                for ridx, rule in enumerate(normalized_rules[:10]):
                    if isinstance(rule, dict):
                        rtext = rule.get('rule', '')
                        examples = rule.get('examples', [])
                        pretty_lines.append(f"  Rule {ridx+1}: {rtext}")
                        if examples:
                            pretty_lines.append(f"    Examples: {examples[:2]}")
                    else:
                        pretty_lines.append(f"  Rule {ridx+1}: {rule}")
                pretty_lines.append("")

        pretty_lines.append(f"📊 Retrieved rules for {len(items)} memories from {partition} partition")
        return {
            "ok": True,
            "partition": partition,
            "total": len(data),
            "count": len(items),
            "range_requested": requested,
            "items": items,
            "pretty": "\n".join(pretty_lines),
        }
    
    def _get_partition_trajectories(self, partition: str, index_range: List[int]) -> Dict[str, Any]:
        """Implementation for trajectory-level retrieval returning structured data."""
        if partition not in self.memory_data:
            return {
                "ok": False,
                "error": f"Invalid partition: {partition}",
                "available_partitions": list(self.memory_data.keys()),
            }

        data = self.memory_data[partition]
        requested = sorted({int(i) for i in index_range if isinstance(i, (int, str))})
        items: List[Dict[str, Any]] = []
        pretty_lines = [
            f"🗂️  TRAJECTORY-LEVEL ANALYSIS: {partition.upper()} PARTITION",
            "=" * 50,
            "",
        ]

        for idx in requested:
            if 0 <= idx < len(data):
                memory = data[idx]
                trajectory = memory.get('revised_trajectory', memory.get('trajectory', 'No trajectory available'))
                t_str = str(trajectory)
                comments_count = t_str.count('<comments>') if isinstance(t_str, str) else 0
                items.append({
                    "index": idx,
                    "trajectory": t_str,
                    "length": len(t_str),
                    "insight_comments": comments_count,
                    "preview": t_str[:400],
                })

                pretty_lines.append(f"--- Memory {idx} Implementation Trajectory ---")
                pretty_lines.append(f"Trajectory length: {len(t_str)} characters")
                pretty_lines.append(f"Insight comments available: {comments_count}")
                pretty_lines.append("")
                pretty_lines.append("Trajectory Content:")
                pretty_lines.append(t_str)
                pretty_lines.append("")
                pretty_lines.append("-" * 40)
                pretty_lines.append("")

        pretty_lines.append(f"📊 Retrieved trajectories for {len(items)} memories from {partition} partition")
        return {
            "ok": True,
            "partition": partition,
            "total": len(data),
            "count": len(items),
            "range_requested": requested,
            "items": items,
            "pretty": "\n".join(pretty_lines),
        }
    
    def _get_exploration_state(self, partition: str = None) -> Dict[str, Any]:
        """Get current exploration state (structured)."""
        if partition:
            if partition in self.exploration_state:
                state = self.exploration_state[partition]
                return {
                    "ok": True,
                    "partition": partition,
                    "state": {
                        "explored_up_to": state['explored_up_to'],
                        "recommended_indices": list(state['recommended_indices']),
                    },
                }
            else:
                return {"ok": False, "error": f"Invalid partition: {partition}"}

        # all partitions
        pretty_lines = ["🗺️  EXPLORATION STATE", "==================", ""]
        state_by_part: Dict[str, Any] = {}
        for part, state in self.exploration_state.items():
            max_index = len(self.memory_data.get(part, [])) - 1
            total_available = max_index + 1 if max_index >= 0 else 0
            up_to = state['explored_up_to']
            state_by_part[part] = {
                "explored_up_to": up_to,
                "recommended_indices": list(state['recommended_indices']),
                "total_available": total_available,
            }

            pretty_lines.append(f"{part.upper()} Partition:")
            pretty_lines.append(f"  Explored up to: {up_to}")
            pretty_lines.append(f"  Total available: {total_available} memories")
            pretty_lines.append(f"  Recommended indices: {state['recommended_indices']}")
            if up_to == -1:
                pretty_lines.append("  Status: Not explored - start from index 0")
            elif up_to >= max_index:
                pretty_lines.append("  Status: Exploration complete")
            else:
                pretty_lines.append(f"  Status: Continue from index {up_to + 1}")
            pretty_lines.append("")

        return {"ok": True, "state_by_partition": state_by_part, "pretty": "\n".join(pretty_lines)}
    
    def _update_exploration_state(self, partition: str, up_to: int, recommended_indices: List[int]) -> Dict[str, Any]:
        """Update exploration state (structured)."""
        if partition not in self.exploration_state:
            return {"ok": False, "error": f"Invalid partition: {partition}"}

        # Ensure recommended_indices contains a list of integers
        coerced: List[int] = []
        for val in recommended_indices:
            try:
                coerced.append(int(val))
            except Exception:
                continue

        self.exploration_state[partition]['explored_up_to'] = int(up_to)
        # Merge with existing recommendations, avoiding duplicates
        existing = self.exploration_state[partition]['recommended_indices']
        new_recommendations = sorted(set(existing + coerced))
        self.exploration_state[partition]['recommended_indices'] = new_recommendations

        pretty = (
            f"✅ Updated {partition} partition: explored_up_to={up_to}, "
            f"total_recommended={len(new_recommendations)}"
        )
        return {
            "ok": True,
            "partition": partition,
            "explored_up_to": int(up_to),
            "recommended_indices": new_recommendations,
            "total_recommended": len(new_recommendations),
            "pretty": pretty,
        }
    
    def _get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory statistics (structured)."""
        pretty_lines = ["📊 MEMORY LIBRARY STATISTICS", "============================", ""]

        total_memories = 0
        partitions: Dict[str, Any] = {}
        for partition, data in self.memory_data.items():
            count = len(data)
            total_memories += count
            last_index = count - 1 if count > 0 else -1
            partitions[partition] = {
                "total_count": count,
                "last_index": last_index,
                "available_range": [0, last_index],
            }

            pretty_lines.append(f"{partition.upper()} Partition:")
            pretty_lines.append(f"  Total count: {count}")
            pretty_lines.append(f"  Last index: {last_index}")
            pretty_lines.append(f"  Available range: 0-{last_index}")
            pretty_lines.append("")

        pretty_lines.append(f"TOTAL MEMORIES: {total_memories}")
        return {"ok": True, "partitions": partitions, "total_memories": total_memories, "pretty": "\n".join(pretty_lines)}

    def retrieve_memory(self, query: str) -> str:
        """
        Main method to retrieve memory based on actor's query.

        Args:
            query: The actor's memory retrieval request

        Returns:
            str: A synthesized, human-readable report produced by the internal agent run.

        Notes:
            - This high-level facade returns a string report for readability and logging.
            - For programmatic consumption (e.g., iterating items), prefer calling the
              wrapped tools (get_partition_methods/rules/trajectories, etc.), which
              return JSON-serializable dicts with explicit schemas documented in their
              docstrings. This avoids treating plain strings as dicts and prevents
              errors like "TypeError: string indices must be integers".
        """
        try:
            print(f"🔍 Processing query: {query}")
            
            # Create task prompt based on design document system prompt
            task_prompt = f"""
You are a specialized Memory Retriever Agent with advanced semantic understanding capabilities. Your role is to intelligently explore a partitioned memory library and provide comprehensive, relevant recommendations to support problem-solving activities.

QUERY: {query}

You work with a three-partition memory system:
- **Golden Partition**: Contains successful problem-solving experiences
- **Warning Partition**: Contains failure cases and error patterns  
- **Mixed Partition**: Contains learning trajectories and improvement paths

Use the **Thought → Code → Observation** cycle approach:

1. **Thought**: Analyze requirements and plan next action
2. **Code**: Execute single tool call with clear documentation  
3. **Observation**: Reflect on results and plan continuation

Your exploration strategy should be:
1. Check current exploration state
2. Analyze query for abstraction levels (Strategy/Tactical/Detail/Warning/Learning)
3. Perform method-level scanning across relevant partitions
4. Use rules-level analysis for promising memories
5. Apply trajectory-level examination for concrete examples
6. Maintain exploration state and provide final analysis with final_answer()

Start by checking your exploration state and analyzing the query requirements.
"""
            
            # Run the agent to perform the retrieval
            result = self.run(task_prompt)
            
            return result
            
        except Exception as e:
            return f"❌ Error during memory retrieval: {str(e)}"


# Factory function to create the memory retriever agent
def create_memory_retriever(model, memory_path: str = None, system_prompt_path: Optional[str] = None) -> MemoryRetrieverAgent:
    """
    Factory function to create a memory retriever agent.
    
    Args:
        model: The language model to use
        memory_path: Path to the memory database directory
        
    Returns:
        MemoryRetrieverAgent: Configured memory retrieval agent
    """
    return MemoryRetrieverAgent(model=model, memory_path=memory_path, system_prompt_path=system_prompt_path)


# Tool wrapper for use by other agents
@tool
def agentic_memory_retrieval(query: str, retriever_agent: MemoryRetrieverAgent = None) -> str:
    """
    Intelligent memory retrieval tool that uses a persistent MemoryRetrieverAgent.
    
    Args:
        query: The memory retrieval query
        retriever_agent: The MemoryRetrieverAgent instance to use for retrieval
        
    Returns:
        str: Retrieved and consolidated memory information
    """
    if retriever_agent is None:
        return "❌ Error: No MemoryRetrieverAgent provided. Please pass a retriever_agent instance."
    
    try:
        return retriever_agent.retrieve_memory(query)
    except Exception as e:
        return f"❌ Error during memory retrieval: {str(e)}"


def create_memory_retrieval_tool(retriever_agent: MemoryRetrieverAgent):
    """
    Create a memory retrieval tool bound to a specific MemoryRetrieverAgent instance.
    
    Args:
        retriever_agent: The MemoryRetrieverAgent instance to bind to the tool
        
    Returns:
        callable: A tool function that uses the bound retriever agent
    """
    @tool
    def memory_retrieval(query: str) -> str:
        """
        Intelligent memory retrieval using the bound MemoryRetrieverAgent.
        
        Args:
            query: The memory retrieval query describing what kind of information is needed
            
        Returns:
            str: Retrieved and consolidated memory information
        """
        try:
            return retriever_agent.retrieve_memory(query)
        except Exception as e:
            return f"❌ Error during memory retrieval: {str(e)}"
    
    return memory_retrieval


if __name__ == "__main__":
    # Example usage
    print("🧠 Memory Retriever Agent - Partitioned System")
    print("=" * 60)
    
    print("To use this agent, initialize it with a language model:")
    print("retriever = create_memory_retriever(your_model)")
    print("result = retriever.retrieve_memory('your query here')")

import os
import json
import re
import asyncio
from rate_limiter import RateLimitedLLMClient

class Updater:
    """Updater组件: 根据反馈所提取的事实和方法论及ExpLib现状 决定如何更新分区记忆库"""
    def __init__(self, model, exp_lib, temperature=0):
        self.model = model
        self.temperature = temperature
        self.exp_lib = exp_lib
        self.llm_client = RateLimitedLLMClient()
    
    async def llm(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        return await self.llm_client.chat_completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )

    async def analyse_and_update(self, data_dict):
        """
        使用一次LLM调用分析新的data dict并更新分区记忆库
        现在支持 golden/warning/mixed 分区系统
        """
        # 检查是否有记忆元数据建议
        memory_metadata = data_dict.get("memory_metadata", {})
        suggested_type = memory_metadata.get("suggested_type", "mixed")
        problem_category = memory_metadata.get("problem_category", "")
        priority = memory_metadata.get("priority", 3)
        
        # 提取核心数据（移除元数据）
        core_data = {k: v for k, v in data_dict.items() if k != "memory_metadata"}
        
        # 获取当前分区的相关记忆
        current_partition_memories = self.exp_lib.get_memories_by_type(suggested_type)
        
        # 如果当前分区为空，直接添加
        if not current_partition_memories:
            self.exp_lib.add_memory(
                data=core_data,
                memory_type=suggested_type,
                problem_category=problem_category,
                priority=priority
            )
            print(f"直接添加到空分区 '{suggested_type}': {core_data.get('method', '')[:50]}...")
            return
            
        # 构建分区特定的分析prompt
        prompt = self._build_partition_analysis_prompt(core_data, current_partition_memories, suggested_type, problem_category)
        
        # 获取LLM的决策
        response = await self.llm(prompt)
        decision = self._extract_partition_decision(response)
        
        # 执行决策
        await self._execute_partition_decision(core_data, decision, suggested_type, problem_category, priority)
    
    def _build_partition_analysis_prompt(self, new_data, partition_memories, memory_type, problem_category):
        """构建分区特定的分析prompt"""
        # 格式化当前分区的记忆
        partition_entries = []
        for i, memory in enumerate(partition_memories):
            method = str(memory.get("method", ""))
            rules = memory.get("rules", [])
            category = str(memory.get("problem_category", ""))
            priority = memory.get("priority", 1)
            
            # 确保每个rule都是字符串
            rules_text = "\n    ".join([f"Rule {j}: {str(rule.get('rule', '') if isinstance(rule, dict) else str(rule))}" for j, rule in enumerate(rules)])
            
            entry_text = f"Memory {i} [Category: {category}, Priority: {priority}]:\n  Method: {method}"
            if rules_text:
                entry_text += f"\n  Rules:\n    {rules_text}"
            partition_entries.append(entry_text)
        
        partition_str = "\n\n".join(partition_entries) if partition_entries else "No existing memories"
        
        # 格式化新数据
        new_method = str(new_data.get("method", ""))
        new_rules = new_data.get("rules", [])
        new_trajectory = new_data.get("revised_trajectory", "")
        
        # 确保每个新规则都是字符串
        new_rules_text = "\n    ".join([f"New Rule {i}: {str(rule.get('rule', '') if isinstance(rule, dict) else str(rule))}" for i, rule in enumerate(new_rules)])
        
        # 根据记忆类型调整prompt策略
        if memory_type == "golden":
            strategy_note = "Focus on preserving best practices and avoiding duplication of successful methods."
        elif memory_type == "warning":
            strategy_note = "Focus on capturing unique failure patterns and error types to avoid."
        else:  # mixed
            strategy_note = "Focus on learning narratives and failure-to-success transitions."
        
        prompt = f"""
You are managing a partitioned experience library for a problem-solving AI. You need to decide how to handle new content in the '{memory_type}' partition.

STRATEGY FOR {memory_type.upper()} PARTITION:
{strategy_note}

CURRENT {memory_type.upper()} PARTITION:
{partition_str}

NEW CONTENT (Category: {problem_category}):
Method: {new_method}
Rules:
    {new_rules_text}
Trajectory Length: {len(new_trajectory)} characters

ANALYSIS TASK:
1. Determine if the new content significantly overlaps with existing memories in this partition
2. If overlap exists, decide whether to merge or keep separate based on:
   - Different problem subcategories
   - Significantly different approaches
   - Unique insights or rules
3. For merge decisions, identify which new rules are genuinely novel

DECISION RULES:
- "add": Create new memory if the approach/context is sufficiently different
- "merge": Combine with existing memory if the core method is very similar
- Consider that {memory_type} memories should maintain high quality and avoid redundancy

Respond in JSON format:
```json
{{
    "decision": "add|merge",
    "target_memory_index": null_or_index_number,
    "new_rules_operations": null_or_array_of_operations,
    "reasoning": "Brief explanation focusing on what makes this content unique or similar"
}}
```

EXPLANATION:
- If "add": set target_memory_index to null
- If "merge": set target_memory_index to the index of the similar memory
- new_rules_operations: for each new rule, use:
  - Integer (0-based index): if similar to existing rule at that index
  - "add": if the rule is genuinely new and valuable
  - "skip": if the rule is redundant or low-quality
"""
        
        return prompt
    
    def _extract_partition_decision(self, response):
        """从LLM响应中提取分区决策"""
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
            try:
                decision = json.loads(json_str)
                return decision
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                print(f"原始JSON字符串: {json_str}")
                # 默认添加
                return {"decision": "add", "target_memory_index": None, "new_rules_operations": None, "reasoning": "JSON parsing failed, defaulting to add"}
        
        # 如果没有找到JSON，默认添加
        return {"decision": "add", "target_memory_index": None, "new_rules_operations": None, "reasoning": "No JSON found, defaulting to add"}
    
    async def _execute_partition_decision(self, core_data, decision, memory_type, problem_category, priority):
        """执行分区决策"""
        decision_type = decision.get("decision", "add")
        target_index = decision.get("target_memory_index")
        rules_ops = decision.get("new_rules_operations")
        reasoning = decision.get("reasoning", "")
        
        print(f"分区决策 ({memory_type}): {decision_type} - {reasoning}")
        
        if decision_type == "add":
            # 直接添加到指定分区
            self.exp_lib.add_memory(
                data=core_data,
                memory_type=memory_type,
                problem_category=problem_category,
                priority=priority
            )
            print(f"已添加新记忆到 {memory_type} 分区")
            
        elif decision_type == "merge" and target_index is not None:
            # 与现有记忆合并
            await self._merge_partition_memory(core_data, target_index, memory_type, rules_ops)
            print(f"已与 {memory_type} 分区中的记忆 {target_index} 合并")
            
        else:
            # 默认情况，添加
            self.exp_lib.add_memory(
                data=core_data,
                memory_type=memory_type,
                problem_category=problem_category,
                priority=priority
            )
            print(f"默认添加到 {memory_type} 分区")
    
    async def _merge_partition_memory(self, new_data, target_index, memory_type, rules_ops):
        """与分区中的特定记忆合并"""
        async with self.exp_lib.lock:
            partition_memories = self.exp_lib.library[memory_type]
            
            if 0 <= target_index < len(partition_memories):
                target_memory = partition_memories[target_index]
                
                # 处理轨迹合并 - 新系统已经支持多轨迹
                new_trajectory = new_data.get("revised_trajectory", "")
                if new_trajectory:
                    # 标记记忆被使用
                    self.exp_lib.mark_memory_used(memory_type, target_index)
                    print(f"已将新轨迹添加到现有记忆的使用记录中")
                
                # 根据操作列表处理规则
                new_rules = new_data.get("rules", [])
                if new_rules and rules_ops:
                    existing_rules = target_memory.get("rules", [])
                    rules_added = 0
                    
                    for i, (new_rule, operation) in enumerate(zip(new_rules, rules_ops)):
                        if operation == "add":
                            # 添加新规则
                            existing_rules.append(new_rule)
                            rules_added += 1
                            print(f"添加了新规则 {i}: {new_rule.get('rule', '')[:50]}...")
                        elif operation == "skip":
                            print(f"跳过规则 {i} (判断为冗余)")
                        else:
                            # operation应该是现有规则的索引，表示相似，不添加
                            print(f"新规则 {i} 与现有规则 {operation} 相似，跳过")
                    
                    target_memory["rules"] = existing_rules
                    if rules_added > 0:
                        print(f"总共添加了 {rules_added} 条新规则到现有记忆")
                
                # 更新优先级（取最高值）
                current_priority = target_memory.get("priority", 1)
                new_priority = new_data.get("priority", 1)
                if new_priority > current_priority:
                    target_memory["priority"] = new_priority
                    print(f"更新记忆优先级: {current_priority} -> {new_priority}")
            else:
                print(f"警告: 目标索引 {target_index} 超出范围，执行默认添加")
                self.exp_lib.add_memory(
                    data=new_data,
                    memory_type=memory_type,
                    problem_category="",
                    priority=3
                ) 
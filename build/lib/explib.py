import json
import os
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

class ExpLib:
    """
    经验库管理系统 - 支持分区存储和智能检索
    
    分区策略：
    - golden: 成功经验，提供最佳实践和方法论
    - warning: 失败警示，提供错误模式和避坑指南  
    - mixed: 混合记忆，包含从失败到成功的完整对比
    """
    
    def __init__(self, path, load_path=None, save_path=None):
        self.path = path
        self.load_path = load_path
        self.save_path = save_path
        self.lock = asyncio.Lock()
        
        # 新的分区存储结构
        self.library = {
            "golden": [],   # 成功经验记忆
            "warning": [],  # 失败警示记忆
            "mixed": []     # 混合对比记忆
        }
        
        # 配置参数
        self.retrieval_weights = {
            "golden": 1.0,
            "warning": 0.8, 
            "mixed": 0.9
        }
        
        self.load()
    
    def add_memory(self, data: Dict, memory_type: str, problem_category: str = "", priority: int = 1) -> None:
        """
        添加记忆到指定分区
        
        Args:
            data: critic提取的记忆数据 (包含method, rules, revised_trajectory)
            memory_type: 记忆类型 ("golden", "warning", "mixed")
            problem_category: 问题类别 (如 "algebra", "geometry" 等)
            priority: 优先级 (1-5, 5为最高)
        """
        if memory_type not in self.library:
            raise ValueError(f"Invalid memory_type: {memory_type}. Must be one of {list(self.library.keys())}")
        
        # 构建增强的记忆条目
        enhanced_memory = {
            **data,  # 包含 method, rules, revised_trajectory
            "memory_type": memory_type,
            "problem_category": problem_category,
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "usage_count": 0,
            "last_used": None
        }
        
        self.library[memory_type].append(enhanced_memory)
    
    def get_memories_by_type(self, memory_type: str, limit: Optional[int] = None) -> List[Dict]:
        """按类型获取记忆"""
        if memory_type not in self.library:
            return []
        
        memories = sorted(self.library[memory_type], key=lambda x: x.get("priority", 1), reverse=True)
        return memories[:limit] if limit else memories
    
    def get_memories_by_category(self, problem_category: str, memory_types: Optional[List[str]] = None) -> List[Dict]:
        """按问题类别获取记忆"""
        if memory_types is None:
            memory_types = list(self.library.keys())
        
        results = []
        for mem_type in memory_types:
            if mem_type in self.library:
                for memory in self.library[mem_type]:
                    if memory.get("problem_category", "").lower() == problem_category.lower():
                        results.append(memory)
        
        # 按优先级和类型权重排序
        results.sort(key=lambda x: (x.get("priority", 1) * self.retrieval_weights.get(x.get("memory_type", ""), 1.0)), reverse=True)
        return results
    
    def search_memories(self, query: str = "", memory_types: Optional[List[str]] = None, 
                       problem_category: str = "", limit: int = 10) -> List[Dict]:
        """
        智能检索记忆
        
        Args:
            query: 搜索关键词
            memory_types: 指定检索的记忆类型
            problem_category: 问题类别过滤
            limit: 返回结果数量限制
        """
        if memory_types is None:
            memory_types = list(self.library.keys())
        
        results = []
        query_lower = query.lower()
        
        for mem_type in memory_types:
            if mem_type not in self.library:
                continue
                
            for memory in self.library[mem_type]:
                # 类别过滤
                if problem_category and memory.get("problem_category", "").lower() != problem_category.lower():
                    continue
                
                # 关键词匹配 (简单的包含匹配，可以后续增强为语义匹配)
                relevance_score = 0
                text_fields = [
                    memory.get("method", ""),
                    memory.get("revised_trajectory", ""),
                    " ".join([rule.get("rule", "") for rule in memory.get("rules", [])])
                ]
                
                for field in text_fields:
                    if query_lower in field.lower():
                        relevance_score += 1
                
                if not query or relevance_score > 0:
                    # 计算综合评分：相关性 * 优先级 * 类型权重
                    score = (relevance_score if query else 1) * memory.get("priority", 1) * self.retrieval_weights.get(mem_type, 1.0)
                    results.append((score, memory))
        
        # 按评分排序并返回
        results.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in results[:limit]]
    
    # ============ 兼容性方法 (保持向后兼容) ============
    
    def get_revised_trajectories(self) -> str:
        """获取所有修订轨迹 (兼容旧接口)"""
        result = []
        for mem_type, memories in self.library.items():
            for i, data in enumerate(memories):
                trajectory = data.get("revised_trajectory", "")
                if trajectory:
                    result.append(f"[{mem_type}] {i}: {trajectory}")
        
        return "\n".join(result)
    
    def get_rules(self) -> str:
        """获取所有规则 (兼容旧接口)"""
        result = []
        for mem_type, memories in self.library.items():
            for i, data in enumerate(memories):
                rules = data.get("rules", [])
                for rule_item in rules:
                    rule_text = rule_item.get("rule", "")
                    examples = rule_item.get("examples", [])
                    if rule_text:
                        result.append(f"[{mem_type}] {i}: {rule_text} | Examples: {', '.join(examples)}")
        
        return "\n".join(result)

    def get_methods(self) -> str:
        """获取所有方法论 (兼容旧接口)"""
        result = []
        for mem_type, memories in self.library.items():
            for i, data in enumerate(memories):
                method = data.get("method", "")
                if method:
                    result.append(f"[{mem_type}] {i}: {method}")
        
        return "\n".join(result)

    def get_exps(self):
        """获取所有经验 (兼容旧接口)"""
        return self.get_revised_trajectories() + '\n' + self.get_rules() + '\n' + self.get_methods()

    def get_library(self) -> Dict[str, List[Dict]]:
        """获取完整的library (返回新的分区结构)"""
        return self.library

    # ============ 统计和管理功能 ============

    def get_statistics(self) -> Dict[str, Any]:
        """获取经验库统计信息"""
        stats = {
            "total_memories": 0,
            "by_type": {},
            "by_category": {},
            "by_priority": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "total_rules": 0,
            "total_methods": 0,
            "total_trajectories": 0
        }
        
        for mem_type, memories in self.library.items():
            stats["by_type"][mem_type] = len(memories)
            stats["total_memories"] += len(memories)
            
            for memory in memories:
                # 统计优先级分布
                priority = memory.get("priority", 1)
                if priority in stats["by_priority"]:
                    stats["by_priority"][priority] += 1
                
                # 统计类别分布
                category = memory.get("problem_category", "unknown")
                stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
                
                # 统计具体内容
                if memory.get("rules"):
                    stats["total_rules"] += len(memory["rules"])
                if memory.get("method"):
                    stats["total_methods"] += 1
                if memory.get("revised_trajectory"):
                    stats["total_trajectories"] += 1
        
        return stats
    
    def mark_memory_used(self, memory_type: str, memory_index: int) -> None:
        """标记记忆被使用 (用于统计和质量评估)"""
        if memory_type in self.library and 0 <= memory_index < len(self.library[memory_type]):
            memory = self.library[memory_type][memory_index]
            memory["usage_count"] = memory.get("usage_count", 0) + 1
            memory["last_used"] = datetime.now().isoformat()
    
    def cleanup_low_quality_memories(self, min_usage: int = 0, days_unused: int = 30) -> int:
        """清理低质量记忆"""
        from datetime import datetime, timedelta
        
        removed_count = 0
        cutoff_date = datetime.now() - timedelta(days=days_unused)
        
        for mem_type in self.library:
            original_count = len(self.library[mem_type])
            self.library[mem_type] = [
                memory for memory in self.library[mem_type]
                if (memory.get("usage_count", 0) > min_usage or 
                    (memory.get("last_used") and 
                     datetime.fromisoformat(memory["last_used"]) > cutoff_date) or
                    memory.get("priority", 1) >= 4)  # 保留高优先级记忆
            ]
            removed_count += original_count - len(self.library[mem_type])
        
        return removed_count
    
    def migrate_legacy_format(self, legacy_data: List[Dict]) -> None:
        """
        迁移旧格式数据到新的分区结构
        
        Args:
            legacy_data: 旧格式的记忆列表
        """
        print("正在迁移旧格式数据到新的分区结构...")
        
        for i, data in enumerate(legacy_data):
            # 简单的启发式规则来判断记忆类型
            # 这可以根据实际情况进行调整
            memory_type = "mixed"  # 默认为混合类型
            
            # 基于内容判断类型 (简单的关键词匹配)
            content = " ".join([
                data.get("method", ""),
                data.get("revised_trajectory", ""),
                " ".join([rule.get("rule", "") for rule in data.get("rules", [])])
            ]).lower()
            
            if any(word in content for word in ["success", "correct", "excellent", "best practice"]):
                memory_type = "golden"
            elif any(word in content for word in ["error", "wrong", "avoid", "mistake", "failed"]):
                memory_type = "warning"
            
            # 添加到新结构
            self.add_memory(
                data=data,
                memory_type=memory_type,
                problem_category="migrated",  # 标记为迁移数据
                priority=2  # 给迁移数据中等优先级
            )
        
        print(f"迁移完成: {len(legacy_data)} 条记忆已迁移到新格式")

    # ============ 保存和加载功能 ============
    
    def save(self, path=None):
        """保存经验库到JSON文件"""
        if path is None:
            path = self.save_path
        
        # 添加版本信息和元数据
        save_data = {
            "version": "2.0",  # 新版本标识
            "created_at": datetime.now().isoformat(),
            "library": self.library,
            "config": {
                "retrieval_weights": self.retrieval_weights
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    def load(self, path=None):
        """从JSON文件加载经验库"""
        if path is None:
            path = self.load_path
        if path is None:
            path = self.save_path
        if path is None:
            return
            
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查是否为新格式
                if isinstance(data, dict) and "version" in data:
                    # 新格式
                    self.library = data.get("library", {"golden": [], "warning": [], "mixed": []})
                    if "config" in data:
                        self.retrieval_weights.update(data["config"].get("retrieval_weights", {}))
                    print(f"加载新格式经验库，版本: {data.get('version', 'unknown')}")
                
                elif isinstance(data, list):
                    # 旧格式，需要迁移
                    print("检测到旧格式数据，正在进行迁移...")
                    self.migrate_legacy_format(data)
                
                else:
                    # 未知格式
                    print("警告: 未知的数据格式，使用空的经验库")
                    self.library = {"golden": [], "warning": [], "mixed": []}
                    
            except (json.JSONDecodeError, IOError) as e:
                print(f"警告: 加载经验库时出错: {str(e)}")
                print("将使用空的经验库重新开始")
                self.library = {"golden": [], "warning": [], "mixed": []}

    def reset(self):
        """重置经验库"""
        self.library = {"golden": [], "warning": [], "mixed": []}
        return

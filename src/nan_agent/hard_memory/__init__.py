"""硬记忆（Hard Memory）模块。

NAN-Agent 的记忆系统核心，负责将代理的执行轨迹（episode）存储为可检索的
记忆单元，并支持经验提取和知识图谱构建。

双存储架构：
- Episodic Store：Episode + Fact → 事实性检索
- Skill Store：Exp → 行为指导
"""

from nan_agent.hard_memory.interface import HardMemory
from nan_agent.hard_memory.memory import Memory

__all__ = [
    "Memory",
    "HardMemory",
]

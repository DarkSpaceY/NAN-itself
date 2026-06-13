"""硬记忆（Hard Memory）核心数据模型。

本模块定义了 Memory 数据类，它是 NAN-Agent 记忆系统的原子存储单元。
每条 Memory 记录了一段代理经历（episode）、事实（fact）或前瞻性查询（foresight），
并通过重要性评分、衰减分数等字段支持记忆的生命周期管理。
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid

from nan_agent.hard_memory.common import new_id


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Memory:
    """硬记忆的原子存储单元。

    表示一条完整的记忆记录，可以是代理执行轨迹（episode）、提取的事实（fact）
    或前瞻性查询（foresight）。每条记忆携带时间戳、重要性评分、衰减分数等
    元信息，用于后续的检索、衰减和聚类。

    Attributes:
        id: 唯一标识符，12 位十六进制字符串。
        type: 记忆类型，可取 "episode"（轨迹）、"fact"（事实）、"foresight"（前瞻查询）。
        content: 记忆的文本内容（摘要或查询文本）。
        episode_text: 原始 episode 的完整文本，用于追溯来源。
        source: 记忆来源，如 "task_agent"（任务代理）、"self_value"（自我价值）等。
        timestamp: 创建时间的 UTC ISO 格式字符串。
        importance: 重要性评分，范围 0.0~1.0，默认 0.5。
        access_count: 该记忆被检索访问的累计次数。
        decay_score: 衰减分数，范围 0.0~1.0，由 MemoryDecay 计算更新。
        parent_id: 父记忆 ID，用于 fact/foresight 关联其来源 episode。
        cluster_id: 聚类 ID，-1 表示未分配。
        metadata: 额外的元数据字典，可存储情感状态等扩展信息。
    """
    id: str = field(default_factory=new_id)
    type: str = ""                      # "episode" | "fact" | "foresight"
    content: str = ""                   # 文本内容
    episode_text: str = ""              # 原始 episode 全文
    source: str = ""                    # "got" | "task_agent" | "self_value"
    timestamp: str = field(default_factory=_now_iso)
    importance: float = 0.5
    access_count: int = 0
    decay_score: float = 1.0
    parent_id: str = ""                 # fact/foresight 的父 episode memory.id
    cluster_id: int = -1                # -1 表示未分配
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """将 Memory 序列化为字典，用于持久化存储。"""
        d = {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "episode_text": self.episode_text,
            "source": self.source,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "access_count": self.access_count,
            "decay_score": self.decay_score,
            "parent_id": self.parent_id,
            "cluster_id": self.cluster_id,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """从字典反序列化恢复 Memory 实例。"""
        return cls(
            id=data.get("id", new_id()),
            type=data.get("type", ""),
            content=data.get("content", ""),
            episode_text=data.get("episode_text", ""),
            source=data.get("source", ""),
            timestamp=data.get("timestamp", _now_iso()),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            decay_score=data.get("decay_score", 1.0),
            parent_id=data.get("parent_id", ""),
            cluster_id=data.get("cluster_id", -1),
            metadata=data.get("metadata", {}),
        )

    def __getitem__(self, key: str):
        """支持字典式属性访问，如 memory["type"]。"""
        return getattr(self, key)

    def get(self, key: str, default=None):
        """支持安全的字典式属性访问，可指定默认值。"""
        return getattr(self, key, default)

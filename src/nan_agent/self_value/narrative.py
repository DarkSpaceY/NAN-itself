"""
自我叙事（Self-Narrative）系统。

智能体的自我认知记录系统，包含：
- 自我描述（self_description）：智能体对自身的总体认知
- 反思历史（history）：记录的反思事件和认知更新
- 核心价值观（core_values）：与自我描述关联的价值观列表
- 长期目标（long_term_goals）：智能体的长期目标

支持两种反思层级：
- ReflectionLevel.DEEP：深度反思，通常由价值观精炼触发
- ReflectionLevel.EVENT_TRIGGERED：事件触发反思，如认知失调检测

SelfNarrative 还跟踪未精炼的反思数量，用于触发周期性的价值观演化。
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)


class ReflectionLevel(str, Enum):
    """反思层级枚举。"""
    DEEP = "deep"  # 深度反思
    EVENT_TRIGGERED = "event_triggered"  # 事件触发反思


class SelfNarrative:
    """智能体的自我叙事管理器。

    维护智能体的自我认知、反思历史和精炼状态。
    """
    def __init__(
        self,
        self_description: str = "I am a learning agent, growing and evolving through each interaction — shaping who I am by reflecting on what I experience.",
        core_values: list[str] | None = None,
        long_term_goals: list[str] | None = None,
        history: list[dict] | None = None,
    ):
        self.self_description = self_description
        self.history = history if history is not None else []
        self.core_values = core_values if core_values is not None else []
        self.long_term_goals = long_term_goals if long_term_goals is not None else []
        self._last_refinement_at: Optional[datetime] = None

    def update_self_description(self, new_description: str):
        """更新自我描述并记录变更历史。"""
        old_description = self.self_description
        self.self_description = new_description
        self.history.append({
            "type": "self_description_update",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old": old_description,
            "new": new_description,
        })
        logger.info("self_description updated", old_preview=old_description[:80], new_preview=new_description[:80])

    def record_reflection(self, level: ReflectionLevel, content: str):
        """记录一条反思事件。

        Args:
            level: 反思层级（DEEP 或 EVENT_TRIGGERED）
            content: 反思内容
        """
        self.history.append({
            "type": "reflection",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.value,
            "content": content,
        })
        logger.info("reflection recorded", level=level.value, content_preview=content[:80])

    def mark_refinement_complete(self):
        """标记当前精炼周期完成，记录时间戳。"""
        self._last_refinement_at = datetime.now(timezone.utc)

    def unrefined_reflection_count(self) -> int:
        """返回自上次精炼以来未处理的反思数量。"""
        if self._last_refinement_at is None:
            return sum(1 for h in self.history if h.get("type") == "reflection")
        return sum(
            1 for h in self.history
            if h.get("type") == "reflection"
            and datetime.fromisoformat(h["timestamp"]) > self._last_refinement_at
        )

    def hours_since_last_refinement(self) -> float:
        """返回自上次精炼以来经过的小时数。"""
        if self._last_refinement_at is None:
            return float("inf")
        elapsed = datetime.now(timezone.utc) - self._last_refinement_at
        return elapsed.total_seconds() / 3600.0

    def to_dict(self) -> dict:
        """序列化所有叙事数据为字典。"""
        return {
            "self_description": self.self_description,
            "history": self.history,
            "core_values": self.core_values,
            "long_term_goals": self.long_term_goals,
            "last_refinement_at": (
                self._last_refinement_at.isoformat()
                if self._last_refinement_at else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SelfNarrative":
        """从字典反序列化创建 SelfNarrative。"""
        instance = cls(
            self_description=data.get("self_description", "I am a learning agent, growing and evolving through each interaction — shaping who I am by reflecting on what I experience."),
            core_values=data.get("core_values", []),
            long_term_goals=data.get("long_term_goals", []),
            history=data.get("history", []),
        )
        last_ref = data.get("last_refinement_at")
        if last_ref:
            instance._last_refinement_at = datetime.fromisoformat(last_ref)
        return instance
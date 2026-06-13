"""
价值观体系（Value System）。

智能体可演化的价值观库。每个价值观条目（ValueItem）具有：
- name：价值观名称
- direction：方向（positive 正面追求 / negative 负面抵制）
- weight：权重（0.0-1.0），反映该价值观在当前智能体中的重要性
- absolute：是否为绝对价值观（不可被自动调整）
- description：描述
- history：权重变更历史

ValueLibrary 管理所有价值观条目，支持：
- 添加/删除/查询价值观
- 调整权重（带历史记录）
- 通过 LLM 驱动的 refinement 周期自动演化价值观
- 序列化/反序列化
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValueItem:
    """单个价值观条目。

    表示智能体认同或排斥的一个价值概念。
    """
    name: str
    direction: str  # "positive"（正面追求）或 "negative"（负面抵制）
    weight: float = 0.5  # 重要性权重，0.0-1.0
    absolute: bool = False  # 是否为绝对价值观，不可被自动调整
    description: str = ""
    history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.weight = max(0.0, min(1.0, self.weight))

    def set_weight(self, new_weight: float, reason: str = ""):
        """设置新权重并记录变更历史。

        Args:
            new_weight: 新权重值（自动钳制到 [0, 1]）
            reason: 变更原因
        """
        old_weight = self.weight
        self.weight = max(0.0, min(1.0, new_weight))
        self.history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_weight": old_weight,
            "new_weight": self.weight,
            "reason": reason,
        })


class ValueLibrary:
    """价值观库。

    管理智能体的所有价值观条目，支持增删改查和批量演化。
    """
    def __init__(self):
        self._values: dict[str, ValueItem] = {}

    def add(self, value: ValueItem):
        """添加一个价值观条目，自动钳制权重。"""
        value.weight = max(0.0, min(1.0, value.weight))
        self._values[value.name] = value
        logger.debug("value added", name=value.name, direction=value.direction)

    def get(self, name: str) -> Optional[ValueItem]:
        """按名称获取价值观条目。"""
        return self._values.get(name)

    def remove(self, name: str):
        """按名称删除价值观条目。"""
        self._values.pop(name, None)
        logger.debug("value removed", name=name)

    def list_all(self) -> list[ValueItem]:
        """返回所有价值观条目的列表。"""
        return list(self._values.values())

    def update_weight(self, name: str, weight: float, reason: str = ""):
        """更新指定价值观的权重。"""
        if name in self._values:
            self._values[name].set_weight(weight, reason)
            logger.debug("value weight updated", name=name, weight=self._values[name].weight)

    def extract_metadata_for_prompt(self) -> list[dict]:
        """提取所有价值观的元数据，用于注入 LLM prompt。"""
        return [
            {
                "name": v.name,
                "direction": v.direction,
                "weight": v.weight,
                "absolute": v.absolute,
                "description": v.description,
            }
            for v in self._values.values()
        ]

    def apply_refinement(self, result: dict) -> dict:
        """应用 LLM 生成的价值观精炼结果。

        Args:
            result: LLM 返回的精炼结果，包含 value_adjustments、new_values、deprecated_values

        Returns:
            {"adjusted": [...], "added": [...], "deprecated": [...]}
        """
        changes = {"adjusted": [], "added": [], "deprecated": []}
        changes["adjusted"] = self._apply_value_adjustments(result.get("value_adjustments", []))
        changes["added"] = self._apply_new_values(result.get("new_values", []))
        changes["deprecated"] = self._apply_deprecated(result.get("deprecated_values", []))
        logger.info("value_refinement_applied", changes=changes)
        return changes

    def _apply_value_adjustments(self, adjustments: list) -> list:
        """应用权重调整，跳过绝对价值观。"""
        adjusted = []
        for adj in adjustments:
            name = adj["name"] if isinstance(adj, dict) else adj
            if isinstance(name, dict):
                name = str(name.get("name", name))
            if not (isinstance(name, str) and name in self._values and not self._values[name].absolute):
                continue
            self.update_weight(
                name,
                adj.get("new_weight", 0.5) if isinstance(adj, dict) else 0.5,
                adj.get("reason", "") if isinstance(adj, dict) else "",
            )
            adjusted.append(name)
        return adjusted

    def _apply_new_values(self, new_values: list) -> list:
        """添加新价值观，跳过已存在的。"""
        added = []
        for new_v in new_values:
            name = new_v["name"] if isinstance(new_v, dict) else str(new_v)
            if isinstance(name, dict):
                name = str(name.get("name", name))
            if not (isinstance(name, str) and name not in self._values):
                continue
            item = ValueItem(
                name=name,
                direction=new_v.get("direction", "positive") if isinstance(new_v, dict) else "positive",
                weight=new_v.get("weight", 0.5) if isinstance(new_v, dict) else 0.5,
                description=new_v.get("description", "") if isinstance(new_v, dict) else "",
            )
            self.add(item)
            added.append(name)
        return added

    def _apply_deprecated(self, deprecated: list) -> list:
        """废弃指定价值观（将权重设为 0），跳过绝对价值观。"""
        removed = []
        for raw_name in (deprecated or []):
            name = raw_name
            if isinstance(raw_name, dict):
                name = str(raw_name.get("name", raw_name))
            if not (isinstance(name, str) and name in self._values and not self._values[name].absolute):
                continue
            self._values[name].set_weight(0.0, "deprecated by value refinement")
            removed.append(name)
        return removed

    def to_dict(self) -> dict:
        """序列化所有价值观为字典，用于持久化存储。"""
        return {
            name: {
                "name": v.name,
                "direction": v.direction,
                "weight": v.weight,
                "absolute": v.absolute,
                "description": v.description,
                "history": v.history,
            }
            for name, v in self._values.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ValueLibrary":
        """从字典反序列化创建 ValueLibrary。"""
        library = cls()
        for item_data in data.values():
            library.add(ValueItem(
                name=item_data["name"],
                direction=item_data["direction"],
                weight=item_data.get("weight", 0.5),
                absolute=item_data.get("absolute", False),
                description=item_data.get("description", ""),
                history=item_data.get("history", []),
            ))
        return library

    @property
    def count(self) -> int:
        """当前价值观条目数量。"""
        return len(self._values)
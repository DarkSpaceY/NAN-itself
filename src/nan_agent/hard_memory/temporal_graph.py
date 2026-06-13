"""时间图（Temporal Graph）。

按时间戳确定性建边，支持时间窗过滤和扩展。
无需 LLM，纯规则构建。

设计参考 MAGMA 时间图：
- 同一天的记忆互连
- 相邻天之间有跨日边
- 检索时按时间窗过滤，作为 RRF 第三路信号
"""

import bisect
from collections import defaultdict
from typing import Optional

from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)


class TemporalGraph:
    """时间图：按日期索引记忆，支持时间窗过滤。

    内部结构：
        _day_index: {date_str → {mem_id}}  日期到记忆ID的映射
        _sorted_days: list[str]             排序后的日期列表（缓存）
        _mem_dates: {mem_id → date_str}     记忆ID到日期的反向映射
    """

    def __init__(self):
        self._day_index: dict[str, set[str]] = defaultdict(set)
        self._sorted_days: list[str] = []
        self._dirty: bool = False
        self._mem_dates: dict[str, str] = {}

    # ── 写入 ──────────────────────────────────────────────────

    def add(self, mem_id: str, timestamp: str) -> None:
        """添加记忆到时间图。

        Args:
            mem_id: 记忆 ID。
            timestamp: ISO 格式时间戳，如 "2023-05-08" 或 "2023-05-08T14:30:00"。
        """
        date = self._parse_date(timestamp)
        if not date:
            return
        self._day_index[date].add(mem_id)
        self._mem_dates[mem_id] = date
        self._dirty = True

    # ── 检索 ──────────────────────────────────────────────────

    def search_by_date(self, date: str, window_days: int = 0) -> list[str]:
        """返回指定日期 ±window_days 内的所有记忆 ID。

        Args:
            date: 目标日期，如 "2023-05-08" 或完整时间戳。
            window_days: 前后扩展天数，0 表示仅当天。

        Returns:
            记忆 ID 列表，按日期排序。
        """
        target = self._parse_date(date)
        if not target:
            return []

        self._ensure_sorted()

        # 二分查找目标日期位置
        idx = bisect.bisect_left(self._sorted_days, target)

        # 收集窗口内的日期
        result_ids: list[str] = []
        for i in range(max(0, idx - window_days), min(len(self._sorted_days), idx + window_days + 1)):
            day = self._sorted_days[i]
            # 确保在窗口范围内
            if abs(self._day_diff(day, target)) <= window_days or day == target:
                result_ids.extend(self._day_index[day])

        return result_ids

    def expand(self, seed_ids: list[str], window_days: int = 1) -> list[str]:
        """从种子记忆的时间戳出发，扩展同天 ±window_days 的记忆。

        用于检索增强：向量+BM25 找到种子后，用时间图补充同时段记忆。

        Args:
            seed_ids: 种子记忆 ID 列表。
            window_days: 扩展天数。

        Returns:
            扩展后的记忆 ID 列表（种子优先，去重）。
        """
        if not seed_ids:
            return []

        # 收集种子涉及的所有日期
        seed_dates: set[str] = set()
        for mid in seed_ids:
            if mid in self._mem_dates:
                seed_dates.add(self._mem_dates[mid])

        # 从每个种子日期扩展
        expanded: set[str] = set(seed_ids)
        for d in seed_dates:
            for mid in self.search_by_date(d, window_days=window_days):
                expanded.add(mid)

        # 种子优先排列
        result = list(seed_ids)
        for mid in expanded:
            if mid not in result:
                result.append(mid)
        return result

    def rank_by_temporal_proximity(
        self, query_timestamp: str, candidate_ids: list[str], half_life_days: float = 7.0
    ) -> list[tuple[str, float]]:
        """按时间接近度对候选记忆排序打分。

        使用指数衰减：score = exp(-ln(2) * |days_diff| / half_life_days)
        越接近查询时间的记忆得分越高。

        Args:
            query_timestamp: 查询的时间戳。
            candidate_ids: 候选记忆 ID。
            half_life_days: 半衰期天数，默认 7 天。

        Returns:
            [(mem_id, score)] 按分数降序。
        """
        import math

        query_date = self._parse_date(query_timestamp)
        if not query_date:
            return [(mid, 0.0) for mid in candidate_ids]

        scored: list[tuple[str, float]] = []
        decay = math.log(2) / half_life_days
        for mid in candidate_ids:
            mem_date = self._mem_dates.get(mid, "")
            if not mem_date:
                scored.append((mid, 0.0))
                continue
            diff = abs(self._day_diff(mem_date, query_date))
            score = math.exp(-decay * diff)
            scored.append((mid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ── 统计 ──────────────────────────────────────────────────

    def stats(self) -> dict:
        if not self._day_index:
            return {"days": 0, "memories": 0}
        return {
            "days": len(self._day_index),
            "memories": sum(len(ids) for ids in self._day_index.values()),
        }

    # ── 内部工具 ──────────────────────────────────────────────

    @staticmethod
    def _parse_date(timestamp: str) -> str:
        """从时间戳提取日期部分 'YYYY-MM-DD'。"""
        if not timestamp:
            return ""
        # 取前10字符 "YYYY-MM-DD"
        date = timestamp[:10]
        if len(date) == 10 and date[4] == "-" and date[7] == "-":
            return date
        return ""

    @staticmethod
    def _day_diff(date_a: str, date_b: str) -> int:
        """计算两个日期之间的天数差。"""
        from datetime import date as date_cls
        try:
            ya, ma, da = int(date_a[:4]), int(date_a[5:7]), int(date_a[8:10])
            yb, mb, db = int(date_b[:4]), int(date_b[5:7]), int(date_b[8:10])
            return (date_cls(ya, ma, da) - date_cls(yb, mb, db)).days
        except (ValueError, IndexError):
            return 0

    def _ensure_sorted(self) -> None:
        """确保 _sorted_days 已排序（惰性计算）。"""
        if self._dirty:
            self._sorted_days = sorted(self._day_index.keys())
            self._dirty = False

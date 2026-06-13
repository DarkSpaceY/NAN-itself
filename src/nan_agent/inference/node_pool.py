from collections import Counter
from datetime import datetime, timezone
from typing import Optional

from nan_agent.inference.graph import GoTNode, NodeType


class NodePool:
    """Active node pool that maintains currently active reasoning nodes.

    Nodes are ordered by priority score for batch selection.
    Higher priority nodes are selected first by pop_top_k / peek_top_k.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, GoTNode] = {}
        self._order: list[str] = []

    @property
    def nodes(self) -> dict[str, GoTNode]:
        return self._nodes

    def add(self, node: GoTNode) -> None:
        self._nodes[node.node_id] = node
        self._order.append(node.node_id)

    def add_batch(self, nodes: list[GoTNode]) -> None:
        for node in nodes:
            self.add(node)

    def pop_top_k(self, k: int = 5) -> list[GoTNode]:
        if not self._nodes:
            return []
        sorted_ids = sorted(
            self._nodes.keys(),
            key=lambda nid: self._priority_score(self._nodes[nid]),
            reverse=True,
        )
        result_ids = sorted_ids[:k]
        result = [self._nodes[nid] for nid in result_ids]
        for nid in result_ids:
            del self._nodes[nid]
            self._order.remove(nid)
        return result

    def peek_top_k(self, k: int = 5) -> list[GoTNode]:
        if not self._nodes:
            return []
        sorted_ids = sorted(
            self._nodes.keys(),
            key=lambda nid: self._priority_score(self._nodes[nid]),
            reverse=True,
        )
        return [self._nodes[nid] for nid in sorted_ids[:k]]

    def size(self) -> int:
        return len(self._nodes)

    def is_empty(self) -> bool:
        return len(self._nodes) == 0

    def contains(self, node_id: str) -> bool:
        return node_id in self._nodes

    def remove(self, node_id: str) -> Optional[GoTNode]:
        node = self._nodes.pop(node_id, None)
        if node is not None and node_id in self._order:
            self._order.remove(node_id)
        return node

    def clear(self) -> None:
        self._nodes.clear()
        self._order.clear()

    def get_statistics(self) -> dict:
        type_counter: Counter = Counter()
        for node in self._nodes.values():
            type_counter[node.type] += 1
        return {"total": self.size(), "by_type": {t.value: c for t, c in type_counter.items()}}

    def get_by_type(self, node_type: NodeType) -> list[GoTNode]:
        return [node for node in self._nodes.values() if node.type == node_type]

    def age_analysis(self) -> dict:
        if not self._nodes:
            return {"oldest_seconds": None, "newest_seconds": None, "average_age_seconds": 0.0}
        now = datetime.now(timezone.utc)
        ages = [(now - node.created_at).total_seconds() for node in self._nodes.values()]
        return {
            "oldest_seconds": max(ages),
            "newest_seconds": min(ages),
            "average_age_seconds": sum(ages) / len(ages),
        }

    def _priority_score(self, node: GoTNode) -> float:
        if node.pruned:
            return -100.0

        base = max(0.0, min(1.0, node.confidence))

        if node.type == NodeType.ACTION_OUTPUT:
            base += 0.4
        elif node.type == NodeType.EXTERNAL_TASK:
            base += 0.5
        elif node.type == NodeType.DMN_SPONTANEOUS:
            base -= 0.1

        if self._order and node.node_id in self._order:
            idx = self._order.index(node.node_id)
            max_recency = max(1, len(self._order) - 1)
            recency_boost = 0.01 * (idx / max_recency)
        else:
            recency_boost = 0.0

        return base + recency_boost
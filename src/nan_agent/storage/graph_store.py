import asyncio
from typing import Any, Optional

from nan_agent.exceptions import StorageError
from nan_agent.logging.logger import get_logger
from nan_agent.storage.base import BaseStore

logger = get_logger(__name__)


class GraphStore(BaseStore):
    def __init__(self):
        self._graph = None
        self._lock = asyncio.Lock()

    def _get_graph(self):
        if self._graph is None:
            import networkx as nx

            self._graph = nx.DiGraph()
        return self._graph

    async def add_node(self, node_id: str, attrs: Optional[dict] = None) -> None:
        async with self._lock:
            try:
                g = self._get_graph()
                g.add_node(node_id, **(attrs or {}))
                logger.debug("node_added", node_id=node_id)
            except Exception as e:
                raise StorageError(
                    f"Failed to add node '{node_id}': {e}",
                    error_code="E311",
                    details={"node_id": node_id},
                ) from e

    async def add_edge(
        self,
        from_id: str,
        to_id: str,
        attrs: Optional[dict] = None,
    ) -> None:
        async with self._lock:
            try:
                g = self._get_graph()
                if not g.has_node(from_id):
                    g.add_node(from_id)
                if not g.has_node(to_id):
                    g.add_node(to_id)
                g.add_edge(from_id, to_id, **(attrs or {}))
                logger.debug("edge_added", from_id=from_id, to_id=to_id)
            except Exception as e:
                raise StorageError(
                    f"Failed to add edge '{from_id}' -> '{to_id}': {e}",
                    error_code="E312",
                    details={"from_id": from_id, "to_id": to_id},
                ) from e

    async def get_node(self, node_id: str) -> Optional[dict]:
        async with self._lock:
            try:
                g = self._get_graph()
                if not g.has_node(node_id):
                    return None
                attrs = dict(g.nodes[node_id])
                attrs["id"] = node_id
                return attrs
            except Exception as e:
                raise StorageError(
                    f"Failed to get node '{node_id}': {e}",
                    error_code="E313",
                    details={"node_id": node_id},
                ) from e

    async def remove_node(self, node_id: str) -> None:
        async with self._lock:
            try:
                g = self._get_graph()
                if g.has_node(node_id):
                    g.remove_node(node_id)
                    logger.debug("node_removed", node_id=node_id)
            except Exception as e:
                raise StorageError(
                    f"Failed to remove node '{node_id}': {e}",
                    error_code="E316",
                    details={"node_id": node_id},
                ) from e

    async def remove_edge(self, from_id: str, to_id: str) -> None:
        async with self._lock:
            try:
                g = self._get_graph()
                if g.has_edge(from_id, to_id):
                    g.remove_edge(from_id, to_id)
                    logger.debug("edge_removed", from_id=from_id, to_id=to_id)
            except Exception as e:
                raise StorageError(
                    f"Failed to remove edge '{from_id}' -> '{to_id}': {e}",
                    error_code="E317",
                    details={"from_id": from_id, "to_id": to_id},
                ) from e

    async def node_count(self) -> int:
        async with self._lock:
            try:
                g = self._get_graph()
                return g.number_of_nodes()
            except Exception as e:
                raise StorageError(
                    f"Failed to count nodes: {e}",
                    error_code="E318",
                ) from e

    async def shortest_path(
        self,
        from_id: str,
        to_id: str,
    ) -> Optional[list[str]]:
        async with self._lock:
            try:
                g = self._get_graph()
                import networkx as nx

                return nx.shortest_path(g, source=from_id, target=to_id)
            except nx.NetworkXNoPath:
                return None
            except nx.NodeNotFound as e:
                raise StorageError(
                    f"Node not found: {e}",
                    error_code="E320",
                    details={"from_id": from_id, "to_id": to_id},
                ) from e
            except Exception as e:
                raise StorageError(
                    f"Failed to find shortest path: {e}",
                    error_code="E321",
                    details={"from_id": from_id, "to_id": to_id},
                ) from e

    async def export_data(self) -> dict[str, Any]:
        async with self._lock:
            try:
                g = self._get_graph()
                import networkx as nx

                return nx.node_link_data(g)
            except Exception as e:
                raise StorageError(
                    f"Failed to export graph data: {e}",
                    error_code="E322",
                ) from e

    async def import_data(self, data: dict[str, Any]) -> None:
        async with self._lock:
            try:
                import networkx as nx

                self._graph = nx.node_link_graph(data)
                logger.info("graph_imported")
            except Exception as e:
                raise StorageError(
                    f"Failed to import graph data: {e}",
                    error_code="E323",
                ) from e

    async def close(self) -> None:
        self._graph = None

    async def health_check(self) -> bool:
        try:
            self._get_graph()
            return True
        except Exception:
            return False
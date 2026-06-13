import asyncio
from typing import Any, Optional

from nan_agent.exceptions import StorageError
from nan_agent.logging.logger import get_logger
from nan_agent.storage.base import BaseStore

logger = get_logger(__name__)


class VectorStore(BaseStore):
    def __init__(
        self,
        persist_directory: str = "./data/chromadb",
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        self._persist_directory = persist_directory
        self._host = host
        self._port = port
        self._client = None
        self._lock = asyncio.Lock()

    def _get_client(self):
        if self._client is None:
            import chromadb

            if self._host and self._port:
                self._client = chromadb.HttpClient(
                    host=self._host,
                    port=self._port,
                )
            else:
                self._client = chromadb.PersistentClient(
                    path=self._persist_directory,
                )
        return self._client

    async def create_collection(self, name: str, metadata: Optional[dict] = None) -> None:
        async with self._lock:
            try:
                client = self._get_client()
                client.create_collection(name=name, metadata=metadata)
                logger.info("collection_created", collection=name)
            except Exception as e:
                raise StorageError(
                    f"Failed to create collection '{name}': {e}",
                    error_code="E301",
                    details={"collection": name},
                ) from e

    async def ensure_collection(self, name: str, metadata: Optional[dict] = None) -> None:
        async with self._lock:
            try:
                client = self._get_client()
                existing = [col.name for col in client.list_collections()]
                if name not in existing:
                    client.create_collection(name=name, metadata=metadata)
                    logger.info("collection_ensured", collection=name)
            except Exception as e:
                raise StorageError(
                    f"Failed to ensure collection '{name}': {e}",
                    error_code="E301",
                    details={"collection": name},
                ) from e

    async def list_collections(self) -> list[str]:
        async with self._lock:
            try:
                client = self._get_client()
                return [col.name for col in client.list_collections()]
            except Exception as e:
                raise StorageError(
                    f"Failed to list collections: {e}",
                    error_code="E302",
                ) from e

    async def add(
        self,
        collection: str,
        id: str,
        embedding: list[float],
        metadata: Optional[dict] = None,
    ) -> None:
        async with self._lock:
            try:
                client = self._get_client()
                col = client.get_or_create_collection(name=collection)
                col.add(
                    ids=[id],
                    embeddings=[embedding],
                    metadatas=[metadata] if metadata else None,
                )
                logger.debug("vector_added", collection=collection, id=id)
            except Exception as e:
                raise StorageError(
                    f"Failed to add vector '{id}' to collection '{collection}': {e}",
                    error_code="E303",
                    details={"collection": collection, "id": id},
                ) from e

    async def search(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        async with self._lock:
            try:
                client = self._get_client()
                col = client.get_or_create_collection(name=collection)
                where_filter = filters if filters else None
                results = col.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_filter,
                )
                items = []
                ids_list = results.get("ids", [[]])[0]
                distances = results.get("distances", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]

                for i, item_id in enumerate(ids_list):
                    item = {"id": item_id}
                    if distances and i < len(distances):
                        item["distance"] = distances[i]
                    if metadatas and i < len(metadatas):
                        item["metadata"] = metadatas[i]
                    items.append(item)

                return items
            except Exception as e:
                raise StorageError(
                    f"Failed to search collection '{collection}': {e}",
                    error_code="E304",
                    details={"collection": collection},
                ) from e

    async def delete(self, collection: str, id: str) -> None:
        async with self._lock:
            try:
                client = self._get_client()
                col = client.get_or_create_collection(name=collection)
                col.delete(ids=[id])
                logger.debug("vector_deleted", collection=collection, id=id)
            except Exception as e:
                raise StorageError(
                    f"Failed to delete vector '{id}' from collection '{collection}': {e}",
                    error_code="E305",
                    details={"collection": collection, "id": id},
                ) from e

    async def count(self, collection: str) -> int:
        async with self._lock:
            try:
                client = self._get_client()
                col = client.get_or_create_collection(name=collection)
                return col.count()
            except Exception as e:
                raise StorageError(
                    f"Failed to count collection '{collection}': {e}",
                    error_code="E306",
                    details={"collection": collection},
                ) from e

    async def close(self) -> None:
        self._client = None

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            client.list_collections()
            return True
        except Exception:
            return False
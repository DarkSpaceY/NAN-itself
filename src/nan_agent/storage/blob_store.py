import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from nan_agent.exceptions import StorageError
from nan_agent.logging.logger import get_logger
from nan_agent.storage.base import BaseStore

logger = get_logger(__name__)

META_SUFFIX = ".meta.json"


class BlobStore(BaseStore):
    def __init__(self, base_dir: str = "./data/blobs"):
        self._base_dir = Path(base_dir)
        self._lock = asyncio.Lock()

    def _ensure_dir(self) -> None:
        os.makedirs(self._base_dir, exist_ok=True)

    def _blob_path(self, key: str) -> Path:
        return self._base_dir / key

    def _meta_path(self, key: str) -> Path:
        return self._base_dir / f"{key}{META_SUFFIX}"

    async def put(self, key: str, data: bytes | str, metadata: Optional[dict] = None) -> None:
        """Store blob data with optional metadata.

        Args:
            key: Unique identifier for the blob.
            data: Binary or string data to store.
            metadata: Optional metadata dictionary to store alongside the blob.
        """
        async with self._lock:
            try:
                self._ensure_dir()
                blob_path = self._blob_path(key)

                # Ensure parent directory exists
                blob_path.parent.mkdir(parents=True, exist_ok=True)

                # Write blob data
                if isinstance(data, str):
                    blob_path.write_text(data, encoding="utf-8")
                else:
                    blob_path.write_bytes(data)

                # Write metadata if provided
                if metadata is not None:
                    meta_path = self._meta_path(key)
                    meta_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

                logger.debug("blob_stored", key=key, size=len(data) if isinstance(data, bytes) else len(data))
            except Exception as e:
                raise StorageError(
                    f"Failed to store blob '{key}': {e}",
                    error_code="E341",
                    details={"key": key},
                ) from e

    async def get_metadata(self, key: str) -> Optional[dict]:
        """Retrieve metadata for a blob.

        Args:
            key: Unique identifier for the blob.

        Returns:
            Metadata dictionary if exists, None otherwise.
        """
        async with self._lock:
            try:
                meta_path = self._meta_path(key)
                if not meta_path.exists():
                    return None
                content = meta_path.read_text(encoding="utf-8")
                return json.loads(content)
            except Exception as e:
                raise StorageError(
                    f"Failed to get metadata for blob '{key}': {e}",
                    error_code="E343",
                    details={"key": key},
                ) from e

    async def get(self, key: str) -> Optional[bytes]:
        async with self._lock:
            try:
                blob_path = self._blob_path(key)
                if not blob_path.exists():
                    return None
                return blob_path.read_bytes()
            except Exception as e:
                raise StorageError(
                    f"Failed to get blob '{key}': {e}",
                    error_code="E342",
                    details={"key": key},
                ) from e

    async def delete(self, key: str) -> None:
        async with self._lock:
            try:
                blob_path = self._blob_path(key)
                if blob_path.exists():
                    if blob_path.is_dir():
                        shutil.rmtree(blob_path)
                    else:
                        blob_path.unlink()

                meta_path = self._meta_path(key)
                if meta_path.exists():
                    meta_path.unlink()

                logger.debug("blob_deleted", key=key)
            except Exception as e:
                raise StorageError(
                    f"Failed to delete blob '{key}': {e}",
                    error_code="E344",
                    details={"key": key},
                ) from e

    async def exists(self, key: str) -> bool:
        async with self._lock:
            try:
                return self._blob_path(key).exists()
            except Exception as e:
                raise StorageError(
                    f"Failed to check existence of blob '{key}': {e}",
                    error_code="E345",
                    details={"key": key},
                ) from e

    async def list(self, prefix: str = "") -> list[str]:
        async with self._lock:
            try:
                self._ensure_dir()
                result = []
                for root, _dirs, files in os.walk(self._base_dir):
                    for f in files:
                        if f.endswith(META_SUFFIX):
                            continue
                        rel_path = os.path.relpath(os.path.join(root, f), self._base_dir)
                        if not prefix or rel_path.startswith(prefix):
                            result.append(rel_path)
                return sorted(result)
            except Exception as e:
                raise StorageError(
                    f"Failed to list blobs with prefix '{prefix}': {e}",
                    error_code="E346",
                    details={"prefix": prefix},
                ) from e

    async def size(self, key: str) -> Optional[int]:
        async with self._lock:
            try:
                blob_path = self._blob_path(key)
                if not blob_path.exists():
                    return None
                return blob_path.stat().st_size
            except Exception as e:
                raise StorageError(
                    f"Failed to get size of blob '{key}': {e}",
                    error_code="E347",
                    details={"key": key},
                ) from e

    async def close(self) -> None:
        pass

    async def health_check(self) -> bool:
        try:
            self._ensure_dir()
            return os.access(self._base_dir, os.W_OK)
        except Exception:
            return False
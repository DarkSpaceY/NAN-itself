import asyncio
import fnmatch
import json
import time
from typing import Any, Optional

from nan_agent.exceptions import StorageError
from nan_agent.logging.logger import get_logger
from nan_agent.storage.base import BaseStore

logger = get_logger(__name__)


class StateStore(BaseStore):
    def __init__(self, db_path: str = "./data/state.db"):
        self._db_path = db_path
        self._conn = None
        self._lock = asyncio.Lock()

    async def _get_conn(self):
        if self._conn is None:
            import aiosqlite

            self._conn = await aiosqlite.connect(self._db_path)
            await self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL
                )
                """
            )
            await self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_state_expires ON state(expires_at)"
            )
            await self._conn.commit()
        return self._conn

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        async with self._lock:
            try:
                conn = await self._get_conn()
                serialized = json.dumps(value)
                expires_at = None
                if ttl_seconds is not None:
                    expires_at = time.time() + ttl_seconds
                await conn.execute(
                    "INSERT OR REPLACE INTO state (key, value, expires_at) VALUES (?, ?, ?)",
                    (key, serialized, expires_at),
                )
                await conn.commit()
                logger.debug("state_set", key=key)
            except Exception as e:
                raise StorageError(
                    f"Failed to set key '{key}': {e}",
                    error_code="E331",
                    details={"key": key},
                ) from e

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            try:
                conn = await self._get_conn()
                cursor = await conn.execute(
                    "SELECT value, expires_at FROM state WHERE key = ?", (key,)
                )
                row = await cursor.fetchone()
                if row is None:
                    return None
                value_str, expires_at = row
                if expires_at is not None and time.time() > expires_at:
                    await conn.execute("DELETE FROM state WHERE key = ?", (key,))
                    await conn.commit()
                    return None
                return json.loads(value_str)
            except Exception as e:
                raise StorageError(
                    f"Failed to get key '{key}': {e}",
                    error_code="E332",
                    details={"key": key},
                ) from e

    async def delete(self, key: str) -> None:
        async with self._lock:
            try:
                conn = await self._get_conn()
                await conn.execute("DELETE FROM state WHERE key = ?", (key,))
                await conn.commit()
                logger.debug("state_deleted", key=key)
            except Exception as e:
                raise StorageError(
                    f"Failed to delete key '{key}': {e}",
                    error_code="E333",
                    details={"key": key},
                ) from e

    async def exists(self, key: str) -> bool:
        async with self._lock:
            try:
                conn = await self._get_conn()
                cursor = await conn.execute(
                    "SELECT expires_at FROM state WHERE key = ?", (key,)
                )
                row = await cursor.fetchone()
                if row is None:
                    return False
                expires_at = row[0]
                if expires_at is not None and time.time() > expires_at:
                    await conn.execute("DELETE FROM state WHERE key = ?", (key,))
                    await conn.commit()
                    return False
                return True
            except Exception as e:
                raise StorageError(
                    f"Failed to check existence of key '{key}': {e}",
                    error_code="E334",
                    details={"key": key},
                ) from e

    async def keys(self, pattern: str = "*") -> list[str]:
        async with self._lock:
            try:
                conn = await self._get_conn()
                now = time.time()
                await conn.execute(
                    "DELETE FROM state WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (now,),
                )
                await conn.commit()
                cursor = await conn.execute("SELECT key FROM state")
                rows = await cursor.fetchall()
                all_keys = [row[0] for row in rows]
                return fnmatch.filter(all_keys, pattern)
            except Exception as e:
                raise StorageError(
                    f"Failed to list keys with pattern '{pattern}': {e}",
                    error_code="E335",
                    details={"pattern": pattern},
                ) from e

    async def expire(self, key: str, ttl_seconds: int) -> None:
        async with self._lock:
            try:
                conn = await self._get_conn()
                expires_at = time.time() + ttl_seconds
                cursor = await conn.execute(
                    "UPDATE state SET expires_at = ? WHERE key = ?",
                    (expires_at, key),
                )
                if cursor.rowcount == 0:
                    raise StorageError(
                        f"Key '{key}' not found",
                        error_code="E336",
                        details={"key": key},
                    )
                await conn.commit()
                logger.debug("state_expire_set", key=key, ttl=ttl_seconds)
            except StorageError:
                raise
            except Exception as e:
                raise StorageError(
                    f"Failed to set TTL for key '{key}': {e}",
                    error_code="E336",
                    details={"key": key},
                ) from e

    async def clear(self) -> None:
        async with self._lock:
            try:
                conn = await self._get_conn()
                await conn.execute("DELETE FROM state")
                await conn.commit()
                logger.info("state_cleared")
            except Exception as e:
                raise StorageError(
                    f"Failed to clear state: {e}",
                    error_code="E338",
                ) from e

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def health_check(self) -> bool:
        try:
            conn = await self._get_conn()
            await conn.execute("SELECT 1")
            return True
        except Exception:
            return False
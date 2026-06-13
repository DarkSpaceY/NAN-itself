"""
Agent 文件系统 - 沙盒化的工作空间文件管理

提供 Agent 专属的文件系统操作能力，包括文件的读写、复制、移动、删除、搜索等功能。
所有操作限制在 workspace_root 范围内，防止路径越狱。支持配额管理、文件监视和临时文件清理。

核心组件：
- AgentFileSystem: 文件系统操作主类
- FileInfo: 文件元数据
- QuotaInfo: 配额信息
"""

import asyncio
import fnmatch
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any, Optional

from nan_agent.exceptions import ActionError
from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)

DEFAULT_WORKSPACE_ROOT = os.path.expanduser("~/.nan-agent/workspace")
DEFAULT_QUOTA_BYTES = 500 * 1024 * 1024
DEFAULT_MAX_TEMP_AGE = 3600

FILE_TYPE_MAP = {
    ".txt": "text",
    ".md": "markdown",
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".csv": "csv",
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".svg": "image",
    ".log": "log",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "config",
    ".conf": "config",
}


@dataclass
class FileInfo:
    path: str
    name: str
    size: int
    type: str
    created_at: float
    is_directory: bool

    @classmethod
    def from_path(cls, file_path: Path, workspace_root: Path) -> "FileInfo":
        st = file_path.stat()
        is_dir = file_path.is_dir()
        rel_path = str(file_path.relative_to(workspace_root))
        ext = file_path.suffix.lower()
        file_type = FILE_TYPE_MAP.get(ext, "unknown")
        if is_dir:
            file_type = "directory"
        return cls(
            path=rel_path,
            name=file_path.name,
            size=st.st_size,
            type=file_type,
            created_at=st.st_ctime,
            is_directory=is_dir,
        )

@dataclass
class QuotaInfo:
    used_bytes: int
    limit_bytes: int
    file_count: int


class AgentFileSystem:
    """Agent 沙盒文件系统。

    所有文件操作限制在 workspace_root 目录内，通过路径解析检测越狱尝试。
    支持配额管理、文件监视回调、临时文件自动清理。

    Attributes:
        workspace_root: 工作空间根目录
    """

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        quota_bytes: int = DEFAULT_QUOTA_BYTES,
        max_temp_age: int = DEFAULT_MAX_TEMP_AGE,
    ):
        """初始化文件系统。

        Args:
            workspace_root: 工作空间根目录路径，默认为 ~/.nan-agent/workspace
            quota_bytes: 磁盘配额上限（字节），默认 500MB
            max_temp_age: 临时文件最大保留时间（秒），默认 3600
        """
        if workspace_root is None:
            workspace_root = DEFAULT_WORKSPACE_ROOT
        self._workspace_root = Path(workspace_root).resolve()
        self._quota_bytes = quota_bytes
        self._max_temp_age = max_temp_age
        self._temp_dir = self._workspace_root / ".tmp"
        self._lock = asyncio.Lock()
        self._watchers: dict[str, list[Callable[[str, str], Any]]] = {}
        self._ensure_dir(self._workspace_root)
        self._ensure_dir(self._temp_dir)

    @property
    def workspace_root(self) -> Path:
        return self._workspace_root

    def _ensure_dir(self, path: Path) -> None:
        os.makedirs(path, exist_ok=True)

    def _resolve_path(self, relative_path: str) -> Path:
        resolved = (self._workspace_root / relative_path).resolve()
        if not str(resolved).startswith(str(self._workspace_root)):
            raise ActionError(
                f"Path escapes workspace: '{relative_path}'",
                error_code="E510",
                details={"path": relative_path},
            )
        return resolved

    def _check_path_exists(self, path: Path, relative_path: str) -> None:
        if not path.exists():
            raise ActionError(
                f"Path not found: '{relative_path}'",
                error_code="E511",
                details={"path": relative_path},
            )

    def _notify_watchers(self, event: str, relative_path: str) -> None:
        for pattern, callbacks in self._watchers.items():
            if fnmatch.fnmatch(relative_path, pattern):
                for cb in callbacks:
                    try:
                        cb(event, relative_path)
                    except Exception:
                        logger.exception(
                            "watcher_callback_error",
                            event=event,
                            path=relative_path,
                        )

    def _calculate_dir_size(self, path: Path) -> int:
        total = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception as e:
            logger.debug("filesystem_dir_size_failed", path=str(path), error=str(e))
        return total

    def _calc_quota_unsafe(self) -> QuotaInfo:
        used = 0
        count = 0
        try:
            for entry in self._workspace_root.rglob("*"):
                if entry.is_file():
                    used += entry.stat().st_size
                    count += 1
        except Exception:
            logger.warning("quota_calculation_error")
        return QuotaInfo(used_bytes=used, limit_bytes=self._quota_bytes, file_count=count)

    def _check_quota_unsafe(self, additional_bytes: int = 0) -> None:
        current = self._calc_quota_unsafe()
        if current.used_bytes + additional_bytes > self._quota_bytes:
            raise ActionError(
                f"Quota exceeded: {current.used_bytes + additional_bytes} / {self._quota_bytes} bytes",
                error_code="E513",
                details={
                    "used_bytes": current.used_bytes,
                    "limit_bytes": self._quota_bytes,
                    "additional_bytes": additional_bytes,
                },
            )

    async def read_file(self, path: str, mode: str = "r") -> str | bytes:
        """读取文件内容。

        Args:
            path: 相对于 workspace_root 的文件路径
            mode: 读取模式，"r" 文本模式，"rb" 二进制模式

        Returns:
            文件内容（文本模式返回 str，二进制模式返回 bytes）

        Raises:
            ActionError: 路径越狱、文件不存在或为目录时抛出
        """
        async with self._lock:
            resolved = self._resolve_path(path)
            self._check_path_exists(resolved, path)
            if resolved.is_dir():
                raise ActionError(
                    f"Cannot read directory as file: '{path}'",
                    error_code="E512",
                    details={"path": path},
                )
            try:
                if "b" in mode:
                    content = resolved.read_bytes()
                else:
                    content = resolved.read_text(encoding="utf-8")
                logger.debug("file_read", path=path)
                return content
            except Exception as e:
                raise ActionError(
                    f"Failed to read file '{path}': {e}",
                    error_code="E520",
                    details={"path": path},
                ) from e

    async def write_file(
        self,
        path: str,
        content: str | bytes,
        mode: str = "w",
        create_parents: bool = True,
    ) -> None:
        """写入文件（创建或覆盖）。

        Args:
            path: 相对于 workspace_root 的文件路径
            content: 文件内容（str 或 bytes）
            mode: 写入模式，"w" 覆盖，"a" 追加
            create_parents: 是否自动创建父目录

        Raises:
            ActionError: 路径越狱或配额超限时抛出
        """
        async with self._lock:
            resolved = self._resolve_path(path)
            if create_parents:
                self._ensure_dir(resolved.parent)

            if isinstance(content, bytes):
                data_len = len(content)
            else:
                data_len = len(content.encode("utf-8"))

            if resolved.exists():
                existing_size = resolved.stat().st_size if resolved.is_file() else 0
                self._check_quota_unsafe(data_len - existing_size)
            else:
                self._check_quota_unsafe(data_len)

            file_existed = resolved.exists()
            try:
                if isinstance(content, bytes):
                    if "b" not in mode:
                        mode = mode + "b"
                    with open(str(resolved), mode) as f:
                        f.write(content)
                else:
                    resolved.write_text(content, encoding="utf-8")
                logger.debug("file_written", path=path)
                self._notify_watchers("modified" if file_existed else "created", path)
            except Exception as e:
                raise ActionError(
                    f"Failed to write file '{path}': {e}",
                    error_code="E521",
                    details={"path": path},
                ) from e

    async def delete(self, path: str, recursive: bool = False) -> None:
        async with self._lock:
            resolved = self._resolve_path(path)
            self._check_path_exists(resolved, path)
            try:
                if resolved.is_dir():
                    if not recursive:
                        raise ActionError(
                            f"Use recursive=True to delete directory: '{path}'",
                            error_code="E514",
                            details={"path": path},
                        )
                    shutil.rmtree(resolved)
                else:
                    resolved.unlink()
                logger.debug("file_deleted", path=path)
                self._notify_watchers("deleted", path)
            except ActionError:
                raise
            except Exception as e:
                raise ActionError(
                    f"Failed to delete '{path}': {e}",
                    error_code="E522",
                    details={"path": path},
                ) from e

    async def copy(self, src: str, dst: str, overwrite: bool = False) -> None:
        async with self._lock:
            resolved_src = self._resolve_path(src)
            resolved_dst = self._resolve_path(dst)
            self._check_path_exists(resolved_src, src)

            if resolved_dst.exists() and not overwrite:
                raise ActionError(
                    f"Destination already exists: '{dst}'",
                    error_code="E515",
                    details={"destination": dst},
                )

            if resolved_src.is_dir():
                additional = self._calculate_dir_size(resolved_src)
            else:
                additional = resolved_src.stat().st_size
            self._check_quota_unsafe(additional)

            try:
                if resolved_src.is_dir():
                    if resolved_dst.exists() and overwrite:
                        shutil.rmtree(resolved_dst)
                    shutil.copytree(resolved_src, resolved_dst)
                else:
                    self._ensure_dir(resolved_dst.parent)
                    shutil.copy2(resolved_src, resolved_dst)
                logger.debug("file_copied", src=src, dst=dst)
                self._notify_watchers("created", dst)
            except Exception as e:
                raise ActionError(
                    f"Failed to copy '{src}' to '{dst}': {e}",
                    error_code="E523",
                    details={"src": src, "dst": dst},
                ) from e

    async def move(self, src: str, dst: str, overwrite: bool = False) -> None:
        async with self._lock:
            resolved_src = self._resolve_path(src)
            resolved_dst = self._resolve_path(dst)
            self._check_path_exists(resolved_src, src)

            if resolved_dst.exists() and not overwrite:
                raise ActionError(
                    f"Destination already exists: '{dst}'",
                    error_code="E515",
                    details={"destination": dst},
                )

            try:
                self._ensure_dir(resolved_dst.parent)
                shutil.move(str(resolved_src), str(resolved_dst))
                logger.debug("file_moved", src=src, dst=dst)
                self._notify_watchers("deleted", src)
                self._notify_watchers("created", dst)
            except Exception as e:
                raise ActionError(
                    f"Failed to move '{src}' to '{dst}': {e}",
                    error_code="E524",
                    details={"src": src, "dst": dst},
                ) from e

    async def exists(self, path: str) -> bool:
        async with self._lock:
            resolved = self._resolve_path(path)
            return resolved.exists()

    async def list_directory(
        self, path: str = ".", recursive: bool = False
    ) -> list[FileInfo]:
        async with self._lock:
            resolved = self._resolve_path(path)
            if not resolved.exists():
                raise ActionError(
                    f"Directory not found: '{path}'",
                    error_code="E511",
                    details={"path": path},
                )
            if not resolved.is_dir():
                raise ActionError(
                    f"Not a directory: '{path}'",
                    error_code="E516",
                    details={"path": path},
                )
            result = []
            try:
                iterator = resolved.rglob("*") if recursive else resolved.glob("*")
                for entry in sorted(iterator, key=lambda p: (not p.is_dir(), p.name.lower())):
                    try:
                        result.append(FileInfo.from_path(entry, self._workspace_root))
                    except Exception:
                        logger.warning("metadata_read_error", path=str(entry))
            except Exception as e:
                raise ActionError(
                    f"Failed to list directory '{path}': {e}",
                    error_code="E525",
                    details={"path": path},
                ) from e
            return result

    async def create_directory(
        self, path: str, parents: bool = True
    ) -> None:
        async with self._lock:
            resolved = self._resolve_path(path)
            if resolved.exists():
                raise ActionError(
                    f"Directory already exists: '{path}'",
                    error_code="E517",
                    details={"path": path},
                )
            try:
                if parents:
                    self._ensure_dir(resolved)
                else:
                    resolved.mkdir()
                logger.debug("directory_created", path=path)
                self._notify_watchers("created", path)
            except Exception as e:
                raise ActionError(
                    f"Failed to create directory '{path}': {e}",
                    error_code="E526",
                    details={"path": path},
                ) from e

    async def search(
        self,
        pattern: str = "*",
        content_pattern: Optional[str] = None,
        recursive: bool = True,
        max_results: int = 100,
    ) -> list[FileInfo]:
        async with self._lock:
            results: list[FileInfo] = []
            iterator = self._workspace_root.rglob("*") if recursive else self._workspace_root.glob("*")
            for entry in iterator:
                if len(results) >= max_results:
                    break
                if entry.is_dir():
                    continue
                rel_name = str(entry.relative_to(self._workspace_root))
                if not fnmatch.fnmatch(rel_name, pattern) and not fnmatch.fnmatch(
                    entry.name, pattern
                ):
                    continue
                if content_pattern:
                    try:
                        content_text = entry.read_text(encoding="utf-8", errors="ignore")
                        if content_pattern not in content_text:
                            continue
                    except Exception as e:
                        logger.warning("search_content_read_failed", path=str(entry), error=str(e))
                        continue
                try:
                    results.append(FileInfo.from_path(entry, self._workspace_root))
                except Exception:
                    logger.warning("search_metadata_error", path=str(entry))
            return results

    async def get_file_info(self, path: str) -> FileInfo:
        async with self._lock:
            resolved = self._resolve_path(path)
            self._check_path_exists(resolved, path)
            try:
                return FileInfo.from_path(resolved, self._workspace_root)
            except Exception as e:
                raise ActionError(
                    f"Failed to get file info for '{path}': {e}",
                    error_code="E527",
                    details={"path": path},
                ) from e

    async def get_quota_info(self) -> QuotaInfo:
        async with self._lock:
            return self._calc_quota_unsafe()

    def watch(
        self,
        pattern: str,
        callback: Callable[[str, str], Any],
    ) -> None:
        if pattern not in self._watchers:
            self._watchers[pattern] = []
        self._watchers[pattern].append(callback)
        logger.debug("watcher_registered", pattern=pattern)

    async def cleanup_temp_files(self, max_age: Optional[int] = None) -> int:
        if max_age is None:
            max_age = self._max_temp_age
        async with self._lock:
            removed = 0
            now = time.time()
            try:
                for entry in self._temp_dir.iterdir():
                    if entry.is_file():
                        age = now - entry.stat().st_mtime
                        if age > max_age:
                            entry.unlink()
                            removed += 1
                            logger.debug("temp_file_cleaned", path=str(entry))
            except Exception as e:
                raise ActionError(
                    f"Failed to cleanup temp files: {e}",
                    error_code="E529",
                    details={},
                ) from e
            return removed

    async def close(self) -> None:
        async with self._lock:
            self._watchers.clear()

    async def health_check(self) -> bool:
        try:
            return os.access(self._workspace_root, os.W_OK)
        except Exception:
            return False
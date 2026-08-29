# @module

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any


class WorkspaceModule(Module):
    id = "workspace"

    SCAN_INTERVAL = 1.0

    IGNORED_DIRS = frozenset({
        ".git",
        ".idea",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "__pycache__",
        "data",
    })

    def __init__(self) -> None:
        self.root = Path.cwd().resolve()

        # Private Module state.
        self._files: dict[str, tuple[int, int]] = {}
        self._scan_count = 0

    async def start(self) -> None:
        while True:
            await self._scan_once()
            await asyncio.sleep(self.SCAN_INTERVAL)

    async def stop(self) -> None:
        pass

    async def _scan_once(self) -> None:
        current = await asyncio.to_thread(
            self._scan_files,
        )

        changed: list[str] = []
        deleted: list[str] = []

        for path, fingerprint in current.items():
            if self._files.get(path) != fingerprint:
                changed.append(path)

        for path in self._files:
            if path not in current:
                deleted.append(path)

        self._files = current
        self._scan_count += 1

        self.data.publish({
            "root": str(self.root),
            "file_count": len(current),
            "changed": sorted(changed),
            "deleted": sorted(deleted),
            "scan_count": self._scan_count,
        })

    def _scan_files(self) -> dict[str, tuple[int, int]]:
        result: dict[str, tuple[int, int]] = {}

        for root, dirs, files in os.walk(self.root):
            root_path = Path(root)

            dirs[:] = [
                directory
                for directory in dirs
                if directory not in self.IGNORED_DIRS
            ]

            for filename in files:
                path = root_path / filename

                try:
                    stat = path.stat()
                except OSError:
                    continue

                relative = path.relative_to(
                    self.root,
                ).as_posix()

                result[relative] = (
                    stat.st_mtime_ns,
                    stat.st_size,
                )

        return result

    def serialize_state(self) -> dict[str, Any]:
        return {
            "files": {
                path: {
                    "mtime_ns": fingerprint[0],
                    "size": fingerprint[1],
                }
                for path, fingerprint in self._files.items()
            },
            "scan_count": self._scan_count,
        }

    def restore_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise TypeError(
                "workspace private state must be an object"
            )

        files = state.get("files", {})
        scan_count = state.get("scan_count", 0)

        if not isinstance(files, dict):
            raise TypeError(
                "workspace private state 'files' must be an object"
            )

        restored: dict[str, tuple[int, int]] = {}

        for path, fingerprint in files.items():
            if not isinstance(path, str):
                raise TypeError(
                    "workspace file path must be a string"
                )

            if (
                not isinstance(fingerprint, dict)
                or not isinstance(
                    fingerprint.get("mtime_ns"),
                    int,
                )
                or not isinstance(
                    fingerprint.get("size"),
                    int,
                )
            ):
                raise TypeError(
                    f"invalid workspace fingerprint: {path}"
                )

            restored[path] = (
                fingerprint["mtime_ns"],
                fingerprint["size"],
            )

        if not isinstance(scan_count, int):
            raise TypeError(
                "workspace scan_count must be an integer"
            )

        self._files = restored
        self._scan_count = scan_count
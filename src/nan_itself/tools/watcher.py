"""
Workspace hot-reload machinery.

Generic, backend-agnostic bookkeeping shared by MCP and local
provider scanning:

    - per-file change fingerprints (mtime, size)
    - sticky error caching (an unchanged broken file is never
      retried until it changes again)

The transactional rule enforced everywhere:

    build the candidate first; only touch live state once every
    candidate succeeded.
"""

from __future__ import annotations

from pathlib import Path


def file_fingerprint(
    path: Path,
) -> tuple[int, int]:
    stat = path.stat()

    return (
        stat.st_mtime_ns,
        stat.st_size,
    )


class SourceTracker:
    """
    Change tracking for one kind of workspace source file.
    """

    def __init__(self) -> None:
        # Source file -> latest fingerprint.
        self.fingerprints: dict[
            Path,
            tuple[int, int],
        ] = {}

        # Source file -> load error for the current fingerprint.
        self.errors: dict[
            Path,
            BaseException,
        ] = {}

    def needs_load(
        self,
        path: Path,
        fingerprint: tuple[int, int],
        *,
        loaded: bool,
    ) -> bool:
        """
        Decide whether a file must be (re)loaded.

        True when the file changed, or when it has no recorded
        outcome for this fingerprint yet. An unchanged broken
        file is skipped until it changes.
        """
        previous = self.fingerprints.get(path)

        if previous != fingerprint:
            return True

        if loaded:
            return False

        return self.errors.get(path) is None

    def mark_seen(
        self,
        path: Path,
        fingerprint: tuple[int, int],
    ) -> None:
        self.fingerprints[path] = fingerprint

        self.errors.pop(path, None)

    def mark_failed(
        self,
        path: Path,
        error: BaseException,
    ) -> None:
        self.errors[path] = error

    def forget(self, path: Path) -> None:
        self.fingerprints.pop(path, None)

        self.errors.pop(path, None)

    def known_files(self) -> set[Path]:
        return set(self.fingerprints)

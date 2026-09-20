"""
Workspace hot-reload machinery.

Generic, backend-agnostic bookkeeping shared by MCP and local
provider scanning:

    - per-file change fingerprints (mtime, size)
    - failure retry with exponential backoff: a broken file is
      retried until it loads, so transient failures (network,
      missing process) self-heal while unchanged files stay quiet
      between attempts

The transactional rule enforced everywhere:

    build the candidate first; only touch live state once every
    candidate succeeded.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..utils.backoff import next_backoff


# Escalating retry schedule for repeated failures of one
# source file: the first retry waits 30s, then doubles, and
# repeated failures keep waiting 5 minutes.
BACKOFF_SCHEDULE: tuple[float, ...] = (
    30.0,
    60.0,
    120.0,
    240.0,
    300.0,
)


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

        # Source file -> (monotonic failed_at, attempts) of the
        # current fingerprint, used to schedule retries.
        self.failures: dict[
            Path,
            tuple[float, int],
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

        True when the file changed, when it has no recorded
        outcome for this fingerprint yet, or when the backoff
        window since the last failure has elapsed.
        """
        previous = self.fingerprints.get(path)

        if previous != fingerprint:
            return True

        if loaded:
            return False

        if self.errors.get(path) is None:
            return True

        return self._backoff_elapsed(path)

    def _backoff_elapsed(
        self,
        path: Path,
    ) -> bool:
        failure = self.failures.get(path)

        if failure is None:
            return True

        failed_at, attempts = failure

        delay, _ = next_backoff(
            BACKOFF_SCHEDULE,
            attempts - 1,
        )

        return (
            time.monotonic() - failed_at
        ) >= delay

    def mark_seen(
        self,
        path: Path,
        fingerprint: tuple[int, int],
    ) -> None:
        # A fingerprint change resets the retry schedule; the
        # same fingerprint keeps its attempt count so repeated
        # failures keep backing off.
        if (
            self.fingerprints.get(path)
            != fingerprint
        ):
            self.failures.pop(path, None)

        self.fingerprints[path] = fingerprint

        self.errors.pop(path, None)

    def mark_failed(
        self,
        path: Path,
        error: BaseException,
    ) -> None:
        self.errors[path] = error

        previous = self.failures.get(path)

        if previous is None:
            self.failures[path] = (
                time.monotonic(),
                1,
            )

        else:
            _, attempts = previous

            self.failures[path] = (
                time.monotonic(),
                attempts + 1,
            )

    def forget(self, path: Path) -> None:
        self.fingerprints.pop(path, None)

        self.errors.pop(path, None)

        self.failures.pop(path, None)

    def known_files(self) -> set[Path]:
        return set(self.fingerprints)

"""Job-log trimming: previous boots are unreachable, so whole dirs go."""

from __future__ import annotations

import os
from pathlib import Path

from nan.execution.jobs import _BOOT_ID, trim_job_logs


def _boot(root: Path, agent: str, boot: str, size: int, modified: float) -> Path:
    directory = root / ".nan" / "jobs" / agent / boot

    directory.mkdir(parents=True)

    log = directory / "j1.log"

    log.write_bytes(b"x" * size)

    os.utime(log, (modified, modified))
    os.utime(directory, (modified, modified))

    return directory


def test_over_budget_boots_go_oldest_first(tmp_path: Path) -> None:
    old = _boot(tmp_path, "root", "old", 100, 1_000)
    new = _boot(tmp_path, "root", "new", 100, 2_000)

    freed = trim_job_logs(tmp_path, budget=150)

    assert freed == 100
    assert not old.exists()
    assert new.exists()


def test_nothing_is_deleted_under_budget(tmp_path: Path) -> None:
    keep = _boot(tmp_path, "root", "keep", 100, 1_000)

    assert trim_job_logs(tmp_path, budget=1_000) == 0
    assert keep.exists()


def test_the_current_boot_is_never_deleted(tmp_path: Path) -> None:
    current = _boot(tmp_path, "root", _BOOT_ID, 100, 1_000)
    old = _boot(tmp_path, "root", "old", 100, 500)

    trim_job_logs(tmp_path, budget=1)

    assert current.exists()
    assert not old.exists()


def test_a_missing_jobs_directory_is_fine(tmp_path: Path) -> None:
    assert trim_job_logs(tmp_path) == 0

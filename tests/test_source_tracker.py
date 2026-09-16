from __future__ import annotations

import time
from pathlib import Path

from nan_itself.tools.watcher import (
    BACKOFF_SCHEDULE,
    SourceTracker,
)


# ============================================================================
# Helpers
# ============================================================================


FP1 = (100, 10)

FP2 = (200, 20)


def _file(tmp_path: Path) -> Path:
    return tmp_path / "source.yaml"


# ============================================================================
# Failure retry with exponential backoff
# ============================================================================


def test_first_load_is_always_needed(tmp_path):
    tracker = SourceTracker()

    assert tracker.needs_load(
        _file(tmp_path),
        FP1,
        loaded=False,
    )


def test_failure_suppresses_retry_within_backoff_window(tmp_path):
    tracker = SourceTracker()

    path = _file(tmp_path)

    tracker.mark_seen(path, FP1)

    tracker.mark_failed(path, ValueError("boom"))

    assert not tracker.needs_load(
        path,
        FP1,
        loaded=False,
    )


def test_failure_is_retried_after_backoff_window(tmp_path):
    tracker = SourceTracker()

    path = _file(tmp_path)

    tracker.mark_seen(path, FP1)

    tracker.mark_failed(path, ValueError("boom"))

    # Simulate the backoff window elapsing.
    failed_at, attempts = tracker.failures[path]

    tracker.failures[path] = (
        failed_at - BACKOFF_SCHEDULE[0],
        attempts,
    )

    assert tracker.needs_load(
        path,
        FP1,
        loaded=False,
    )


def test_repeated_failures_back_off_exponentially(tmp_path):
    tracker = SourceTracker()

    path = _file(tmp_path)

    tracker.mark_seen(path, FP1)

    tracker.mark_failed(path, ValueError("boom"))

    # Same fingerprint: the attempt count survives mark_seen,
    # so the second window is twice as long.
    tracker.mark_seen(path, FP1)

    tracker.mark_failed(path, ValueError("boom again"))

    failed_at, attempts = tracker.failures[path]

    assert attempts == 2

    # 45s ago: inside the second window (60s) -> no retry.
    tracker.failures[path] = (
        time.monotonic() - 45.0,
        attempts,
    )

    assert not tracker.needs_load(
        path,
        FP1,
        loaded=False,
    )

    # 61s ago: outside the second window -> retry.
    tracker.failures[path] = (
        time.monotonic() - 61.0,
        attempts,
    )

    assert tracker.needs_load(
        path,
        FP1,
        loaded=False,
    )


def test_backoff_is_capped(tmp_path):
    tracker = SourceTracker()

    path = _file(tmp_path)

    tracker.mark_seen(path, FP1)

    many = len(BACKOFF_SCHEDULE) + 5

    for _ in range(many):
        tracker.mark_seen(path, FP1)

        tracker.mark_failed(path, ValueError("boom"))

    _, attempts = tracker.failures[path]

    assert attempts == many

    # Even after many failures the wait never exceeds the cap:
    # elapsed >= cap -> retry.
    failed_at, _ = tracker.failures[path]

    tracker.failures[path] = (
        failed_at - BACKOFF_SCHEDULE[-1],
        attempts,
    )

    assert tracker.needs_load(
        path,
        FP1,
        loaded=False,
    )


def test_fingerprint_change_resets_backoff(tmp_path):
    tracker = SourceTracker()

    path = _file(tmp_path)

    tracker.mark_seen(path, FP1)

    tracker.mark_failed(path, ValueError("boom"))

    tracker.mark_seen(path, FP2)

    assert path not in tracker.failures

    assert tracker.needs_load(
        path,
        FP2,
        loaded=False,
    )


def test_loaded_file_never_needs_reload(tmp_path):
    tracker = SourceTracker()

    path = _file(tmp_path)

    tracker.mark_seen(path, FP1)

    tracker.mark_failed(path, ValueError("boom"))

    assert not tracker.needs_load(
        path,
        FP1,
        loaded=True,
    )


def test_forget_clears_failure_state(tmp_path):
    tracker = SourceTracker()

    path = _file(tmp_path)

    tracker.mark_seen(path, FP1)

    tracker.mark_failed(path, ValueError("boom"))

    tracker.forget(path)

    assert not tracker.fingerprints

    assert not tracker.errors

    assert not tracker.failures

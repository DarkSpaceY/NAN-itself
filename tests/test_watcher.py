"""
SourceTracker and file_fingerprint unit tests.

These lock the hot-reload decision table:

    changed fingerprint            -> load
    same fingerprint + loaded      -> skip
    same fingerprint + clean       -> load (first sight)
    same fingerprint + broken      -> skip until file changes
"""

from __future__ import annotations

import time
from pathlib import Path

from src.nan_itself.tools.watcher import (
    SourceTracker,
    file_fingerprint,
)


def make_file(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name

    path.write_text(text, encoding="utf-8")

    return path


def test_file_fingerprint_reflects_mtime_and_size(tmp_path):
    path = make_file(tmp_path, "a.yaml", "hello")

    first = file_fingerprint(path)

    assert isinstance(first, tuple)
    assert len(first) == 2

    # Size change alone must move the fingerprint.
    path.write_text("hello!", encoding="utf-8")

    second = file_fingerprint(path)

    assert second != first
    assert second[1] == len("hello!")


def test_needs_load_true_for_unknown_file(tmp_path):
    tracker = SourceTracker()

    path = make_file(tmp_path, "a.yaml", "x")

    assert tracker.needs_load(
        path,
        file_fingerprint(path),
        loaded=False,
    )


def test_needs_load_false_when_same_and_loaded(tmp_path):
    tracker = SourceTracker()

    path = make_file(tmp_path, "a.yaml", "x")
    fp = file_fingerprint(path)

    tracker.mark_seen(path, fp)

    assert not tracker.needs_load(
        path,
        fp,
        loaded=True,
    )


def test_needs_load_true_when_same_but_never_loaded(tmp_path):
    """
    A seen-but-never-loaded file (no cached error) is retried:
    this is the first-successful-load window.
    """
    tracker = SourceTracker()

    path = make_file(tmp_path, "a.yaml", "x")
    fp = file_fingerprint(path)

    tracker.mark_seen(path, fp)

    assert tracker.needs_load(
        path,
        fp,
        loaded=False,
    )


def test_broken_file_is_skipped_until_it_changes(tmp_path):
    tracker = SourceTracker()

    path = make_file(tmp_path, "broken.yaml", "bad:")
    fp = file_fingerprint(path)

    tracker.mark_seen(path, fp)
    tracker.mark_failed(path, ValueError("nope"))

    # Unchanged + broken: never retried.
    assert not tracker.needs_load(
        path,
        fp,
        loaded=False,
    )

    # File changes: retried again.
    time.sleep(0.01)
    path.write_text("good:", encoding="utf-8")

    new_fp = file_fingerprint(path)

    assert tracker.needs_load(
        path,
        new_fp,
        loaded=False,
    )


def test_mark_seen_clears_previous_error(tmp_path):
    tracker = SourceTracker()

    path = make_file(tmp_path, "a.yaml", "x")
    fp = file_fingerprint(path)

    tracker.mark_failed(path, RuntimeError("old"))

    tracker.mark_seen(path, fp)

    assert tracker.errors.get(path) is None


def test_mark_failed_keeps_fingerprint(tmp_path):
    tracker = SourceTracker()

    path = make_file(tmp_path, "a.yaml", "x")
    fp = file_fingerprint(path)

    tracker.mark_seen(path, fp)
    tracker.mark_failed(path, RuntimeError("x"))

    assert tracker.fingerprints[path] == fp
    assert isinstance(tracker.errors[path], RuntimeError)


def test_forget_removes_all_traces(tmp_path):
    tracker = SourceTracker()

    path = make_file(tmp_path, "a.yaml", "x")
    fp = file_fingerprint(path)

    tracker.mark_seen(path, fp)
    tracker.mark_failed(path, RuntimeError("x"))

    tracker.forget(path)

    assert path not in tracker.fingerprints
    assert path not in tracker.errors
    assert path not in tracker.known_files()


def test_known_files_lists_everything_seen(tmp_path):
    tracker = SourceTracker()

    a = make_file(tmp_path, "a.yaml", "1")
    b = make_file(tmp_path, "b.yaml", "2")

    tracker.mark_seen(a, file_fingerprint(a))
    tracker.mark_seen(b, file_fingerprint(b))

    assert tracker.known_files() == {a, b}

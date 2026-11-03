"""Pytest fixtures for the nan test suite."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _restore_paths() -> Iterator[None]:
    """
    Snapshot the fixed layout and restore it after each test.

    `Settings` instances are per-test values, but the layout they
    expose (`paths`) is the installation's one `PATHS` object, so a
    test that relocates it must not leak into the next one.
    """
    from nan.config.models import PATHS

    paths = PATHS.model_copy(deep=True)

    yield

    for name in type(paths).model_fields:
        setattr(PATHS, name, getattr(paths, name))


@pytest.fixture
def no_sink() -> Any:
    """Ensure no event bus is attached (the default state)."""
    from nan.events import sink

    sink.reset()

    yield sink

    sink.reset()

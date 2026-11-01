"""Pytest fixtures for the nan test suite."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _restore_settings() -> Iterator[None]:
    """
    Snapshot the process-wide settings and restore them after each
    test: components read the live object, so a test that retunes
    it must not leak into the next one.
    """
    from nan.config import settings
    from nan.config.models import PATHS, Paths

    snapshot = {
        name: getattr(settings, name).model_copy(deep=True) for name in type(settings).model_fields
    }

    paths = PATHS.model_copy(deep=True)

    yield

    for name, value in snapshot.items():
        setattr(settings, name, value)

    for name in Paths.model_fields:
        setattr(PATHS, name, getattr(paths, name))


@pytest.fixture
def no_sink() -> Any:
    """Ensure no event bus is attached (the default state)."""
    from nan.events import sink

    sink.reset()

    yield sink

    sink.reset()

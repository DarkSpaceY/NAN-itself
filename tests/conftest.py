"""Pytest fixtures for the nan test suite."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def no_sink() -> Any:
    """Ensure no event bus is attached (the default state)."""
    from nan.events import sink

    sink.reset()

    yield sink

    sink.reset()

"""Failure classification: what may be retried, what must die."""

from __future__ import annotations

import asyncio

import pytest

from nan.agent.failure import FailureKind, Unrecoverable, classify


class _Status(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")

        self.status_code = status_code


def _named(name: str) -> Exception:
    return type(name, (Exception,), {})("boom")


def test_cancelled_is_its_own_kind() -> None:
    assert classify(asyncio.CancelledError()) is FailureKind.CANCELLED


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_retryable_status(status: int) -> None:
    assert classify(_Status(status)) is FailureKind.RETRYABLE


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_status_is_fatal(status: int) -> None:
    assert classify(_Status(status)) is FailureKind.FATAL


@pytest.mark.parametrize(
    "name",
    [
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "ReadTimeout",
        "RemoteProtocolError",
    ],
)
def test_transport_names_are_retryable(name: str) -> None:
    assert classify(_named(name)) is FailureKind.RETRYABLE


def test_builtin_timeouts_and_connections_are_retryable() -> None:
    assert classify(TimeoutError()) is FailureKind.RETRYABLE
    assert classify(ConnectionError("reset")) is FailureKind.RETRYABLE


def test_wrapped_cause_is_followed() -> None:
    wrapped = ValueError("stream ended")

    wrapped.__cause__ = TimeoutError()

    assert classify(wrapped) is FailureKind.RETRYABLE


@pytest.mark.parametrize("exc", [ValueError("bad"), KeyError("missing"), AssertionError("broken")])
def test_unknown_exceptions_are_fatal(exc: Exception) -> None:
    assert classify(exc) is FailureKind.FATAL


def test_unrecoverable_is_fatal() -> None:
    exc = Unrecoverable("retries exhausted")

    assert classify(exc) is FailureKind.FATAL
    assert exc.reason == "retries exhausted"
    assert str(exc) == "retries exhausted"

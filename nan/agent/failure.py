"""
Failure vocabulary: one classification for everything a turn throws.

`classify` answers one question -- may this step run again? -- with
three answers:

    cancelled  the loop's control path owns it (pause, kill, shutdown)
    retryable  the same step may run again later (429, network,
               timeout, a broken stream, 5xx)
    fatal      no retry can repair it

Only the model call is retried: a verb is at-most-once, and a verb
that throws returns its error as the model-visible result instead of
failing the turn (see Agent._execute_call). A retry ladder that runs out
is fatal by definition (`Unrecoverable`).

`classify` is the one place exception types are interpreted. It reads
status codes and transport exception names structurally, so the SDKs
may wrap a cause without being imported here.
"""

from __future__ import annotations

import asyncio
from enum import StrEnum


class FailureKind(StrEnum):
    """What may be done about a failure."""

    RETRYABLE = "retryable"
    FATAL = "fatal"
    CANCELLED = "cancelled"


class Unrecoverable(Exception):
    """
    Retrying is pointless: raised when a retry ladder is exhausted,
    carrying the reason the turn failed for.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)

        self.reason = reason


# Transport/SDK exceptions that mean "try the same turn again".
# Names are matched structurally so the vendor packages stay out
# of this module.
_RETRYABLE_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ConnectError",
        "ConnectTimeout",
        "ReadError",
        "ReadTimeout",
        "WriteError",
        "WriteTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
        "ProtocolError",
        "TransportError",
        "TimeoutException",
    }
)

# How far a wrapper chain is unwrapped before giving up.
_MAX_CAUSE_DEPTH = 4


def classify(exc: BaseException) -> FailureKind:
    """The kind of one thrown failure."""
    if isinstance(exc, asyncio.CancelledError):
        return FailureKind.CANCELLED

    if _retryable(exc):
        return FailureKind.RETRYABLE

    return FailureKind.FATAL


def _retryable(exc: BaseException, depth: int = 0) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    status = getattr(exc, "status_code", None)

    if isinstance(status, int):
        if status == 429 or status >= 500:
            return True

        if status >= 400:
            return False

    if type(exc).__name__ in _RETRYABLE_NAMES:
        return True

    if depth >= _MAX_CAUSE_DEPTH:
        return False

    cause = exc.__cause__ or exc.__context__

    return cause is not None and _retryable(cause, depth + 1)

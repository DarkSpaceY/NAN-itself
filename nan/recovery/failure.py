"""
Failure vocabulary: one classification for everything a run can
throw.

A failure has a domain -- where it happened -- and, for a turn, a
kind -- what may be done about it:

    process  startup: a missing persona/sandbox/hub/config refuses
             to start; there is no runtime to recover
    agent    an agent-scope fault (auth, corrupt history, a child
             that cannot start): the agent dies
    turn     one turn: retryable, model-correctable, fatal, or a
             cancellation

Retryable means the same turn may run again later (429, network,
timeout, a broken stream, 5xx). Model-correctable means the model
sees the error as the verb result and rewrites its call; it is not
an exception. Fatal means no retry can repair it; the retry
ladder running out is fatal by definition (`Unrecoverable`).

`classify` is the one place exception types are interpreted. It
reads status codes and transport exception names structurally,
so the SDKs may wrap a cause without being imported here.
"""

from __future__ import annotations

import asyncio
from enum import StrEnum


class FailureDomain(StrEnum):
    """Where a failure happened."""

    PROCESS = "process"
    AGENT = "agent"
    TURN = "turn"


class FailureKind(StrEnum):
    """What may be done about a failure."""

    RETRYABLE = "retryable"
    MODEL_CORRECTABLE = "model_correctable"
    FATAL = "fatal"
    CANCELLED = "cancelled"


class CancelReason(StrEnum):
    """Why a turn was cancelled: the three non-failure exits."""

    SHUTDOWN = "shutdown"
    PAUSE = "pause"
    KILL = "kill"


class Unrecoverable(Exception):
    """
    Retrying is pointless: the agent dies. Raised when a retry
    ladder is exhausted, or when a turn proved permanently
    broken.
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

    if isinstance(exc, Unrecoverable):
        return FailureKind.FATAL

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

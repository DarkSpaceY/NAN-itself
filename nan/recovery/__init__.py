"""Recovery domain: failure classification, snapshots and boot restore."""

from __future__ import annotations

from .failure import FailureKind, Unrecoverable, classify
from .restore import restore
from .snapshot import AgentSnapshot, InFlight, SnapshotStore

__all__ = [
    "AgentSnapshot",
    "FailureKind",
    "InFlight",
    "SnapshotStore",
    "Unrecoverable",
    "classify",
    "restore",
]

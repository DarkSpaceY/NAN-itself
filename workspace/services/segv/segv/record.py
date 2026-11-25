"""One archived turn, described without any protocol vocabulary.

The core owns these fields: identity, span, the message run chunks are
built from, a ledger headline, per-turn counters, and the lineage/epoch
bookkeeping the store and cache key on. `raw` is whatever the adapter
archived next to them; the core stores and returns it without reading it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TurnRecord:
    agent_id: str
    started_at: float
    ended_at: float | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    stats: dict[str, int] = field(default_factory=dict)
    epoch: int = 0
    depth: int = 0
    parent_id: str | None = None
    origin: str = "normal"
    death: str | None = None
    llm_started_at: float | None = None
    llm_ended_at: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_json(self, *, seq: int, turn_id: str) -> dict[str, Any]:
        """The stored form: what readers (fact, ledger, panel) see."""
        return {
            "agent_id": self.agent_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "seq": seq,
            "turn_id": turn_id,
            "messages": self.messages,
            "summary": self.summary,
            "stats": self.stats,
            "meta": {
                "epoch": self.epoch,
                "depth": self.depth,
                "parent_id": self.parent_id,
                "origin": self.origin,
                "death": self.death,
            },
            "raw": self.raw,
        }

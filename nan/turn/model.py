"""
Turn: the unit of agent life, and the only runtime record.

A turn is ONE observation -> model call -> verb run cycle. The
contract is LIST self-containment: a list of Turns is enough to
reconstruct the whole conversation (each completed Turn freezes
its own rendered messages), so no Turn carries history and no
persistent history exists anywhere.

    identity   agent_id / parent_id / depth
    inputs     inbox (up messages: user input, child sends,
               finish reports) / injections (per-turn service
               contributions, errors included)
    flow       reply / calls / results, written back as the
               turn completes
    frozen     rendered -- the turn's message run, computed once
               and never recomputed (byte-stable history for
               prefix caching)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from nan.llm import Message, ToolCall, Usage

InboxKind = Literal["user", "parentagent", "subagent", "unknown"]


@dataclass(frozen=True)
class LLMCall:
    """
    The metering record of the one model call a turn makes:
    identity, token usage, outcome, wall time, and how many
    turn-level backoff retries preceded the successful call.
    """

    model: str
    usage: Usage | None = None
    finish_reason: str | None = None
    latency_s: float = 0.0
    retries: int = 0


@dataclass(frozen=True)
class InboxMessage:
    """
    One message received at a turn boundary. `kind` names the
    SENDER'S ROLE, resolved at delivery time:

    "user"         root only; text from the gateway
    "parentagent"  my parent sent down (includes the spawn task)
    "subagent"     one of my subagents sent up; `event` is "send"
                   or "finish" (text IS the finish report)

    Death is not an inbox message: agent lifecycle changes ride
    the <delta> section (agent_delta) instead.
    """

    kind: InboxKind
    source: str
    text: str
    event: str = "send"
    received_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Injection:
    """
    One injection service's contribution to this turn's prompt.

    Failures stay on the record (ok=False, text=reason): the
    Turn is a complete runtime log, including what went wrong.
    """

    service_id: str
    text: str
    ok: bool = True
    duration_s: float = 0.0


@dataclass(frozen=True)
class DeltaEvent:
    """
    One system-state line, grouped by section at render time:

        <delta>
        <service>+echo online</service>
        <subagent>-9d21 died: killed</subagent>
        </delta>

    `section` is an open string: a new section kind arrives as a
    new producer with no renderer change. Render order is
    first-appearance in the turn's event list (deterministic given
    the list; per-turn content anyway, so no canonical order is
    needed for prefix stability).
    """

    section: str
    line: str


@dataclass(frozen=True)
class Turn:
    agent_id: str
    parent_id: str | None
    depth: int

    inbox: tuple[InboxMessage, ...] = ()
    injections: tuple[Injection, ...] = ()
    delta: tuple[DeltaEvent, ...] = ()  # grouped into <delta><section> at render

    reply: str | None = None
    calls: tuple[ToolCall, ...] = ()
    results: tuple[str, ...] = ()

    llm: LLMCall | None = None

    error: str | None = None

    rendered: tuple[Message, ...] | None = None

    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None

    def to_json(self) -> dict[str, Any]:
        """
        The full Turn as JSON -- the injection payload and the
        reconstruction record. `rendered` is excluded: it is
        derivable and large.
        """
        return {
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "inbox": [
                {
                    "kind": m.kind,
                    "source": m.source,
                    "text": m.text,
                    "event": m.event,
                    "received_at": m.received_at,
                }
                for m in self.inbox
            ],
            "injections": [
                {
                    "service_id": i.service_id,
                    "text": i.text,
                    "ok": i.ok,
                    "duration_s": i.duration_s,
                }
                for i in self.injections
            ],
            "delta": [{"section": e.section, "line": e.line} for e in self.delta],
            "reply": self.reply,
            "calls": [c.model_dump() for c in self.calls],
            "results": list(self.results),
            "llm": (
                {
                    "model": self.llm.model,
                    "usage": (self.llm.usage.model_dump() if self.llm.usage is not None else None),
                    "finish_reason": self.llm.finish_reason,
                    "latency_s": self.llm.latency_s,
                    "retries": self.llm.retries,
                }
                if self.llm is not None
                else None
            ),
            "error": self.error,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }

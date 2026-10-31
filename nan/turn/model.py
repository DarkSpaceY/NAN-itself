"""
Turn: the unit of agent life, and the only runtime record.

A turn is ONE observation -> model call -> verb run cycle. The
contract is LIST self-containment: a list of Turns is enough to
reconstruct the whole conversation (each completed Turn freezes
its own rendered messages); history lives in the Turn list.

    identity   agent_id / parent_id / depth
    inputs     inbox (up messages: user input, child sends,
               finish reports) / injections (per-turn service
               contributions, errors included)
    flow       reply / calls / results, written back as the
               turn completes; `death` when this turn ended the agent
    frozen     rendered -- the turn's message run, frozen after the
               first render (byte-stable for prefix caching)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from nan.llm import Message, ThinkingBlock, ToolCall, Usage

InboxKind = Literal["user", "parentagent", "subagent", "unknown"]

# Why a turn began: the first of an agent's life, an ordinary
# turn, or the turn right after the context was cleared.
TurnOrigin = Literal["first", "normal", "context_cleared"]


@dataclass(frozen=True)
class Death:
    """
    The one exit record: why the agent stopped, and the finish
    report when the model ended it itself (None for external
    deaths). A turn that ends an agent carries it; the first
    record wins.
    """

    reason: str
    report: str | None = None


@dataclass(frozen=True)
class LLMCall:
    """
    The metering record of the one model call a turn makes:
    identity, wall interval, token usage, outcome, and how many
    turn-level backoff retries preceded the successful call.
    """

    model: str
    started_at: float
    ended_at: float
    usage: Usage | None = None
    finish_reason: str | None = None
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

    Agent lifecycle changes, including death, ride the <delta>
    section (agent_delta).
    """

    kind: InboxKind
    source: str
    text: str
    event: str = "send"
    received_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Injection:
    """
    One injection service's contribution to this turn's prompt,
    with the request interval it took to produce.

    ok=False with text=reason records a failed request.
    """

    service_id: str
    text: str
    started_at: float
    ended_at: float
    ok: bool = True


@dataclass(frozen=True)
class ToolResult:
    """
    One executed tool call, positional with Turn.calls: `output`
    is the text the model saw, the interval is the verb's wall
    time.
    """

    call_id: str
    name: str
    output: str
    started_at: float
    ended_at: float


@dataclass(frozen=True)
class DeltaEvent:
    """
    One system-state line, grouped by section at render time:

        <delta>
        <service>+echo online</service>
        <subagent>-9d21 died: killed by holder r0</subagent>
        </delta>

    `section` is an open string. Render order is first-appearance
    in the turn's event list.
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

    origin: TurnOrigin = "normal"

    # The new persona text when it changed before this turn: the
    # epoch marker for the system prompt. Null when unchanged.
    persona: str | None = None

    reply: str | None = None
    thinking: tuple[ThinkingBlock, ...] = ()
    calls: tuple[ToolCall, ...] = ()
    results: tuple[ToolResult, ...] = ()

    llm: LLMCall | None = None

    # Set when this turn ended the agent, in which case the agent
    # does not begin another one.
    death: Death | None = None

    rendered: tuple[Message, ...] | None = None

    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None

    def to_json(self) -> dict[str, Any]:
        """
        The complete Turn as JSON. A completed turn is a full
        reconstruction record; an in-flight one carries whatever
        exists so far.
        """
        return {
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "origin": self.origin,
            "persona": self.persona,
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
                    "started_at": i.started_at,
                    "ended_at": i.ended_at,
                    "ok": i.ok,
                }
                for i in self.injections
            ],
            "delta": [{"section": e.section, "line": e.line} for e in self.delta],
            "reply": self.reply,
            "thinking": [b.model_dump() for b in self.thinking],
            "calls": [c.model_dump() for c in self.calls],
            "results": [
                {
                    "call_id": r.call_id,
                    "name": r.name,
                    "output": r.output,
                    "started_at": r.started_at,
                    "ended_at": r.ended_at,
                }
                for r in self.results
            ],
            "llm": (
                {
                    "model": self.llm.model,
                    "started_at": self.llm.started_at,
                    "ended_at": self.llm.ended_at,
                    "usage": (self.llm.usage.model_dump() if self.llm.usage is not None else None),
                    "finish_reason": self.llm.finish_reason,
                    "retries": self.llm.retries,
                }
                if self.llm is not None
                else None
            ),
            "death": (
                {"reason": self.death.reason, "report": self.death.report}
                if self.death is not None
                else None
            ),
            "rendered": (
                [m.model_dump(mode="json") for m in self.rendered]
                if self.rendered is not None
                else None
            ),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }

    def to_fresh_json(self) -> dict[str, Any]:
        """
        The static arrival shape: exactly the fields known when a
        turn is delivered to injection services (before the model
        call). reply / calls / results / llm / death / ended_at /
        rendered are absent from the shape.
        """
        return {
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "origin": self.origin,
            "persona": self.persona,
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
                    "started_at": i.started_at,
                    "ended_at": i.ended_at,
                    "ok": i.ok,
                }
                for i in self.injections
            ],
            "delta": [{"section": e.section, "line": e.line} for e in self.delta],
            "started_at": self.started_at,
        }

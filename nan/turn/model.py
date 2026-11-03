"""
Turn: the unit of agent life, and the only runtime record.

A turn is ONE observation -> model call -> verb run cycle. The
contract is LIST self-containment: a list of Turns is enough to
reconstruct the whole conversation (each completed Turn freezes
its own rendered messages); history lives in the Turn list.

    identity   agent_id / parent_id / depth / epoch / origin
    time       started_at / ended_at
    inputs     persona (when it changed) / inbox (up messages: user
               input, child sends, finish reports) / injections
               (per-turn service contributions, errors included) /
               delta (system-state lines)
    flow       phase / reply / thinking / verb_calls / verb_results / llm;
               `death` when this turn ended the agent
    frozen     rendered -- the turn's message run, frozen after the
               first render (byte-stable for prefix caching)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from nan.llm.types import Message, ThinkingBlock, Usage, VerbCall

InboxKind = Literal["user", "parentagent", "subagent"]

# Why a turn began: the first of an agent's life, an ordinary
# turn, or the turn right after the context was cleared. This is
# a wire label for observers; behaviour reads `epoch`.
TurnOrigin = Literal["first", "normal", "context_cleared"]

# Where a turn stands. Monotone: a turn only moves forward, so a
# parked or restored turn resumes exactly where it stopped.
#     observed  the turn opened on its inputs
#     injected  service contributions collected
#     thought   the model answered
#     acted     the verbs reported
#     committed frozen and durable
TurnPhase = Literal["observed", "injected", "thought", "acted", "committed"]

# What the model sees for a call whose verb never ran or never
# reported back (a pause, a crash mid-act, a restored turn). Also
# recorded in turn.verb_results, so the turn stays self-contained.
INTERRUPTED = "interrupted: agent stopped during execution"


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
    One message received at a turn boundary. `kind` names the edge
    it came in through, which the sender knows and sets:

    "user"         from the gateway into the root
    "parentagent"  sent down into a child (includes the spawn task)
    "subagent"     sent up into a parent; `event` is "send" or
                   "finish" (text IS the finish report)

    Agent lifecycle changes, including death, ride the <delta>
    section (agent_delta).
    """

    kind: InboxKind
    source: str
    text: str
    event: str = "send"
    received_at: float = field(default_factory=time.time)

    @classmethod
    def from_user(cls, text: str) -> InboxMessage:
        """Gateway input, delivered to the root."""
        return cls(kind="user", source="user", text=text)

    @classmethod
    def from_parent(cls, text: str, *, source: str) -> InboxMessage:
        """A message sent down into a child."""
        return cls(kind="parentagent", source=source, text=text)

    @classmethod
    def from_child(cls, text: str, *, source: str, event: str = "send") -> InboxMessage:
        """A message sent up into a parent, or a finish report."""
        return cls(kind="subagent", source=source, text=text, event=event)

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "text": self.text,
            "event": self.event,
            "received_at": self.received_at,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> InboxMessage:
        return cls(
            kind=data["kind"],
            source=data["source"],
            text=data["text"],
            event=data.get("event", "send"),
            received_at=data.get("received_at", 0.0),
        )


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
class VerbResult:
    """
    One executed verb call, positional with Turn.verb_calls: `output`
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
    # -- identity --------------------------------------------------------
    agent_id: str
    parent_id: str | None
    depth: int

    # The context epoch this turn belongs to. Clearing the context
    # opens the next epoch; history renders within one epoch.
    epoch: int = 0

    # Why the turn began, for observers (see TurnOrigin).
    origin: TurnOrigin = "normal"

    # -- time ------------------------------------------------------------
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None

    # -- inputs ----------------------------------------------------------
    # The new persona text when it changed before this turn: the
    # epoch marker for the system prompt. Null when unchanged.
    persona: str | None = None

    inbox: tuple[InboxMessage, ...] = ()
    injections: tuple[Injection, ...] = ()

    # System-state lines, grouped into <delta><section> at render.
    delta: tuple[DeltaEvent, ...] = ()

    # -- flow ------------------------------------------------------------
    # How far the turn has run (see TurnPhase). The only stage
    # marker: no caller infers progress from the payload shape.
    phase: TurnPhase = "observed"

    reply: str | None = None
    thinking: tuple[ThinkingBlock, ...] = ()
    verb_calls: tuple[VerbCall, ...] = ()
    verb_results: tuple[VerbResult, ...] = ()

    llm: LLMCall | None = None

    # Set when this turn ended the agent, in which case the agent
    # does not begin another one.
    death: Death | None = None

    # -- frozen ----------------------------------------------------------
    rendered: tuple[Message, ...] | None = None

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
            "epoch": self.epoch,
            "origin": self.origin,
            "phase": self.phase,
            "persona": self.persona,
            "inbox": [message.to_json() for message in self.inbox],
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
            "calls": [c.model_dump() for c in self.verb_calls],
            "results": [
                {
                    "call_id": r.call_id,
                    "name": r.name,
                    "output": r.output,
                    "started_at": r.started_at,
                    "ended_at": r.ended_at,
                }
                for r in self.verb_results
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

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Turn:
        """
        Rebuild the complete Turn from `to_json` output. Nested
        LLM DTOs are pydantic; the frozen records are rebuilt
        field by field.
        """
        llm = data.get("llm")
        death = data.get("death")
        rendered = data.get("rendered")

        return cls(
            agent_id=data["agent_id"],
            parent_id=data.get("parent_id"),
            depth=data["depth"],
            inbox=tuple(InboxMessage.from_json(message) for message in data.get("inbox", ())),
            injections=tuple(
                Injection(
                    service_id=injection["service_id"],
                    text=injection["text"],
                    started_at=injection["started_at"],
                    ended_at=injection["ended_at"],
                    ok=injection.get("ok", True),
                )
                for injection in data.get("injections", ())
            ),
            delta=tuple(
                DeltaEvent(section=event["section"], line=event["line"])
                for event in data.get("delta", ())
            ),
            origin=data.get("origin", "normal"),
            epoch=data.get("epoch", 0),
            phase=data.get("phase", "observed"),
            persona=data.get("persona"),
            reply=data.get("reply"),
            thinking=tuple(
                ThinkingBlock.model_validate(block) for block in data.get("thinking", ())
            ),
            verb_calls=tuple(VerbCall.model_validate(call) for call in data.get("calls", ())),
            verb_results=tuple(
                VerbResult(
                    call_id=result["call_id"],
                    name=result["name"],
                    output=result["output"],
                    started_at=result["started_at"],
                    ended_at=result["ended_at"],
                )
                for result in data.get("results", ())
            ),
            llm=(
                LLMCall(
                    model=llm["model"],
                    started_at=llm["started_at"],
                    ended_at=llm["ended_at"],
                    usage=(
                        Usage.model_validate(llm["usage"]) if llm.get("usage") is not None else None
                    ),
                    finish_reason=llm.get("finish_reason"),
                    retries=llm.get("retries", 0),
                )
                if llm is not None
                else None
            ),
            death=(
                Death(reason=death["reason"], report=death.get("report"))
                if death is not None
                else None
            ),
            rendered=(
                tuple(Message.model_validate(message) for message in rendered)
                if rendered is not None
                else None
            ),
            started_at=data.get("started_at", 0.0),
            ended_at=data.get("ended_at"),
        )

    def noted(self, line: str, *, section: str = "agent") -> Turn:
        """
        This turn plus one delta line: the one way a runtime fact
        (paused, resumed, a failure) is attached to the turn it
        happened during.
        """
        return replace(
            self,
            delta=tuple([*self.delta, DeltaEvent(section=section, line=line)]),
        )

    def close_interrupted(self, *, ended_at: float | None = None) -> Turn:
        """
        End a turn with verbs still missing results: every call
        without a result gets an interrupted placeholder, then the
        turn closes. Rendering pairs by call id, so the fill makes
        the model see exactly what happened.
        """
        finished = {result.call_id for result in self.verb_results}

        now = time.time() if ended_at is None else ended_at

        return replace(
            self,
            verb_results=tuple(
                [
                    *self.verb_results,
                    *(
                        VerbResult(
                            call_id=call.id,
                            name=call.name,
                            output=INTERRUPTED,
                            started_at=now,
                            ended_at=now,
                        )
                        for call in self.verb_calls
                        if call.id not in finished
                    ),
                ]
            ),
            phase="committed",
            ended_at=now,
        )

    def to_fresh_json(self) -> dict[str, Any]:
        """
        The static arrival shape: exactly the fields known when a
        turn is delivered to injection services (before the model
        call). reply / verb_calls / verb_results / llm / death / ended_at /
        rendered are absent from the shape.
        """
        return {
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "epoch": self.epoch,
            "origin": self.origin,
            "persona": self.persona,
            "inbox": [message.to_json() for message in self.inbox],
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

"""The Turn record: JSON round-trip, the fresh arrival shape,
interrupted closing and delta noting."""

from __future__ import annotations

from nan.llm.types import Message, ThinkingBlock, Usage, VerbCall
from nan.turn import (
    INTERRUPTED,
    Death,
    DeltaEvent,
    InboxMessage,
    Injection,
    LLMCall,
    Turn,
    VerbResult,
)


def _full_turn(*, death: Death | None = None) -> Turn:
    thinking = ThinkingBlock(text="reasoning", signature="sig", format="anthropic")
    call = VerbCall(id="c1", name="shell", arguments={"command": "ls"})

    return Turn(
        agent_id="a1",
        parent_id=None,
        depth=0,
        inbox=(InboxMessage(kind="user", source="user", text="hello", received_at=1.0),),
        injections=(
            Injection(service_id="svc", text="context", started_at=2.0, ended_at=2.5, ok=False),
        ),
        delta=(DeltaEvent(section="subagent", line="+a2 spawned"),),
        epoch=1,
        origin="context_cleared",
        phase="committed",
        persona="new persona",
        reply="answer",
        thinking=(thinking,),
        verb_calls=(call,),
        verb_results=(
            VerbResult(call_id="c1", name="shell", output="ok", started_at=3.0, ended_at=3.1),
        ),
        llm=LLMCall(
            model="m",
            started_at=3.0,
            ended_at=4.0,
            usage=Usage(prompt_tokens=5, completion_tokens=7),
            finish_reason="tool_calls",
            retries=2,
        ),
        death=death,
        rendered=(
            Message(role="user", content="hello"),
            Message(role="assistant", content="answer", thinking=[thinking], tool_calls=[call]),
            Message(role="tool", tool_call_id="c1", content="ok"),
        ),
        started_at=1.0,
        ended_at=4.0,
    )


def test_turn_round_trips_through_json() -> None:
    turn = _full_turn(death=Death(reason="finished", report="done"))

    assert Turn.from_json(turn.to_json()) == turn


def test_turn_from_minimal_json_uses_defaults() -> None:
    turn = Turn.from_json({"agent_id": "a1", "parent_id": None, "depth": 0})

    assert turn.agent_id == "a1"
    assert turn.origin == "normal"
    assert turn.epoch == 0
    assert turn.phase == "observed"
    assert turn.inbox == ()
    assert turn.reply is None
    assert turn.rendered is None
    assert turn.started_at == 0.0


def test_fresh_shape_carries_only_arrival_fields() -> None:
    """The inject channel sees the turn before the model call: no
    reply/calls/results/llm/death/ended_at/rendered."""
    turn = _full_turn(death=Death(reason="finished", report="done"))

    fresh = turn.to_fresh_json()

    assert fresh["agent_id"] == "a1"
    assert fresh["inbox"] == [message.to_json() for message in turn.inbox]
    assert fresh["delta"] == [{"section": "subagent", "line": "+a2 spawned"}]
    assert fresh["started_at"] == 1.0

    for absent in ("reply", "calls", "results", "llm", "death", "ended_at", "rendered"):
        assert absent not in fresh


def test_close_interrupted_fills_only_the_missing_results() -> None:
    call_a = VerbCall(id="a", name="shell", arguments={})
    call_b = VerbCall(id="b", name="shell", arguments={})

    reported = VerbResult(call_id="a", name="shell", output="done", started_at=1.0, ended_at=2.0)

    turn = Turn(
        agent_id="a1",
        parent_id=None,
        depth=0,
        phase="thought",
        verb_calls=(call_a, call_b),
        verb_results=(reported,),
    )

    closed = turn.close_interrupted(ended_at=9.0)

    assert closed.phase == "committed"
    assert closed.ended_at == 9.0
    assert [result.call_id for result in closed.verb_results] == ["a", "b"]
    assert closed.verb_results[0] == reported
    assert closed.verb_results[1].output == INTERRUPTED
    assert closed.verb_results[1].started_at == 9.0


def test_noted_appends_one_delta_line() -> None:
    turn = Turn(agent_id="a1", parent_id=None, depth=0)

    noted = turn.noted("paused")

    assert noted.delta == (DeltaEvent(section="agent", line="paused"),)


def test_inbox_from_json_defaults_the_optional_fields() -> None:
    message = InboxMessage.from_json({"kind": "user", "source": "user", "text": "hi"})

    assert message.event == "send"
    assert message.received_at == 0.0


def test_inbox_factories_name_the_edge() -> None:
    assert InboxMessage.from_user("hi").kind == "user"

    down = InboxMessage.from_parent("hi", source="p")
    assert (down.kind, down.source) == ("parentagent", "p")

    up = InboxMessage.from_child("report", source="c", event="finish")
    assert (up.kind, up.source, up.event) == ("subagent", "c", "finish")

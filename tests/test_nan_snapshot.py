"""Snapshots: a Turn list round-trips, and the store stays atomic."""

from __future__ import annotations

from pathlib import Path

import pytest

from nan.llm import Message, ThinkingBlock, ToolCall, Usage
from nan.recovery import AgentSnapshot, InFlight, SnapshotStore
from nan.turn import (
    Death,
    DeltaEvent,
    InboxMessage,
    Injection,
    LLMCall,
    ToolResult,
    Turn,
)


def _turn(
    *,
    agent_id: str = "a1",
    parent_id: str | None = None,
    ended: bool = True,
    death: Death | None = None,
) -> Turn:
    thinking = ThinkingBlock(text="reasoning", signature="sig", format="anthropic")
    call = ToolCall(id="c1", name="shell", arguments={"command": "ls"})

    return Turn(
        agent_id=agent_id,
        parent_id=parent_id,
        depth=0 if parent_id is None else 1,
        inbox=(InboxMessage(kind="user", source="user", text="hello", received_at=1.0),),
        injections=(
            Injection(service_id="svc", text="context", started_at=2.0, ended_at=2.5, ok=False),
        ),
        delta=(DeltaEvent(section="subagent", line="+a2 spawned"),),
        origin="context_cleared",
        persona="new persona",
        reply="answer",
        thinking=(thinking,),
        calls=(call,),
        results=(
            ToolResult(call_id="c1", name="shell", output="ok", started_at=3.0, ended_at=3.1),
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
        ended_at=4.0 if ended else None,
    )


def test_turn_round_trips_through_json() -> None:
    turn = _turn(death=Death(reason="finished", report="done"))

    assert Turn.from_json(turn.to_json()) == turn


def test_turn_from_minimal_json_uses_defaults() -> None:
    turn = Turn.from_json({"agent_id": "a1", "parent_id": None, "depth": 0})

    assert turn.agent_id == "a1"
    assert turn.origin == "normal"
    assert turn.inbox == ()
    assert turn.reply is None
    assert turn.rendered is None


def test_snapshot_derives_identity_from_its_turns() -> None:
    snapshot = AgentSnapshot(turns=(_turn(agent_id="a2", parent_id="a1"),))

    assert snapshot.agent_id == "a2"
    assert snapshot.parent_id == "a1"
    assert snapshot.depth == 1
    assert snapshot.death is None


def test_snapshot_rejects_empty_or_mixed_turns() -> None:
    with pytest.raises(ValueError):
        AgentSnapshot(turns=())

    with pytest.raises(ValueError):
        AgentSnapshot(turns=(_turn(agent_id="a1"), _turn(agent_id="a2")))


def test_snapshot_round_trips() -> None:
    snapshot = AgentSnapshot(
        turns=(_turn(), _turn(agent_id="a1", death=Death(reason="killed"))),
    )

    assert AgentSnapshot.from_json(snapshot.to_json()) == snapshot


@pytest.mark.asyncio
async def test_store_saves_and_loads(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)

    snapshot = AgentSnapshot(turns=(_turn(),))

    await store.save(snapshot)

    assert store.path("a1") == tmp_path / "a1.json"

    loaded = store.load_all()

    assert list(loaded) == ["a1"]
    assert loaded["a1"] == snapshot


@pytest.mark.asyncio
async def test_save_overwrites_whole_state(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)

    await store.save(AgentSnapshot(turns=(_turn(),)))

    await store.save(
        AgentSnapshot(turns=(_turn(agent_id="a1", death=Death(reason="killed")),)),
    )

    loaded = store.load_all()["a1"]

    assert loaded.turns[0].death == Death(reason="killed")
    assert len(loaded.turns) == 1


@pytest.mark.asyncio
async def test_save_leaves_no_temporary_files(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)

    await store.save(AgentSnapshot(turns=(_turn(),)))

    assert sorted(path.name for path in tmp_path.iterdir()) == ["a1.json"]


@pytest.mark.asyncio
async def test_remove_deletes_the_snapshot(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)

    await store.save(AgentSnapshot(turns=(_turn(),)))

    store.remove("a1")

    assert store.load_all() == {}

    store.remove("a1")


def test_load_all_without_a_directory_is_empty(tmp_path: Path) -> None:
    assert SnapshotStore(tmp_path / "missing").load_all() == {}


@pytest.mark.asyncio
async def test_corrupt_snapshot_is_skipped(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)

    await store.save(AgentSnapshot(turns=(_turn(),)))

    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")

    assert list(store.load_all()) == ["a1"]


@pytest.mark.asyncio
async def test_id_mismatch_is_skipped(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)

    await store.save(AgentSnapshot(turns=(_turn(),)))

    (tmp_path / "a1.json").rename(tmp_path / "other.json")

    assert store.load_all() == {}


def test_no_in_flight_when_the_last_turn_ended() -> None:
    assert InFlight.read(AgentSnapshot(turns=(_turn(),))) is None


def test_in_flight_without_an_answer_is_requeued() -> None:
    open_turn = Turn(
        agent_id="a1",
        parent_id=None,
        depth=0,
        inbox=(InboxMessage(kind="user", source="user", text="hi"),),
        started_at=1.0,
    )

    flight = InFlight.read(AgentSnapshot(turns=(_turn(), open_turn)))

    assert flight is not None
    assert flight.turn is open_turn
    assert flight.answered is False
    assert flight.requeue == open_turn.inbox


def test_in_flight_with_an_answer_is_replayed() -> None:
    open_turn = Turn(
        agent_id="a1",
        parent_id=None,
        depth=0,
        reply="partial",
        calls=(ToolCall(id="c1", name="shell", arguments={}),),
        started_at=1.0,
    )

    flight = InFlight.read(AgentSnapshot(turns=(open_turn,)))

    assert flight is not None
    assert flight.answered is True
    assert flight.requeue == ()

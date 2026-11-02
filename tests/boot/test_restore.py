"""Boot restore: the snapshot files are the whole live state."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from nan_fakes import FakeLLM, await_until

from nan.boot import restore
from nan.events import sink
from nan.execution import Sandbox
from nan.injection import InjectionHub
from nan.llm import ToolCall
from nan.persistence import AgentSnapshot, SnapshotStore
from nan.turn import INTERRUPTED, Death, InboxMessage, Turn, freeze_turn

pytestmark = pytest.mark.asyncio


def _done(agent_id: str, parent_id: str | None, depth: int, reply: str) -> Turn:
    return freeze_turn(
        Turn(
            agent_id=agent_id,
            parent_id=parent_id,
            depth=depth,
            origin="first",
            phase="committed",
            reply=reply,
            ended_at=time.time(),
        )
    )


async def _restore(store: SnapshotStore) -> Any:
    sink.reset(boot_id="test-boot")

    return await restore(
        llm=FakeLLM(),
        injections=InjectionHub(),
        sandbox=Sandbox(),
        snapshots=store,
    )


async def _stop(root: Any) -> None:
    root.request_death("test cleanup")

    if root.task is not None:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await root.task


async def test_restore_rebuilds_the_tree_and_closes_the_open_turn(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "snapshots")

    await store.save(
        AgentSnapshot(
            agent_id="r0",
            parent_id=None,
            depth=0,
            turns=(_done("r0", None, 0, "hello"),),
        )
    )

    await store.save(
        AgentSnapshot(
            agent_id="c0",
            parent_id="r0",
            depth=1,
            turns=(
                Turn(
                    agent_id="c0",
                    parent_id="r0",
                    depth=1,
                    origin="first",
                    phase="thought",
                    reply="",
                    calls=(ToolCall(id="k1", name="sleep", arguments={"seconds": 0}),),
                ),
            ),
        )
    )

    root = await _restore(store)

    assert root is not None
    assert root.agent_id == "r0"
    assert root.parent is None

    child = root.children.get("c0")

    assert child is not None
    assert child.parent is root
    assert child.depth == 1
    assert child.task is not None

    # The loop continues the open turn from its phase: the model
    # already answered, so its missing verb result is closed as
    # interrupted without re-running the verb.
    await await_until(lambda: bool(child.turns) and child.turns[0].phase == "committed")

    turn = child.turns[0]

    assert turn.ended_at is not None
    assert [result.output for result in turn.results] == [INTERRUPTED]
    assert turn.rendered is not None
    assert child.history[: len(turn.rendered)] == list(turn.rendered)

    # The completed root turn was already frozen; restore keeps it.
    assert root.history == list(root.last_turn.rendered)

    restored = [event["t"] for event in sink.history()]

    assert "snapshot_restored" in restored
    assert "turn_interrupted" in restored

    await _stop(root)


async def test_restore_drops_an_unanswered_turn_and_requeues_its_inbox(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "snapshots")

    lost = Turn(
        agent_id="r0",
        parent_id=None,
        depth=0,
        inbox=(InboxMessage(kind="user", source="user", text="lost input"),),
    )

    await store.save(AgentSnapshot(agent_id="r0", parent_id=None, depth=0, turns=(lost,)))

    root = await _restore(store)

    assert root is not None
    assert root.turns == []

    # The agent itself stays live: its snapshot is rewritten with
    # no turns, which is a spawned agent that observed nothing yet.
    assert store.load_all()["r0"].turns == ()
    assert len(root._inbox) == 1

    message = root._inbox.pop(0)

    assert message.text == "lost input"

    await _stop(root)


async def test_restore_delivers_an_unobserved_message(tmp_path: Path) -> None:
    """A message that arrived while no turn was open rides the
    snapshot, so a stop before the next turn loses nothing."""
    store = SnapshotStore(tmp_path / "snapshots")

    await store.save(
        AgentSnapshot(
            agent_id="r0",
            parent_id=None,
            depth=0,
            turns=(_done("r0", None, 0, "hello"),),
            pending=(InboxMessage(kind="user", source="user", text="typed mid-turn"),),
        )
    )

    root = await _restore(store)

    assert root is not None

    await await_until(lambda: len(root.turns) == 2 and root.turns[-1].phase == "committed")

    assert [message.text for message in root.turns[-1].inbox] == ["typed mid-turn"]

    await _stop(root)


async def test_restore_keeps_a_child_with_no_turns(tmp_path: Path) -> None:
    """A child spawned just before the stop is durable from birth:
    its snapshot has no turns, and restore keeps it live."""
    store = SnapshotStore(tmp_path / "snapshots")

    await store.save(
        AgentSnapshot(
            agent_id="r0",
            parent_id=None,
            depth=0,
            turns=(_done("r0", None, 0, "hello"),),
        )
    )

    await store.save(
        AgentSnapshot(
            agent_id="c0",
            parent_id="r0",
            depth=1,
            pending=(InboxMessage.from_parent("first task", source="r0"),),
        )
    )

    root = await _restore(store)

    assert root is not None

    child = root.children.get("c0")

    assert child is not None and child.task is not None

    await await_until(lambda: bool(child.turns) and child.turns[-1].phase == "committed")

    assert child.turns[0].inbox[0].text == "first task"

    await _stop(root)


async def test_restore_keeps_closed_turns_before_the_open_one(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "snapshots")

    await store.save(
        AgentSnapshot(
            agent_id="r0",
            parent_id=None,
            depth=0,
            turns=(
                _done("r0", None, 0, "hello"),
                Turn(agent_id="r0", parent_id=None, depth=0),
            ),
        )
    )

    root = await _restore(store)

    assert root is not None
    assert len(root.turns) == 1
    assert root.turns[0].ended_at is not None

    await _stop(root)


async def test_restore_discards_orphans_and_the_dead(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "snapshots")

    # No root at all: nothing is reachable.
    await store.save(
        AgentSnapshot(
            agent_id="c0",
            parent_id="missing",
            depth=1,
            turns=(_done("c0", "missing", 1, "orphan"),),
        )
    )

    assert await _restore(store) is None
    assert store.load_all() == {}

    # A turn carrying an exit record means the agent already died.
    dead = replace(_done("r9", None, 0, "gone"), death=Death("finished", "gone"))

    await store.save(AgentSnapshot(agent_id="r9", parent_id=None, depth=0, turns=(dead,)))

    assert await _restore(store) is None
    assert store.load_all() == {}


async def test_restore_lifts_a_parked_root(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "snapshots")

    parked = Turn(
        agent_id="r0", parent_id=None, depth=0, origin="first", phase="thought", reply="ok"
    )

    await store.save(AgentSnapshot(agent_id="r0", parent_id=None, depth=0, turns=(parked,)))

    root = await _restore(store)

    assert root is not None
    assert not root.paused

    # The parked turn is not a death: the loop lifts it and commits.
    await await_until(lambda: root.last_turn is not None and root.last_turn.phase == "committed")

    assert root.last_turn is not None and root.last_turn.ended_at is not None

    await _stop(root)

"""Boot restore: the snapshot files are the whole live state."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from nan_fakes import FakeLLM

from nan.agent.handle import Book
from nan.agent.sandbox import Sandbox
from nan.events import sink
from nan.injection import InjectionHub
from nan.llm import ToolCall
from nan.recovery import AgentSnapshot, SnapshotStore, restore
from nan.turn import INTERRUPTED, Death, InboxMessage, Turn, freeze_turn

pytestmark = pytest.mark.asyncio


def _done(agent_id: str, parent_id: str | None, depth: int, reply: str) -> Turn:
    return freeze_turn(
        Turn(
            agent_id=agent_id,
            parent_id=parent_id,
            depth=depth,
            origin="first",
            reply=reply,
            ended_at=time.time(),
        )
    )


async def _restore(store: SnapshotStore, *, book: Book | None = None) -> Any:
    sink.reset(boot_id="test-boot")

    return await restore(
        book=book or Book(),
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

    await store.save(AgentSnapshot(turns=(_done("r0", None, 0, "hello"),)))

    await store.save(
        AgentSnapshot(
            turns=(
                Turn(
                    agent_id="c0",
                    parent_id="r0",
                    depth=1,
                    origin="first",
                    reply="",
                    calls=(ToolCall(id="k1", name="sleep", arguments={"seconds": 0}),),
                ),
            )
        )
    )

    book = Book()

    root = await _restore(store, book=book)

    assert root is not None
    assert root.agent_id == "r0"
    assert root.parent is None

    child = book.get("c0")

    assert child is not None
    assert child.parent is root
    assert child.depth == 1
    assert root.child_handle("c0") is not None
    assert root.child_handle("c0").task is not None

    turn = child.turns[-1]

    assert turn.ended_at is not None
    assert [result.output for result in turn.results] == [INTERRUPTED]
    assert turn.rendered is not None
    assert len(child.history) == len(turn.rendered)

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

    await store.save(AgentSnapshot(turns=(lost,)))

    root = await _restore(store)

    assert root is not None
    assert root.turns == []
    assert not store.path("r0").exists()
    assert root._inbox.qsize() == 1

    message = root._inbox.get_nowait()

    assert message.text == "lost input"

    await _stop(root)


async def test_restore_keeps_closed_turns_before_the_open_one(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "snapshots")

    await store.save(
        AgentSnapshot(
            turns=(
                _done("r0", None, 0, "hello"),
                Turn(agent_id="r0", parent_id=None, depth=0),
            )
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
    await store.save(AgentSnapshot(turns=(_done("c0", "missing", 1, "orphan"),)))

    assert await _restore(store) is None
    assert store.load_all() == {}

    # A turn carrying an exit record means the agent already died.
    dead = replace(_done("r9", None, 0, "gone"), death=Death("finished", "gone"))

    await store.save(AgentSnapshot(turns=(dead,)))

    assert await _restore(store) is None
    assert store.load_all() == {}


async def test_restore_lifts_a_parked_root(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "snapshots")

    parked = Turn(agent_id="r0", parent_id=None, depth=0, origin="first", reply="ok", paused=True)

    await store.save(AgentSnapshot(turns=(parked,)))

    root = await _restore(store)

    assert root is not None
    assert not root.paused
    assert root.last_turn is not None and root.last_turn.ended_at is not None

    await _stop(root)

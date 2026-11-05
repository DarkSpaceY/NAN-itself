"""The spawn tree: durability at birth, subtree pause/suspend,
slot release and the depth/children limits."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest
from nan_fakes import FakeLLM, await_until, make_agent

from nan.events import sink
from nan.persistence import SnapshotStore
from nan.turn import InboxMessage

pytestmark = pytest.mark.asyncio


async def test_pause_freezes_the_whole_subtree(tmp_path: Path, no_sink: None) -> None:
    """Pause is one latch over the spawn tree: a paused root freezes
    its children too, and a resume lifts both."""
    root = make_agent(tmp_path=tmp_path, agent={"autonomous_interval": 0.0})

    root.start()

    child, err = await root.spawn_child("child pause probe")

    assert err is None and child is not None

    await await_until(lambda: bool(child.turns) and child.turns[-1].phase == "committed")

    root.pause()

    assert root.paused and child.paused

    # Frozen means frozen: a delivery waits in the child's inbox.
    child.deliver(InboxMessage.from_parent("after pause", source=root.agent_id))

    await asyncio.sleep(0.05)

    assert len(child.turns) == 1

    root.resume()

    assert not root.paused and not child.paused

    await await_until(lambda: len(child.turns) == 2)

    assert child.turns[-1].inbox[0].text == "after pause"

    root.request_death("test cleanup")

    with pytest.raises(asyncio.CancelledError):
        await root._task


async def test_pause_mid_spawn_still_starts_the_child(
    tmp_path: Path, no_sink: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pause that lands inside the child's first write must not
    orphan it: the child is registered and holds its task, so it
    starts (frozen) rather than sitting as a zombie."""
    llm = FakeLLM()

    llm.script("spawn-me", [("tool", "spawn", {"task": "late child"})])

    root = make_agent(tmp_path=tmp_path, llm=llm, agent={"autonomous_interval": 0.0})

    store = root._snapshots
    original = store.save
    entered = asyncio.Event()
    never = asyncio.Event()

    async def save(snapshot: Any) -> None:
        if not snapshot.turns:
            entered.set()

            await never.wait()  # cancelled by the pause

        await original(snapshot)

    monkeypatch.setattr(store, "save", save)

    root.start()

    root.deliver(InboxMessage.from_user("spawn-me"))

    await asyncio.wait_for(entered.wait(), timeout=3)

    root.pause()

    child = next(iter(root.children.values()))

    # The pause is delivered asynchronously: the spawn's finally
    # starts the child (frozen) right after the turn task unwinds.
    await await_until(lambda: child.task is not None)

    assert child.paused

    root.resume()

    assert not child.paused

    await await_until(lambda: bool(child.turns) and child.turns[-1].phase == "committed")

    assert child.turns[0].inbox[0].text == "late child"

    root.request_death("test cleanup")

    with pytest.raises(asyncio.CancelledError):
        await root._task


async def test_spawn_persists_the_child_before_it_runs(tmp_path: Path, no_sink: None) -> None:
    """A child is durable from birth: the snapshot exists with the
    task pending and no turns, before its loop takes anything."""
    root = make_agent(tmp_path=tmp_path)

    child, err = await root.spawn_child("durable from birth")

    assert err is None and child is not None

    snapshot = SnapshotStore(tmp_path / "snapshots").load_all()[child.agent_id]

    assert snapshot.turns == ()
    assert [message.text for message in snapshot.pending] == ["durable from birth"]

    child.request_death("test cleanup")

    with contextlib.suppress(asyncio.CancelledError):
        await child.task


async def test_shutdown_persists_the_whole_subtree(tmp_path: Path, no_sink: None) -> None:
    """Shutdown is persistence, not death: every agent in the
    subtree keeps its turn list on disk."""
    llm = FakeLLM()

    root = make_agent(tmp_path=tmp_path, llm=llm, agent={"autonomous_interval": 0.0})

    root.start()

    root.deliver(InboxMessage.from_user("root work"))

    await await_until(lambda: bool(root.turns) and root.turns[-1].phase == "committed")

    child, err = await root.spawn_child("child work")

    assert err is None and child is not None

    await await_until(lambda: bool(child.turns) and child.turns[-1].phase == "committed")

    root.suspend()

    with pytest.raises(asyncio.CancelledError):
        await root._task

    written = {path.stem for path in (tmp_path / "snapshots").glob("*.json")}

    assert written == {root.agent_id, child.agent_id}


async def test_max_depth_and_children_enforced(tmp_path: Path, no_sink: None) -> None:
    llm = FakeLLM()

    root = make_agent(
        tmp_path=tmp_path,
        llm=llm,
        agent={"max_depth": 1, "max_children": 1},
    )

    child, err = await root.spawn_child("A")

    assert err is None and child is not None

    # depth: child is at 1 == max_depth; a grandchild would be 2
    grandchild, err = await child.spawn_child("B")

    assert grandchild is None and err is not None

    assert "Max depth" in err

    # children: the one slot is taken
    sibling, err = await root.spawn_child("C")

    assert sibling is None and err is not None

    assert "Max children" in err


async def test_child_slot_released_after_death(tmp_path: Path, no_sink: None) -> None:
    """
    A dead child frees its max_children slot; otherwise every
    spawn would permanently consume one.
    """
    root = make_agent(tmp_path=tmp_path, agent={"max_children": 1})

    first, err = await root.spawn_child("A")

    assert err is None and first is not None

    first.request_death("test")

    assert first.task is not None

    with contextlib.suppress(asyncio.CancelledError):
        await first.task

    await await_until(lambda: first.agent_id not in root.children)

    second, err = await root.spawn_child("B")

    assert err is None and second is not None
    assert second.agent_id != first.agent_id


async def test_kind_is_named_by_the_edge(tmp_path: Path, no_sink: None) -> None:
    """The sender knows the edge it used, so it names the kind."""
    root = make_agent(tmp_path=tmp_path)

    child, err = await root.spawn_child("A")

    assert err is None and child is not None

    root.deliver(InboxMessage.from_child("report", source=child.agent_id))

    message = root._inbox.pop(0)

    assert message.kind == "subagent"
    assert message.source == child.agent_id


async def test_agent_exited_reason_is_the_death_reason(tmp_path: Path, no_sink: Any) -> None:
    """The exit event carries the death record verbatim."""
    sink.reset(boot_id="test-boot")

    llm = FakeLLM()

    llm.script("finish now", [("tool", "finish", {"report": "done"})])

    root = make_agent(tmp_path=tmp_path, llm=llm, agent={"autonomous_interval": 0.0})

    child, err = await root.spawn_child("finish now")

    assert err is None and child is not None

    if child.task is not None:
        with contextlib.suppress(asyncio.CancelledError):
            await child.task

    killed, err = await root.spawn_child("wait here")

    assert err is None and killed is not None

    killed.request_death(f"killed by parent {root.agent_id}")

    if killed.task is not None:
        with contextlib.suppress(asyncio.CancelledError):
            await killed.task

    exits = {
        event["content"]["agent_id"]: event["content"]
        for event in sink.history()
        if event["t"] == "agent_exited"
    }

    assert exits[child.agent_id]["reason"] == "finished"
    assert exits[child.agent_id]["report"] == "done"
    assert exits[killed.agent_id]["reason"] == f"killed by parent {root.agent_id}"
    assert exits[killed.agent_id]["report"] is None

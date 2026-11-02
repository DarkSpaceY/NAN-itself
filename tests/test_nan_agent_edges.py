"""Behavior increments over nan_core, as pytest edge cases:
pause parks the resident turn, history clear-all, persona re-read
per turn, max depth/children."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest
from nan_fakes import FakeLLM, SlowFakeLLM, await_until, make_agent

from nan.persistence import SnapshotStore
from nan.turn import InboxMessage

pytestmark = pytest.mark.asyncio


class ExplodingLLM(FakeLLM):
    """FakeLLM whose model call raises: the turn pipeline fails."""

    async def generate(self, request: Any) -> Any:
        self.requests.append(request)

        raise RuntimeError("model exploded")

        yield  # pragma: no cover


async def test_pause_parks_turn_and_resume_continues_it(tmp_path: Path, no_sink: None) -> None:
    llm = SlowFakeLLM()

    llm.script("pause-me", [("text",)])

    root = make_agent(tmp_path=tmp_path, llm=llm, agent={"autonomous_interval": 0.0})

    root.start()

    assert root._task is not None

    # a user message starts a turn; the model call blocks mid-turn
    root.deliver(InboxMessage.from_user("pause-me"))

    await asyncio.wait_for(llm.first_call_started.wait(), timeout=3)

    await await_until(lambda: root._turn_task is not None and not root._turn_task.done())

    root.pause()

    # the turn stays resident with its input; nothing is requeued
    await await_until(lambda: root.last_turn is not None and root.last_turn.phase == "injected")

    assert root._inbox == []

    assert root.history == []
    assert root.last_turn is not None and root.last_turn.reply is None

    # resume: the same turn re-runs its model call and commits
    llm.release_first.set()

    root.resume()

    await await_until(lambda: root.last_turn is not None and root.last_turn.ended_at is not None)

    assert len(root.turns) == 1
    assert len(root.history) == 2  # observation + assistant
    assert "pause-me" in root.history[0].content
    assert not root.paused

    root.request_death("test cleanup")

    with pytest.raises(asyncio.CancelledError):
        await root._task


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


async def test_pause_is_idempotent_and_resume_wakes_waiter(tmp_path: Path, no_sink: None) -> None:
    llm = SlowFakeLLM()

    llm.script("hello", [("text",)])

    root = make_agent(tmp_path=tmp_path, llm=llm)

    root.start()

    assert root._task is not None

    root.pause()

    root.pause()  # idempotent, no turn in flight

    root.deliver(InboxMessage.from_user("hello"))

    await asyncio.sleep(0.05)  # paused: the loop must not run a turn

    assert len(root.turns) == 0 and root.history == []

    llm.release_first.set()

    root.resume()

    root.resume()  # idempotent

    await await_until(lambda: len(root.turns) == 1)

    root.request_death("test cleanup")

    with pytest.raises(asyncio.CancelledError):
        await root._task


async def test_history_over_limit_clears_at_the_next_epoch(tmp_path: Path, no_sink: None) -> None:
    llm = FakeLLM()

    root = make_agent(
        tmp_path=tmp_path,
        llm=llm,
        agent={"history_char_limit": 200},
    )

    # turn 1: small, kept
    root.deliver(InboxMessage.from_user("short"))
    await root.run_turn()

    assert len(root.history) == 2

    # turn 2: blows the limit, but the epoch is only cut when the
    # next turn opens
    root.deliver(InboxMessage.from_user("X" * 5000))
    await root.run_turn()

    assert len(root.history) == 4

    # turn 3: opens a new epoch; turns before it are dropped
    root.deliver(InboxMessage.from_user("after clear"))
    turn = await root.run_turn()

    assert turn.origin == "context_cleared"
    assert len(root.turns) == 1
    assert len(root.history) == 2
    assert "after clear" in root.history[0].content
    assert "short" not in root.history[0].content


async def test_persona_change_applies_from_that_turn(tmp_path: Path, no_sink: None) -> None:
    persona_file = tmp_path / "persona.md"

    persona_file.write_text("PERSONA_V1", encoding="utf-8")

    llm = FakeLLM()

    root = make_agent(tmp_path=tmp_path, llm=llm, persona_path=persona_file)

    root.deliver(InboxMessage.from_user("first"))
    t1 = await root.run_turn()

    assert llm.requests[0].messages[0].content == "PERSONA_V1"
    assert t1.persona is None  # unchanged since boot

    # content change = legal system-prompt invalidation point
    persona_file.write_text("PERSONA_V2", encoding="utf-8")

    root.deliver(InboxMessage.from_user("second"))
    t2 = await root.run_turn()

    assert llm.requests[1].messages[0].content == "PERSONA_V2"
    assert t2.persona == "PERSONA_V2"  # the change rides the turn

    assert root._persona_last == "PERSONA_V2"

    # unchanged content: no action, same prompt
    root.deliver(InboxMessage.from_user("third"))
    t3 = await root.run_turn()

    assert llm.requests[2].messages[0].content == "PERSONA_V2"
    assert t3.persona is None


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


async def test_failed_turn_parks_the_root(tmp_path: Path, no_sink: None) -> None:
    """
    A fatal failure parks the turn instead of discarding it: the
    turn keeps its inputs and facts, the agent keeps the turn
    resident, and the root waits for a resume rather than dying.
    """
    root = make_agent(tmp_path=tmp_path, llm=ExplodingLLM())

    root.deliver(InboxMessage.from_user("lost input"))

    turn = await root.run_turn()

    assert turn.phase == "injected"
    assert turn.ended_at is None
    assert turn.reply is None
    assert len(root.turns) == 1
    assert root.history == []
    assert root.paused
    assert root.last_turn is not None and root.last_turn.phase == "injected"


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


async def test_interval_agent_paces_itself(tmp_path: Path, no_sink: None) -> None:
    """An interval agent takes one turn per interval with no input
    at all: the cadence alone opens turns."""
    llm = FakeLLM()

    root = make_agent(tmp_path=tmp_path, llm=llm, agent={"autonomous_interval": 0.05})

    root.start()

    assert root._task is not None

    # no deliver: two turns come from the cadence
    await await_until(lambda: len(root.turns) >= 2)

    first, second = root.turns[0], root.turns[1]

    assert first.inbox == () and second.inbox == ()
    assert first.ended_at is not None

    # the second turn opened no earlier than one interval after the
    # first ended
    assert second.started_at - first.ended_at >= 0.05

    root.request_death("test cleanup")

    with pytest.raises(asyncio.CancelledError):
        await root._task


async def test_tool_call_step_continues_without_new_input(tmp_path: Path, no_sink: None) -> None:
    """
    Event-driven ReACT: a turn that called a verb still owes the
    model its results, so the loop takes that next step on its own
    and parks only once a turn ends without calls.
    """
    llm = FakeLLM()

    llm.script("react-probe", [("tool", "sleep", {"seconds": 0}), ("text",)])

    root = make_agent(tmp_path=tmp_path, llm=llm, agent={"autonomous_interval": 0.0})

    root.start()

    assert root._task is not None

    root.deliver(InboxMessage.from_user("react-probe"))

    # no second input: the tool result itself drives turn 2
    await await_until(lambda: len(root.turns) == 2)

    assert len(llm.requests) == 2

    # the follow-up request already carries turn 1's tool result
    assert [m.role for m in llm.requests[1].messages].count("tool") == 1

    # turn 2 had no calls: the loop parks here
    assert root.last_turn is not None and root.last_turn.calls == ()

    root.request_death("test cleanup")

    with pytest.raises(asyncio.CancelledError):
        await root._task

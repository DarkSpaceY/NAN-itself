"""Ported smoke test: spawn -> finish -> report, kill -> death
notice, send up/down, whole-tree death (from
paper_search/nan_core/tests/smoke_agents.py)."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest
from nan_fakes import FakeLLM, await_until, make_agent

from nan.agent.core import Death
from nan.agent.handle import Book
from nan.agent.verbs import tool_definitions

pytestmark = pytest.mark.asyncio


async def test_spawn_finish_report(tmp_path: Path, no_sink: None) -> None:
    book = Book()

    llm = FakeLLM()

    root = make_agent(tmp_path=tmp_path, book=book, llm=llm)

    # 1. spawn via verb, child finishes, report lands in root's next turn
    llm.script("ROOTPERSONA", [("tool", "spawn", {"task": "TASKSMOKE"})])
    await root.run_turn()

    child = next(a for a in book.all() if a.depth == 1)

    assert child.depth == 1

    llm.script("TASKSMOKE", [("tool", "finish", {"report": "SMOKE_REPORT"})])

    assert child._task is not None

    await child._task

    assert child._death == Death("finished", "SMOKE_REPORT")

    turn = await root.run_turn()

    assert any(
        m.kind == "subagent" and m.event == "finish" and m.text == "SMOKE_REPORT"
        for m in turn.inbox
    )

    # 2. the report is in history; rendered frozen; add-only holds
    assert turn.rendered is not None

    assert any("SMOKE_REPORT" in (m.content or "") for m in root._messages)


async def test_kill_produces_death_notice(tmp_path: Path, no_sink: None) -> None:
    book = Book()

    root = make_agent(tmp_path=tmp_path, book=book, llm=FakeLLM())

    victim, err = root.spawn_child("VICTIMTASK")

    assert err is None and victim is not None

    assert victim._task is not None

    # event-driven child consumes its task turn, then blocks idle
    await await_until(lambda: victim.cycles == 1)

    victim.request_death("smoke kill")

    with contextlib.suppress(asyncio.CancelledError):
        await victim._task

    turn = await root.run_turn()

    assert any(e.section == "subagent" and f"-{victim.agent_id} died" in e.line for e in turn.delta)


async def test_send_down_and_up(tmp_path: Path, no_sink: None) -> None:
    llm = FakeLLM()

    root = make_agent(tmp_path=tmp_path, llm=llm)

    target, err = root.spawn_child("SENDTARGET")

    assert err is None and target is not None

    llm.script("SENDTARGET", [("text", "child hears you")])

    handle = root.child_handle(target.agent_id)

    assert handle is not None

    await handle.send("hello down")

    # the message is consumed at the child's turn boundary and
    # lives in its frozen history
    await await_until(lambda: "hello down" in " ".join((m.content or "") for m in target._messages))

    # child sends up: direct parent reference routing
    assert target.parent is root

    await target.parent.receive_up("hello up", target.agent_id)

    turn = await root.run_turn()

    assert "hello up" in [m.text for m in turn.inbox]


async def test_root_death_cascades_and_book_empties(tmp_path: Path, no_sink: None) -> None:
    book = Book()

    llm = FakeLLM()

    root = make_agent(tmp_path=tmp_path, book=book, llm=llm)

    llm.script("ROOTPERSONA", [("tool", "spawn", {"task": "CASCADETASK"})])
    await root.run_turn()

    # a second, deeper level: grandchild under the child
    child = next(a for a in book.all() if a.depth == 1)

    assert child is not None

    grandchild, err = child.spawn_child("GRANDTASK")

    assert err is None and grandchild is not None

    root.start()

    assert root._task is not None

    root.request_death("smoke shutdown")

    with contextlib.suppress(asyncio.CancelledError):
        await root._task

    await await_until(lambda: book.all() == [])

    assert book.all() == []

    assert grandchild._death is not None and grandchild._death.report is None


async def test_finish_verb_scope(tmp_path: Path, no_sink: None) -> None:
    root = make_agent(tmp_path=tmp_path)

    assert "finish" not in [t.name for t in tool_definitions(depth=root.depth)]

    child, _ = root.spawn_child("FINISHTASK")

    assert child is not None

    assert "finish" in [t.name for t in tool_definitions(depth=child.depth)]

    # the verb itself refuses at the root, runs below it
    class Call:
        name = "finish"

        arguments = {"report": "nope"}

    assert "only available to subagents" in await root._run_call(Call())

    assert "finished" in await child._run_call(Call())

    assert child._death == Death("finished", "nope")


class RecordingHub:
    """Injection stand-in: records collections."""

    def __init__(self) -> None:
        self.collects: list[tuple[Any, Any]] = []

    async def collect(self, turn: Any, *, prev_turn: Any = None) -> tuple[tuple, list]:
        self.collects.append((turn, prev_turn))

        return (), []

    def forget(self, agent_id: str) -> None:
        pass


async def test_turn_states_and_exit_delivery(tmp_path: Path, no_sink: None) -> None:
    hub = RecordingHub()

    agent = make_agent(
        tmp_path=tmp_path,
        book=Book(),
        llm=FakeLLM(),
        agent={"history_char_limit": 1},
    )

    agent.injections = hub

    # 1. first turn: start (run_turn direct calls bypass the loop,
    #    so the loop's post-turn cycles increment is mimicked)
    t1 = await agent.run_turn()
    assert t1.state == "start"
    assert hub.collects[0][1] is None

    agent.cycles += 1

    # 2. history over the 1-char limit cleared wholesale -> next
    #    turn opens a new context epoch
    t2 = await agent.run_turn()
    assert t2.state == "context_cleared"
    assert hub.collects[1][1] is t1

    # 3. exit delivery: the same collect call, with no arriving
    #    turn; the last turn complete rides prev_turn, marked end
    agent.request_death("test")

    await agent._teardown()

    assert len(hub.collects) == 3

    exit_turn, exit_prev = hub.collects[2]

    assert exit_turn is None
    assert exit_prev.state == "end"
    assert exit_prev.agent_id == t2.agent_id
    assert exit_prev.reply is not None
    assert exit_prev.rendered is not None


async def test_running_state_without_clear(tmp_path: Path, no_sink: None) -> None:
    hub = RecordingHub()

    agent = make_agent(
        tmp_path=tmp_path,
        book=Book(),
        llm=FakeLLM(),
    )

    agent.injections = hub

    t1 = await agent.run_turn()

    agent.cycles += 1

    t2 = await agent.run_turn()

    assert (t1.state, t2.state) == ("start", "running")

    agent.request_death("test")

    with contextlib.suppress(Exception):
        await agent._teardown()

    exit_turn, exit_prev = hub.collects[-1]

    assert exit_turn is None
    assert exit_prev.state == "end"
    assert exit_prev.agent_id == t2.agent_id

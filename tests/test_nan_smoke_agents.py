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
    """Injection stand-in: records injections and archives."""

    def __init__(self) -> None:
        self.injects: list[Any] = []
        self.archives: list[Any] = []

    async def inject(self, turn: Any) -> tuple[tuple, list]:
        self.injects.append(turn)

        return (), []

    async def archive(self, turn: Any) -> None:
        self.archives.append(turn)

    def forget(self, agent_id: str) -> None:
        pass


async def test_turn_origins_and_archive(tmp_path: Path, no_sink: None) -> None:
    hub = RecordingHub()

    agent = make_agent(
        tmp_path=tmp_path,
        book=Book(),
        llm=FakeLLM(),
        agent={"history_char_limit": 1},
    )

    agent.injections = hub

    # 1. first turn: origin=first (run_turn direct calls bypass the
    #    loop, so the loop's post-turn cycles increment is mimicked)
    t1 = await agent.run_turn()
    assert t1.origin == "first"

    agent.cycles += 1

    # 2. history over the 1-char limit cleared wholesale -> next
    #    turn opens a new context epoch
    t2 = await agent.run_turn()
    assert t2.origin == "context_cleared"

    # 3. every turn that ended is archived once, in order; the exit
    #    itself delivers nothing
    assert hub.archives == [t1, t2]

    agent.request_death("test")

    await agent._teardown()

    assert hub.archives == [t1, t2]
    assert len(hub.injects) == 2


async def test_normal_origin_without_clear(tmp_path: Path, no_sink: None) -> None:
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

    assert (t1.origin, t2.origin) == ("first", "normal")

    agent.request_death("test")

    with contextlib.suppress(Exception):
        await agent._teardown()

    assert hub.archives == [t1, t2]
    assert len(hub.injects) == 2


async def test_thinking_enters_history_and_replay(tmp_path: Path, no_sink: None) -> None:
    llm = FakeLLM()
    llm.thinking = "REASONING_MARKER"

    agent = make_agent(tmp_path=tmp_path, book=Book(), llm=llm)

    t1 = await agent.run_turn()

    # The turn records the reasoning, and the frozen render carries it.
    assert [block.text for block in t1.thinking] == ["REASONING_MARKER"]
    assert t1.rendered is not None

    assistant = next(m for m in t1.rendered if m.role == "assistant")
    assert [block.text for block in assistant.thinking] == ["REASONING_MARKER"]

    agent.cycles += 1
    await agent.run_turn()

    # Strict cache: the next request replays the frozen assistant turn verbatim.
    replayed = [m for m in llm.requests[1].messages if m.role == "assistant"][0]
    assert [block.text for block in replayed.thinking] == ["REASONING_MARKER"]

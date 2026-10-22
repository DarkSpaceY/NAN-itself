"""Behavior increments over nan_core, as pytest edge cases:
pause discards the interrupted turn, history clear-all, persona
re-read per turn, max depth/children."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from nan_fakes import FakeLLM, SlowFakeLLM, await_until, make_agent

from nan.agent import AgentConfig

pytestmark = pytest.mark.asyncio


async def test_pause_discards_turn_and_resume_opens_clean_turn(
    tmp_path: Path, no_sink: None
) -> None:
    llm = SlowFakeLLM()

    llm.script("pause-me", [("text",)])

    root = make_agent(tmp_path=tmp_path, llm=llm, config=AgentConfig(autonomous_interval=0))

    root.start()

    assert root._task is not None

    # a user message starts a turn; the model call blocks mid-turn
    await root.receive_up("pause-me", "user")

    await asyncio.wait_for(llm.first_call_started.wait(), timeout=3)

    await await_until(lambda: root._turn_task is not None and not root._turn_task.done())

    root.pause()

    # the interrupted turn is discarded whole: nothing rendered,
    # nothing in history; its input rides back into the inbox
    await await_until(lambda: root._inbox.qsize() == 1)

    assert root._messages == []

    assert root.last_turn is None

    assert root._paused

    # resume: a clean turn re-reads the re-queued input
    llm.release_first.set()

    root.resume()

    await await_until(lambda: root.cycles == 1)

    assert len(root._messages) == 2  # observation + assistant

    assert "pause-me" in root._messages[0].content

    assert not root._paused

    root.request_death("test cleanup")

    with pytest.raises(asyncio.CancelledError):
        await root._task


async def test_pause_is_idempotent_and_resume_wakes_waiter(tmp_path: Path, no_sink: None) -> None:
    llm = SlowFakeLLM()

    llm.script("hello", [("text",)])

    root = make_agent(tmp_path=tmp_path, llm=llm)

    root.start()

    assert root._task is not None

    root.pause()

    root.pause()  # idempotent, no turn in flight

    await root.receive_up("hello", "user")

    await asyncio.sleep(0.05)  # paused: the loop must not run a turn

    assert root.cycles == 0 and root._messages == []

    llm.release_first.set()

    root.resume()

    root.resume()  # idempotent

    await await_until(lambda: root.cycles == 1)

    root.request_death("test cleanup")

    with pytest.raises(asyncio.CancelledError):
        await root._task


async def test_history_over_limit_clears_all(tmp_path: Path, no_sink: None) -> None:
    llm = FakeLLM()

    root = make_agent(
        tmp_path=tmp_path,
        llm=llm,
        config=AgentConfig(history_char_limit=200),
    )

    # turn 1: small, kept
    await root.receive_up("short", "user")
    await root.run_turn()
    assert len(root._messages) == 2

    # turn 2: blows the limit -> wholesale clear (a reset, not a rewrite)
    await root.receive_up("X" * 5000, "user")
    await root.run_turn()
    assert root._messages == []

    # turn 3: history starts from scratch
    await root.receive_up("after clear", "user")
    await root.run_turn()
    assert len(root._messages) == 2
    assert "after clear" in root._messages[0].content
    assert "short" not in root._messages[0].content


async def test_persona_change_applies_from_that_turn(tmp_path: Path, no_sink: None) -> None:
    persona_file = tmp_path / "persona.md"

    persona_file.write_text("PERSONA_V1", encoding="utf-8")

    llm = FakeLLM()

    root = make_agent(tmp_path=tmp_path, llm=llm, persona_path=persona_file)

    await root.receive_up("first", "user")
    await root.run_turn()

    assert llm.requests[0].messages[0].content == "PERSONA_V1"

    # content change = legal system-prompt invalidation point
    persona_file.write_text("PERSONA_V2", encoding="utf-8")

    await root.receive_up("second", "user")
    await root.run_turn()

    assert llm.requests[1].messages[0].content == "PERSONA_V2"

    assert root._persona_last == "PERSONA_V2"

    # unchanged content: no action, same prompt
    await root.receive_up("third", "user")
    await root.run_turn()

    assert llm.requests[2].messages[0].content == "PERSONA_V2"


async def test_max_depth_and_children_enforced(tmp_path: Path, no_sink: None) -> None:
    llm = FakeLLM()

    root = make_agent(
        tmp_path=tmp_path,
        llm=llm,
        config=AgentConfig(max_depth=1, max_children=1),
    )

    child, err = root.spawn_child("A")

    assert err is None and child is not None

    # depth: child is at 1 == max_depth; a grandchild would be 2
    grandchild, err = child.spawn_child("B")

    assert grandchild is not None and err is not None

    assert "Max depth" in err

    # children: the one slot is taken
    sibling, err = root.spawn_child("C")

    assert sibling is not None and err is not None

    assert "Max children" in err


async def test_unverified_source_delivered_as_unknown(tmp_path: Path, no_sink: None) -> None:
    llm = FakeLLM()

    root = make_agent(tmp_path=tmp_path, llm=llm)

    await root.receive_up("hello", "stranger")

    assert root._inbox.qsize() == 1

    message = root._inbox.get_nowait()

    assert message.kind == "unknown"
    assert message.source == "stranger"

    # a held child is labelled subagent, not unknown
    child, err = root.spawn_child("A")

    assert err is None and child is not None

    await root.receive_up("report", child.agent_id)

    message = root._inbox.get_nowait()

    assert message.kind == "subagent"

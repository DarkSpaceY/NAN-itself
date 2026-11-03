"""The agent loop: pause/resume, context epochs, persona, pacing,
ReACT continuation and the failed-turn terminal."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from nan_fakes import FakeLLM, SlowFakeLLM, await_until, make_agent

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
    assert root.last_turn is not None and root.last_turn.verb_calls == ()

    root.request_death("test cleanup")

    with pytest.raises(asyncio.CancelledError):
        await root._task

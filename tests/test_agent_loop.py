from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from src.nan_itself.agent.loop import (
    AgentLoop,
    Inbox,
)


class StubAgent:
    """
    Duck-typed CoreAgent for loop-level tests.
    """

    def __init__(self) -> None:
        self.runs: list[str] = []
        self._pending_reports: list[str] = []
        self.fail_next = 0
        self.started_turns = 0
        self.finished_turns = 0
        self.block: asyncio.Event | None = None

    def has_pending_reports(self) -> bool:
        return bool(self._pending_reports)

    async def run(self, inputs: str) -> object:
        self.started_turns += 1

        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError("llm down")

        if self.block is not None:
            await self.block.wait()

        self.runs.append(inputs)
        self._pending_reports.clear()
        self.finished_turns += 1

        return object()


def make_loop(agent: StubAgent, **kwargs):
    inbox = Inbox()

    loop = AgentLoop(
        agent,
        inbox,
        backoff=(0.01, 0.02),
        **kwargs,
    )

    return inbox, loop


async def wait_until(
    predicate,
    *,
    timeout: float = 2.0,
) -> None:
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return

        await asyncio.sleep(0.01)

    raise AssertionError("condition not met in time")


# ============================================================================
# Cycle behaviour
# ============================================================================


@pytest.mark.asyncio
async def test_inbox_items_are_joined_into_one_cycle():
    agent = StubAgent()

    inbox, loop = make_loop(agent)

    inbox.put("first")
    inbox.put("second")

    task = asyncio.create_task(loop.run_forever())

    await wait_until(lambda: len(agent.runs) == 1)

    assert agent.runs[0] == "first\n\nsecond"

    inbox.put("third")

    await wait_until(lambda: len(agent.runs) == 2)

    assert agent.runs[1] == "third"

    loop.request_stop()

    await asyncio.wait_for(task, timeout=2.0)

    # Blocking on an empty inbox did not add cycles.
    assert len(agent.runs) == 2


@pytest.mark.asyncio
async def test_loop_survives_repeated_failures():
    agent = StubAgent()

    agent.fail_next = 2

    inbox, loop = make_loop(agent)

    inbox.put("go")

    task = asyncio.create_task(loop.run_forever())

    await wait_until(lambda: agent.finished_turns == 1)

    # Three attempts: two failures then success.
    assert agent.started_turns == 3
    assert loop.cycles == 1

    loop.request_stop()

    await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_late_report_triggers_empty_input_cycle():
    agent = StubAgent()

    agent._pending_reports.append("pending report")

    inbox, loop = make_loop(agent)

    task = asyncio.create_task(loop.run_forever())

    await wait_until(lambda: len(agent.runs) == 1)

    # The report itself is seeded by the agent; the loop only
    # supplies an empty input so delivery happens promptly.
    assert agent.runs[0] == ""

    loop.request_stop()

    await asyncio.wait_for(task, timeout=2.0)


# ============================================================================
# Shutdown
# ============================================================================


@pytest.mark.asyncio
async def test_stop_mid_turn_waits_grace_then_cancels():
    agent = StubAgent()

    block = asyncio.Event()

    agent.block = block

    inbox, loop = make_loop(
        agent,
        turn_grace=0.05,
    )

    inbox.put("long work")

    task = asyncio.create_task(loop.run_forever())

    await wait_until(lambda: agent.started_turns == 1)

    loop.request_stop()

    await asyncio.wait_for(task, timeout=2.0)

    # The blocked turn never finished; it was cancelled.
    assert agent.finished_turns == 0


@pytest.mark.asyncio
async def test_stop_finishes_turn_within_grace():
    agent = StubAgent()

    started = asyncio.Event()

    original_run = agent.run

    async def slow_run(inputs: str):
        started.set()

        await asyncio.sleep(0.05)

        return await original_run(inputs)

    agent.run = slow_run

    inbox, loop = make_loop(
        agent,
        turn_grace=5.0,
    )

    inbox.put("quick-ish work")

    task = asyncio.create_task(loop.run_forever())

    await wait_until(lambda: started.is_set())

    loop.request_stop()

    await asyncio.wait_for(task, timeout=2.0)

    # The turn completed inside the grace window.
    assert agent.finished_turns == 1


# ============================================================================
# Inbox
# ============================================================================


def test_inbox_drops_oldest_on_overflow():
    inbox = Inbox(maxsize=3)

    for item in ["a", "b", "c", "d"]:
        inbox.put(item)

    assert inbox.drain() == ["b", "c", "d"]


def test_inbox_ignores_empty_items():
    inbox = Inbox()

    inbox.put("")
    inbox.put("x")

    assert inbox.drain() == ["x"]

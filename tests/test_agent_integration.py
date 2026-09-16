from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from nan_itself.agent.core import CoreAgent
from nan_itself.modules.loading import import_module_class
from nan_itself.modules.model import DataSpace, Turn


# ============================================================================
# Async helpers
# ============================================================================


def run(coro):
    return asyncio.run(coro)


# ============================================================================
# Test doubles
# ============================================================================


class FakeModules:
    """
    Minimal live Module facade.

    snapshot() is the exact boundary that turns live Module state
    into one Agent-turn world.
    """

    def __init__(self):
        self.space = DataSpace(
            owner="state",
            initial={
                "value": 1,
            },
        )

        self.snapshot_calls = 0

        self.inbox = None

    def snapshot(self):
        self.snapshot_calls += 1

        return {
            "state": self.space.snapshot(),
        }

    async def query_snapshot(
        self,
        turn,
        **kwargs,
    ):
        return []

    def deliver_turn(
        self,
        record,
    ):
        pass

    def get(
        self,
        module_id: str,
    ):
        return self.inbox


class FakeSkills:
    """
    Skill runtime that exposes only refresh/catalog semantics needed
    by CoreAgent.
    """

    def __init__(self):
        self.refresh_calls = 0

    def refresh(self):
        self.refresh_calls += 1

    def catalog(self):
        return ()

    def names(self):
        return ()


class FakeProviders:
    def provider_names(self):
        return ()

    def get_provider(
        self,
        name,
    ):
        return None


class FakeLLM:
    model = "fake-model"
    provider = "fake-provider"


@dataclass
class CapturedExecution:
    context: object
    persona: str
    history: tuple


class CapturingEngine:
    """
    Replaces StepEngine while retaining CoreAgent's real turn-boundary
    behavior.

    This allows us to verify exactly what CoreAgent constructs and
    passes into the execution engine.
    """

    def __init__(self):
        self.executions: list[
            CapturedExecution
        ] = []

        self.on_execute = None

        self.reply = "done"

    async def execute(
        self,
        *,
        context,
        persona,
        history,
        report_sink=None,
        sink=None,
    ):
        self.executions.append(
            CapturedExecution(
                context=context,
                persona=persona,
                history=tuple(history),
            )
        )

        callback = self.on_execute

        if callback is not None:
            outcome = callback(
                self,
                context,
            )

            if asyncio.iscoroutine(
                outcome
            ):
                await outcome

        return SimpleNamespace(
            content=self.reply,
            messages=(
                SimpleNamespace(
                    content="assistant-done",
                ),
            ),
        )


def make_agent(
    *,
    engine=None,
    history_char_limit=100_000,
    autonomous_interval=0.0,
):
    core = CoreAgent(
        llm=FakeLLM(),
        modules=FakeModules(),
        tools=FakeProviders(),
        skills=FakeSkills(),
        persona_source=lambda: "You are NAN.",
        history_char_limit=history_char_limit,
        autonomous_interval=autonomous_interval,
    )

    if engine is not None:
        core.engine = engine

    return core


# ============================================================================
# Turn boundaries (CoreAgent.run)
# ============================================================================


def test_run_assembles_turn_boundaries():
    engine = CapturingEngine()
    core = make_agent(engine=engine)

    result = run(core.run())

    assert result.content == "done"

    # One execution captured, one turn's messages retained.
    assert len(engine.executions) == 1

    captured = engine.executions[0]

    assert captured.persona == "You are NAN."
    assert captured.history == ()

    assert [
        m.content for m in core.history
    ] == ["assistant-done"]

    # Per-turn boundaries: refresh + snapshot exactly once.
    assert core.skills.refresh_calls == 1
    assert core.modules.snapshot_calls == 1

    # Root context: depth 0, no task (input rides in the inbox).
    assert captured.context.depth == 0
    assert captured.context.task is None


def test_run_extends_history_across_turns():
    engine = CapturingEngine()
    core = make_agent(engine=engine)

    run(core.run())
    run(core.run())

    first, second = engine.executions

    assert first.history == ()
    assert [
        m.content for m in second.history
    ] == ["assistant-done"]


def test_run_clears_history_over_char_limit():
    engine = CapturingEngine()
    core = make_agent(engine=engine)

    core.history_char_limit = 5

    core.history.append(
        SimpleNamespace(
            content="x" * 100,
        )
    )

    run(core.run())

    # History was cleared before the turn; the new turn's
    # messages are then appended.
    assert engine.executions[0].history == ()
    assert [
        m.content for m in core.history
    ] == ["assistant-done"]


# ============================================================================
# Subagent report parking
# ============================================================================


def test_park_reports_routes_to_inbox_module():
    core = make_agent()

    class FakeInbox:
        def __init__(self):
            self.items = []

        def put(
            self,
            item,
        ):
            self.items.append(item)

    fake = FakeInbox()
    core.modules.inbox = fake

    core._park_reports(
        [
            "[Subagent Report]\nstatus: completed",
            "[Subagent Report]\nstatus: failed",
        ]
    )

    assert len(fake.items) == 2


def test_park_reports_drops_without_inbox():
    core = make_agent()

    # No inbox module running: must not raise.
    core._park_reports(
        [
            "[Subagent Report]\nstatus: completed",
        ]
    )


# ============================================================================
# Autonomous loop (CoreAgent.run_forever)
# ============================================================================


def test_run_forever_runs_turns_until_stop():
    engine = CapturingEngine()

    core = make_agent(engine=engine)

    def stop_after_second(
        engine_self,
        context,
    ):
        if len(engine_self.executions) >= 2:
            core.request_stop()

    engine.on_execute = stop_after_second

    run(core.run_forever())

    assert len(engine.executions) == 2
    assert core.cycles == 2
    assert core.stopping


def test_run_forever_backoff_retries_failed_turn():
    engine = CapturingEngine()

    core = make_agent(engine=engine)
    core.backoff = (0.0,)

    calls = 0

    def fail_once(
        engine_self,
        context,
    ):
        nonlocal calls

        calls += 1

        if calls == 1:
            raise RuntimeError("boom")

        core.request_stop()

    engine.on_execute = fail_once

    run(core.run_forever())

    # The failed turn retried immediately (zero backoff) and the
    # retry succeeded.
    assert calls == 2
    assert core.cycles == 1


def test_run_forever_grace_completes_turn_on_stop():
    engine = CapturingEngine()

    core = make_agent(engine=engine)
    core.turn_grace = 2.0

    async def stop_and_finish(
        engine_self,
        context,
    ):
        core.request_stop()

    engine.on_execute = stop_and_finish

    run(core.run_forever())

    # The in-flight turn was allowed to finish inside the grace
    # window and its result counted as a completed cycle.
    assert len(engine.executions) == 1
    assert core.cycles == 1


def test_run_forever_cancels_turn_after_grace():
    engine = CapturingEngine()

    core = make_agent(engine=engine)
    core.turn_grace = 0.05

    async def stop_and_hang(
        engine_self,
        context,
    ):
        core.request_stop()

        await asyncio.sleep(
            1.0,
        )

    engine.on_execute = stop_and_hang

    run(core.run_forever())

    # Grace expired: the turn was cancelled mid-flight.
    assert core.stopping
    assert core.cycles == 0


# ============================================================================
# InboxModule semantics (loaded straight from its hot-reload file)
# ============================================================================


def _turn(depth=0):
    return Turn(
        agent_hash="h",
        parent_hash=None,
        depth=depth,
        task=None,
        world={},
    )


def _load_inbox(max_size=256):
    path = (
        Path(__file__)
        .resolve()
        .parents[1]
        / "builtin"
        / "modules"
        / "inbox.py"
    )

    cls, _, _ = import_module_class(path)

    inbox = cls()

    inbox.max_size = max_size

    return inbox


def test_inbox_query_drains_messages():
    inbox = _load_inbox()

    inbox.put("hello")
    inbox.put("second")

    body = run(inbox.query(_turn()))

    assert body == "[Inbox]\nhello\n\nsecond"

    # Drained: the next query sees nothing.
    assert run(inbox.query(_turn())) is None


def test_inbox_hidden_for_subagents():
    inbox = _load_inbox()

    inbox.put("secret")

    assert run(inbox.query(_turn(depth=2))) is None

    # Still queued for the main agent.
    assert "secret" in run(inbox.query(_turn()))


def test_inbox_overflow_drops_oldest():
    inbox = _load_inbox(max_size=2)

    inbox.put("one")
    inbox.put("two")
    inbox.put("three")

    body = run(inbox.query(_turn()))

    assert "one" not in body
    assert "two" in body
    assert "three" in body


def test_inbox_ignores_empty_put():
    inbox = _load_inbox()

    inbox.put("")

    assert run(inbox.query(_turn())) is None

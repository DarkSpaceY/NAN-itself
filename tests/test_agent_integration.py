from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from nan_itself.agent.core import Agent
from nan_itself.agent.engine import StepEngine
from nan_itself.agent.model import Report
from nan_itself.agent.prompts import render_turn
from nan_itself.agent.verbs import SpawnVerb
from nan_itself.events import EventBus, sink
from nan_itself.modules.loading import import_module_class
from nan_itself.modules.model import DataSpace, Turn
from nan_itself.utils.llm import Message, ToolCall


# ============================================================================
# Async helpers
# ============================================================================


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset_sink():
    yield

    sink.reset()


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
    Skill runtime that exposes only the refresh/catalog semantics
    needed by the Agent and the skill verbs.
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


class EchoLLM:
    """
    LLM that answers plain text and retains every request, used
    for exercising the real StepEngine without tool calls.
    """

    model = "fake-model"
    provider = "fake-provider"

    def __init__(self):
        self.requests = []

    async def generate_complete(
        self,
        request,
    ):
        self.requests.append(request)

        return SimpleNamespace(
            content="reply",
            tool_calls=[],
            model="fake-model",
            usage=None,
            provider="fake-provider",
            finish_reason="stop",
        )


@dataclass
class CapturedExecution:
    turn: object

    @property
    def persona(self):
        return self.turn.persona

    @property
    def history(self):
        return self.turn.history


class CapturingEngine:
    """
    Replaces StepEngine while retaining the Agent's real
    turn-boundary behavior.

    This allows us to verify exactly what the Agent constructs and
    passes into the engine. The fake mirrors the engine's contract:
    each turn's Turn record carries the history snapshot as of its
    start and the assistant's outcome is written back structurally
    (reply); the Agent chains turns through it (no persistent
    history anywhere).
    """

    def __init__(self):
        self.executions: list[
            CapturedExecution
        ] = []

        self.on_execute = None

    async def step(
        self,
        turn,
    ):
        self.executions.append(
            CapturedExecution(
                turn=turn,
            )
        )

        callback = self.on_execute

        if callback is not None:
            outcome = callback(
                self,
                turn,
            )

            if asyncio.iscoroutine(
                outcome
            ):
                await outcome

        return replace(
            turn,
            reply="assistant-done",
        )

    def render_turn(
        self,
        turn,
    ):
        return render_turn(turn)


def make_agent(
    *,
    engine=None,
    modules=None,
    history_char_limit=100_000,
    autonomous_interval=0.0,
):
    return Agent(
        engine=engine or CapturingEngine(),
        modules=modules or FakeModules(),
        tools=FakeProviders(),
        skills=FakeSkills(),
        history_char_limit=history_char_limit,
        autonomous_interval=autonomous_interval,
    )


def make_child(
    parent,
    *,
    task,
):
    """
    A child Agent without its own loop task: the harvest tests only
    need the tree shape and the child's report slot.
    """
    return Agent(
        engine=parent.engine,
        modules=parent.modules,
        tools=parent.tools,
        skills=parent.skills,
        task=task,
        parent=parent,
    )


# ============================================================================
# Turn boundaries (Agent.run)
# ============================================================================


def test_run_assembles_turn_boundaries(
    tmp_path,
):
    (tmp_path / "persona.md").write_text(
        "You are NAN.",
        encoding="utf-8",
    )

    engine = CapturingEngine()

    modules = FakeModules()

    agent = make_agent(
        engine=engine,
        modules=modules,
    )

    result = run(agent.run())

    assert result.reply == "assistant-done"

    # One execution captured, one turn chained.
    assert len(engine.executions) == 1

    captured = engine.executions[0]

    assert captured.persona == "You are NAN."

    # First turn: no prior turn, nothing derived.
    assert captured.history == ()

    # The completed Turn rides out on run() and becomes the
    # single last_turn reference; nothing else is retained. Its
    # message run renders as the (empty) observation plus the
    # assistant reply.
    assert agent.last_turn is not None

    assert [
        m.content
        for m in render_turn(agent.last_turn)
    ] == ["", "assistant-done"]

    # Per-turn boundaries: refresh + snapshot exactly once.
    assert agent.skills.refresh_calls == 1
    assert agent.modules.snapshot_calls == 1

    # Root turn: depth 0, no task (input rides in the inbox).
    assert captured.turn.depth == 0
    assert captured.turn.task is None


def test_run_extends_history_across_turns():
    engine = CapturingEngine()

    agent = make_agent(
        engine=engine,
    )

    run(agent.run())
    run(agent.run())

    first, second = engine.executions

    # First turn starts from nothing; the second turn derives
    # its history snapshot from the first turn's record
    # (observation + assistant reply).
    assert first.history == ()

    assert [
        m.content for m in second.history
    ] == ["", "assistant-done"]


def test_agent_retains_history_char_limit_and_clears_over_it():
    # The retention policy lives on the Agent (the engine no
    # longer knows about it); it is applied at derivation time.
    engine = CapturingEngine()

    agent = make_agent(
        engine=engine,
        history_char_limit=5,
    )

    assert agent.history_char_limit == 5

    run(agent.run())

    # The first turn's messages exceed the limit, so the second
    # turn's derived snapshot is cleared to empty.
    run(agent.run())

    assert len(engine.executions) == 2

    assert (
        engine.executions[1].history
        == ()
    )


def test_finish_tool_hidden_for_main_agent_and_visible_for_subagents():
    engine = StepEngine(
        llm=FakeLLM(),
    )

    root_names = {
        tool.name
        for tool in engine._tool_definitions(
            depth=0
        )
    }

    child_names = {
        tool.name
        for tool in engine._tool_definitions(
            depth=2
        )
    }

    assert "finish" not in root_names

    assert "finish" in child_names


def test_step_sees_history_plus_the_turn_messages():
    llm = EchoLLM()

    engine = StepEngine(
        llm=llm,
    )

    history = (
        Message(role="user", content="old"),
        Message(role="assistant", content="last"),
    )

    turn = Turn(
        agent_hash="hash123",
        parent_hash=None,
        depth=0,
        task=None,
        world={"state": {"value": 1}},
        persona="p",
        history=history,
        # A bare-string report passes through unframed, which
        # keeps the observation text exactly "observation".
        reports=("observation",),
    )

    result = run(
        engine.step(turn)
    )

    # The turn's history snapshot became the model-visible
    # prefix: system + history + this turn's rendering.
    request = llm.requests[0]

    assert len(request.messages) == 4

    assert [
        m.content
        for m in request.messages[:3]
    ] == ["p", "old", "last"]

    # The completed Turn carries the same snapshot plus its
    # structured flow (reply) -- the full model input is
    # exactly build_messages(result).
    assert [
        m.content for m in result.history
    ] == ["old", "last"]

    assert [
        m.content for m in render_turn(result)
    ] == ["observation", "reply"]


# ============================================================================
# Child report harvesting (one level up, no inbox)
# ============================================================================


def _child_report(
    task: str,
) -> Report:
    return Report(
        agent_id="abc12345",
        task=task,
        status="completed",
        body=f"{task} result",
    )


def test_harvest_children_collects_and_mirrors_finished_reports():
    bus = EventBus()

    sink.attach(bus)

    agent = make_agent()

    report = _child_report("child task")

    finished = make_child(
        agent,
        task="child task",
    )

    finished.done = True
    finished.report = report

    running = make_child(
        agent,
        task="running task",
    )

    agent.children = [finished, running]

    reports = agent._harvest_children()

    # The finished child's report is collected and handed back
    # for folding into this agent's own next observation.
    assert reports == [report]

    # Harvested children are removed; running ones stay listed.
    assert agent.children == [running]

    events = bus.history()

    kinds = [event["t"] for event in events]

    assert "record_started" in kinds
    assert "record_detail" in kinds
    assert "record_done" in kinds

    started = events[0]

    assert started["content"]["kind"] == "agent"

    assert (
        started["content"]["name"]
        == "report · child task"
    )

    lines = [
        event["content"]["line"]
        for event in events
        if event["t"] == "record_detail"
    ]

    # The UI mirror reads the structured fields: no prompt
    # frame, just the report's identity and its body.
    assert lines == [
        "id: abc12345",
        "task: child task",
        "status: completed",
        "child task result",
    ]

    done = [
        event
        for event in events
        if event["t"] == "record_done"
    ][0]

    assert done["content"]["summary"] == "report"


def test_harvest_children_drops_finished_child_without_report():
    agent = make_agent()

    lost = make_child(
        agent,
        task="lost task",
    )

    lost.done = True
    lost.report = None

    agent.children = [lost]

    # Must not raise, and must not invent a report.
    assert agent._harvest_children() == []

    assert agent.children == []


# ============================================================================
# Child report delivery (one level up)
# ============================================================================


class ScriptedLLM:
    """
    LLM driven by each turn's observation (messages[-1] at the
    first model call of every turn):

        - the depth-2 grandchild finishes with its report
          (optionally gated, to simulate a slow child);
        - the depth-1 parent waits for the child report to reach
          its observation and then finishes (or finishes on
          schedule when finish_on_call is set).
    """

    model = "fake-model"
    provider = "fake-provider"

    def __init__(
        self,
        *,
        finish_on_call: int | None = None,
        grandchild_gate: asyncio.Event | None = None,
    ):
        self.requests = []
        self.finish_on_call = finish_on_call
        self.grandchild_gate = grandchild_gate
        self.parent_calls = 0
        self._call_ids = 0

    def _id(self) -> str:
        self._call_ids += 1

        return f"call-{self._call_ids}"

    def _finish(
        self,
        report: str,
    ):
        return SimpleNamespace(
            content=None,
            tool_calls=[
                ToolCall(
                    id=self._id(),
                    name="finish",
                    arguments={"report": report},
                ),
            ],
            model="fake-model",
            usage=None,
            provider="fake-provider",
            finish_reason="tool_calls",
        )

    @staticmethod
    def _text(text: str):
        return SimpleNamespace(
            content=text,
            tool_calls=[],
            model="fake-model",
            usage=None,
            provider="fake-provider",
            finish_reason="stop",
        )

    async def generate_complete(
        self,
        request,
    ):
        self.requests.append(request)

        # A real yield point: without it the all-inline fakes
        # would never let sibling tasks (the spawned grandchild)
        # run between this agent's turns.
        await asyncio.sleep(0)

        observation = (
            request.messages[-1].content or ""
        )

        # The parent's observation always carries its task; check
        # it FIRST, because a delivered child report quotes the
        # grandchild task verbatim in its header.
        if "parent task" in observation:
            self.parent_calls += 1

            if (
                self.finish_on_call is not None
                and self.parent_calls
                >= self.finish_on_call
            ):
                # Finish while the grandchild is still running:
                # unblock it only after the parent called
                # finish.
                if self.grandchild_gate is not None:
                    self.grandchild_gate.set()

                return self._finish("parent report")

            if "<subagent_report>" in observation:
                return self._finish("parent report")

            if self.parent_calls > 50:
                raise AssertionError(
                    "parent never received the child report"
                )

            return self._text("still working")

        if "grandchild task" in observation:
            if self.grandchild_gate is not None:
                await self.grandchild_gate.wait()

            return self._finish(
                "grandchild report"
            )

        raise AssertionError(
            "unrouted observation: {!r:.200}".format(
                observation
            )
        )


async def _run_spawn_scenario(
    llm,
):
    """
    Spawn through the real SpawnVerb: a real parent Agent at depth 1
    spawns its own grandchild (depth 2, task 'grandchild task')
    through the verb, runs its homogeneous loop against the real
    StepEngine, and finishes once the child report reached its
    observation.

    Returns the settled parent Agent.
    """
    modules = FakeModules()

    engine = StepEngine(
        llm=llm,
    )

    root = Agent(
        engine=engine,
        modules=modules,
        tools=FakeProviders(),
        skills=FakeSkills(),
    )

    parent = root.spawn("parent task")

    call = SimpleNamespace(
        id="call-0",
        name="spawn",
        arguments={"task": "grandchild task"},
    )

    await SpawnVerb().execute(
        call=call,
        agent=parent,
    )

    try:
        await asyncio.wait_for(
            parent._task,
            timeout=5.0,
        )

    finally:
        if not parent._task.done():
            parent._task.cancel()

            await asyncio.gather(
                parent._task,
                return_exceptions=True,
            )

    return parent


def test_child_report_reaches_parent_observation():
    llm = ScriptedLLM()

    parent = run(
        _run_spawn_scenario(
            llm,
        )
    )

    # The parent finished by synthesizing the report that was
    # harvested into its observation.
    assert parent.done

    assert parent.report.status == "completed"

    assert "parent report" in parent.report.body

    # Some parent turn's observation carried the <subagent_report>
    # block from the grandchild.
    observations = [
        request.messages[-1].content or ""
        for request in llm.requests
    ]

    assert any(
        "<subagent_report>" in observation
        and "grandchild report" in observation
        for observation in observations
    )


def test_early_finish_drops_running_children_reports():
    gate = asyncio.Event()

    llm = ScriptedLLM(
        finish_on_call=2,
        grandchild_gate=gate,
    )

    parent = run(
        _run_spawn_scenario(
            llm,
        )
    )

    # The parent finished before the grandchild reported: the
    # late report is dropped, not folded into the finish report.
    assert parent.done

    assert "parent report" in parent.report.body

    assert (
        "grandchild report"
        not in parent.report.body
    )


# ============================================================================
# The autonomous loop (Agent.loop)
# ============================================================================


def test_loop_runs_turns_until_stop():
    engine = CapturingEngine()

    agent = make_agent(
        engine=engine,
    )

    def stop_after_second(
        engine_self,
        turn,
    ):
        if len(engine_self.executions) >= 2:
            agent.request_stop()

    engine.on_execute = stop_after_second

    run(
        asyncio.wait_for(
            agent.loop(),
            timeout=1.0,
        )
    )

    assert len(engine.executions) == 2
    assert agent.cycles == 2
    assert agent.stopping


def test_loop_backoff_retries_failed_turn():
    engine = CapturingEngine()

    agent = make_agent(
        engine=engine,
    )

    agent.backoff = (0.0,)

    calls = 0

    def fail_once(
        engine_self,
        turn,
    ):
        nonlocal calls

        calls += 1

        if calls == 1:
            raise RuntimeError("boom")

        agent.request_stop()

    engine.on_execute = fail_once

    run(
        asyncio.wait_for(
            agent.loop(),
            timeout=1.0,
        )
    )

    # The failed turn retried immediately (zero backoff) and the
    # retry succeeded.
    assert calls == 2
    assert agent.cycles == 1


def test_stop_takes_effect_at_the_next_turn_boundary():
    """
    A stop request is cooperative: an in-flight turn always runs to
    completion and the loop exits at the next boundary. There is no
    grace window and the turn is never cancelled mid-flight.
    """
    engine = CapturingEngine()

    agent = make_agent(
        engine=engine,
    )

    observed = {}

    async def stop_mid_turn(
        engine_self,
        turn,
    ):
        agent.request_stop()

        # Still inside the in-flight turn: it keeps running.
        await asyncio.sleep(0)

        observed["in_flight"] = True

    engine.on_execute = stop_mid_turn

    run(
        asyncio.wait_for(
            agent.loop(),
            timeout=1.0,
        )
    )

    # The in-flight turn ran to completion despite the stop
    # request and counted as a cycle; then the loop exited.
    assert observed["in_flight"]

    assert len(engine.executions) == 1
    assert agent.cycles == 1
    assert agent.stopping


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

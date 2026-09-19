from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from nan_itself.agent.core import CoreAgent
from nan_itself.agent.engine import StepEngine
from nan_itself.agent.runtime import AgentRuntime
from nan_itself.agent.verbs import ExecutionState, SpawnVerb
from nan_itself.modules.loading import import_module_class
from nan_itself.modules.model import DataSpace, Turn
from nan_itself.utils.llm import Message, ToolCall


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
    context: object
    persona: str
    last_turn: object | None
    turn: object


class CapturingEngine:
    """
    Replaces StepEngine while retaining CoreAgent's real turn-boundary
    behavior.

    This allows us to verify exactly what CoreAgent constructs and
    passes into the execution engine. The fake mirrors the engine's
    snapshot-derivation contract: each turn's Turn record carries
    the history snapshot as of its start, and CoreAgent chains
    turns through it (no persistent history anywhere).
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
        last_turn=None,
        report_sink=None,
        sink=None,
    ):
        # Mirror the real StepEngine contract: derive this
        # turn's history snapshot from the last turn.
        prior: tuple = ()

        if last_turn is not None:
            prior = (
                last_turn.history
                + last_turn.messages
            )

        turn = SimpleNamespace(
            history=prior,
            messages=(
                SimpleNamespace(
                    content="assistant-done",
                ),
            ),
        )

        self.executions.append(
            CapturedExecution(
                context=context,
                persona=persona,
                last_turn=last_turn,
                turn=turn,
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
            turn=turn,
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

    # One execution captured, one turn chained.
    assert len(engine.executions) == 1

    captured = engine.executions[0]

    assert captured.persona == "You are NAN."

    # First turn: no prior turn, nothing derived.
    assert captured.last_turn is None

    # The completed Turn rides on the result and becomes the
    # single last_turn reference; nothing else is retained.
    assert core.last_turn is captured.turn

    assert [
        m.content
        for m in captured.turn.messages
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

    # First turn starts from nothing; the second turn derives
    # its history snapshot from the first turn's record.
    assert first.last_turn is None

    assert second.last_turn is first.turn

    assert [
        m.content for m in second.turn.history
    ] == ["assistant-done"]


def test_core_agent_forwards_history_char_limit_to_engine():
    # The retention policy lives in the StepEngine; CoreAgent
    # only forwards the configured limit at construction time.
    core = make_agent(history_char_limit=5)

    assert core.engine.history_char_limit == 5


def test_finish_tool_hidden_for_main_agent_and_visible_for_subagents():
    engine = StepEngine(
        llm=FakeLLM(),
        modules=FakeModules(),
        tools=FakeProviders(),
        skills=FakeSkills(),
        agent_runtime=SimpleNamespace(),
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


def test_engine_derives_history_snapshot_from_last_turn():
    llm = EchoLLM()

    engine = StepEngine(
        llm=llm,
        modules=FakeModules(),
        tools=FakeProviders(),
        skills=FakeSkills(),
        agent_runtime=SimpleNamespace(),
    )

    context = SimpleNamespace(
        agent_hash="hash123",
        parent_hash=None,
        depth=0,
        task=None,
        world={"state": {"value": 1}},
    )

    last_turn = SimpleNamespace(
        history=(
            Message(role="user", content="old"),
        ),
        messages=(
            Message(role="assistant", content="last"),
        ),
    )

    result = run(
        engine.execute(
            context=context,
            persona="p",
            last_turn=last_turn,
        )
    )

    # The derived snapshot became the model-visible prefix:
    # system + history + this turn's observation.
    request = llm.requests[0]

    assert len(request.messages) == 4

    # The completed Turn carries the same snapshot plus this
    # turn's messages (observation + assistant reply) -- the
    # full model input is history + messages.
    assert [
        m.content for m in result.turn.history
    ] == ["old", "last"]

    assert [
        m.content for m in result.turn.messages
    ] == ["", "reply"]


def test_engine_clears_history_over_char_limit():
    llm = EchoLLM()

    engine = StepEngine(
        llm=llm,
        modules=FakeModules(),
        tools=FakeProviders(),
        skills=FakeSkills(),
        agent_runtime=SimpleNamespace(),
        history_char_limit=5,
    )

    context = SimpleNamespace(
        agent_hash="hash123",
        parent_hash=None,
        depth=0,
        task=None,
        world={"state": {"value": 1}},
    )

    last_turn = SimpleNamespace(
        history=(
            Message(role="user", content="x" * 100),
        ),
        messages=(),
    )

    result = run(
        engine.execute(
            context=context,
            persona="p",
            last_turn=last_turn,
        )
    )

    # The oversized snapshot was cleared at derivation time:
    # only the system message and this turn's observation
    # reached the model.
    assert len(llm.requests) == 1

    assert len(llm.requests[0].messages) == 2

    # The chain restarts fresh from the completed turn; the
    # broken-off turns stay on record elsewhere.
    assert result.turn.history == ()

    assert [
        m.content for m in result.turn.messages
    ] == ["", "reply"]


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
# Child report delivery (one level up)
# ============================================================================


class ScriptedLLM:
    """
    LLM driven by each turn's observation (messages[-1] at the
    first model call of every turn):

        - the depth-2 grandchild finishes with its report
          (optionally gated, to simulate a slow child);
        - the depth-1 parent spawns on its first turn, then
          finishes once the child report reached its observation
          (or immediately when finish_on_call is set).
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

    def _spawn(self, task: str):
        return SimpleNamespace(
            content=None,
            tool_calls=[
                ToolCall(
                    id=self._id(),
                    name="spawn",
                    arguments={"task": task},
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
        # would never let sibling tasks (the spawned grandchild,
        # the report archive) run between this agent's turns.
        await asyncio.sleep(0)

        observation = (
            request.messages[-1].content or ""
        )

        # The parent's observation always carries its task; check
        # it FIRST, because a delivered child report quotes the
        # grandchild task verbatim in its header.
        if "parent task" in observation:
            self.parent_calls += 1

            if self.parent_calls == 1:
                return self._spawn("grandchild task")

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

            if "[Subagent Report]" in observation:
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
    Spawn through the real SpawnVerb so the real worker loop runs:
    the spawned agent (depth 2, task 'parent task') spawns its own
    grandchild (depth 3, task 'grandchild task'), then finishes.
    Returns the spawned agent's settled result.
    """
    engine = StepEngine(
        llm=llm,
        modules=FakeModules(),
        tools=FakeProviders(),
        skills=FakeSkills(),
        agent_runtime=AgentRuntime(
            max_subagent_depth=3
        ),
    )

    state = ExecutionState(persona="p")

    context = SimpleNamespace(
        agent_hash="parent-hash",
        parent_hash="root-hash",
        depth=1,
        task="parent task",
        world={},
    )

    call = SimpleNamespace(
        id="call-0",
        name="spawn",
        arguments={"task": "parent task"},
    )

    await SpawnVerb().execute(
        call=call,
        context=context,
        state=state,
        engine=engine,
    )

    result = await state.children[0].handle.wait()

    await engine.agent_runtime.shutdown()

    return result


def test_child_report_reaches_parent_observation():
    llm = ScriptedLLM()

    result = run(_run_spawn_scenario(llm))

    # The parent finished by synthesizing the report that was
    # delivered into its observation.
    assert result.finished is True

    assert result.content == "parent report"

    # Some parent turn's observation carried the [Subagent
    # Report] block from the grandchild.
    observations = [
        request.messages[-1].content or ""
        for request in llm.requests
    ]

    assert any(
        "[Subagent Report]" in observation
        and "grandchild report" in observation
        for observation in observations
    )


def test_early_finish_waits_out_running_children():
    gate = asyncio.Event()

    llm = ScriptedLLM(
        finish_on_call=2,
        grandchild_gate=gate,
    )

    result = run(_run_spawn_scenario(llm))

    # The parent finished before the grandchild; the worker
    # waited the grandchild out and its report rode out with the
    # finish report instead of being lost.
    assert result.finished is True

    assert (result.content or "").startswith(
        "parent report"
    )

    assert "[Subagent Report]" in (
        result.content or ""
    )

    assert "grandchild report" in (
        result.content or ""
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

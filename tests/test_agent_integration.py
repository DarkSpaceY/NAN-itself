
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from nan_itself.agent.core import CoreAgent
from nan_itself.agent.loop import (
    AgentLoop,
    Inbox,
)
from nan_itself.modules.model import DataSpace


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

    def snapshot(self):
        self.snapshot_calls += 1

        return {
            "state": self.space.snapshot(),
        }

    async def query_snapshot(
        self,
        turn,
        snapshot,
        **kwargs,
    ):
        return []

    def deliver_turn(
        self,
        record,
    ):
        pass


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
    user_input: str
    persona: str


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

    async def execute(
        self,
        *,
        context,
        user_input,
        persona,
        seed_reports=None,
        report_sink=None,
        sink=None,
    ):
        self.executions.append(
            CapturedExecution(
                context=context,
                user_input=user_input,
                persona=persona,
            )
        )

        callback = self.on_execute

        if callback is not None:
            await callback(
                context
            )

        return SimpleNamespace(
            content="done"
        )


# ============================================================================
# AgentLoop doubles
# ============================================================================


class RecordingAgent:
    """
    Agent used to inspect AgentLoop's input lifecycle.

    The first turn blocks while the test injects an Inbox message.
    The second turn sees the message and requests shutdown.
    """

    def __init__(self):
        self.calls: list[str] = []

        self.started = asyncio.Event()
        self.release = asyncio.Event()

        self.loop: AgentLoop | None = None

    async def run(
        self,
        user_input: str,
    ):
        self.calls.append(
            user_input
        )

        if len(self.calls) == 1:
            self.started.set()

            await self.release.wait()

        elif len(self.calls) == 2:
            assert self.loop is not None

            self.loop.request_stop()

        return SimpleNamespace(
            content="done"
        )

    def has_pending_reports(self):
        return False


class RetryAgent:
    """
    First execution fails.

    The second execution succeeds and requests shutdown.

    The test verifies that the exact input from the failed turn
    survives the retry boundary.
    """

    def __init__(self):
        self.calls: list[str] = []

        self.loop: AgentLoop | None = None

    async def run(
        self,
        user_input: str,
    ):
        self.calls.append(
            user_input
        )

        if len(self.calls) == 1:
            raise RuntimeError(
                "intentional test failure"
            )

        assert self.loop is not None

        self.loop.request_stop()

        return SimpleNamespace(
            content="recovered"
        )

    def has_pending_reports(self):
        return False


# ============================================================================
# CoreAgent construction
# ============================================================================


def make_core(
    modules: FakeModules,
    skills: FakeSkills,
    *,
    persona_source,
):
    core = CoreAgent(
        llm=FakeLLM(),
        modules=modules,
        providers=FakeProviders(),
        skills=skills,
        persona_source=persona_source,
    )

    engine = CapturingEngine()

    core.engine = engine

    return core, engine


# ============================================================================
# AgentLoop / Inbox semantics
# ============================================================================


def test_agent_loop_does_not_consume_inbox_during_active_turn():
    """
    An Inbox message arriving while a turn is executing belongs to
    the next turn.

    Timeline:

        turn 1 starts with ""
             |
             +---- inbox receives "hello"
             |
        turn 1 continues
             |
        turn 1 finishes
             |
        turn 2 receives "hello"
    """

    async def scenario():
        agent = RecordingAgent()

        inbox = Inbox(
            maxsize=8
        )

        loop = AgentLoop(
            agent,
            inbox,
            backoff=(0.0,),
        )

        agent.loop = loop

        loop_task = asyncio.create_task(
            loop.run_forever()
        )

        try:
            await asyncio.wait_for(
                agent.started.wait(),
                timeout=1,
            )

            # The first turn is already executing.
            assert agent.calls == [
                ""
            ]

            inbox.put(
                "hello"
            )

            # Input exists, but the current turn is not cancelled.
            assert not loop.stopping

            agent.release.set()

            await asyncio.wait_for(
                loop_task,
                timeout=1,
            )

            assert agent.calls == [
                "",
                "hello",
            ]

        finally:
            if not loop_task.done():
                loop.request_stop()

                await loop_task

    run(
        scenario()
    )


def test_agent_loop_retry_preserves_failed_turn_input():
    """
    Input belonging to a failed turn must survive and be retried.
    """

    async def scenario():
        agent = RetryAgent()

        inbox = Inbox(
            maxsize=8
        )

        loop = AgentLoop(
            agent,
            inbox,
            backoff=(0.0,),
        )

        agent.loop = loop

        inbox.put(
            "important task"
        )

        await loop.run_forever()

        assert agent.calls == [
            "important task",
            "important task",
        ]

        assert loop.cycles == 1
        assert loop.stopping

    run(
        scenario()
    )


def test_agent_loop_empty_inbox_is_valid_autonomous_turn():
    """
    Empty Inbox must not prevent an autonomous turn.

    The first input is therefore the empty string.
    """

    async def scenario():
        agent = RecordingAgent()

        inbox = Inbox()

        loop = AgentLoop(
            agent,
            inbox,
            backoff=(0.0,),
        )

        agent.loop = loop

        loop_task = asyncio.create_task(
            loop.run_forever()
        )

        try:
            await asyncio.wait_for(
                agent.started.wait(),
                timeout=1,
            )

            assert agent.calls == [
                ""
            ]

            agent.release.set()

            await asyncio.wait_for(
                loop_task,
                timeout=1,
            )

        finally:
            if not loop_task.done():
                loop.request_stop()

                await loop_task

    run(
        scenario()
    )


def test_agent_loop_stop_request_allows_current_turn_gracefully():
    """
    request_stop() does not immediately cancel the current turn.

    A turn that finishes inside the shutdown grace window is counted
    as completed.
    """

    class SlowAgent:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

            self.loop: AgentLoop | None = None

        async def run(
            self,
            user_input,
        ):
            self.started.set()

            await self.release.wait()

            return SimpleNamespace(
                content="completed"
            )

        def has_pending_reports(self):
            return False

    async def scenario():
        agent = SlowAgent()

        inbox = Inbox()

        loop = AgentLoop(
            agent,
            inbox,
            turn_grace=1.0,
        )

        agent.loop = loop

        task = asyncio.create_task(
            loop.run_forever()
        )

        try:
            await asyncio.wait_for(
                agent.started.wait(),
                timeout=1,
            )

            loop.request_stop()

            # Current turn is still allowed to finish.
            assert not task.done()

            agent.release.set()

            await asyncio.wait_for(
                task,
                timeout=1,
            )

            assert loop.cycles == 1

        finally:
            if not task.done():
                loop.request_stop()

                await task

    run(
        scenario()
    )


# ============================================================================
# CoreAgent turn boundary
# ============================================================================


def test_core_agent_captures_one_world_snapshot_per_turn():
    async def scenario():
        modules = FakeModules()
        skills = FakeSkills()

        core, engine = make_core(
            modules,
            skills,
            persona_source=lambda: "persona",
        )

        await core.run(
            "first"
        )

        assert (
            modules.snapshot_calls
            == 1
        )

        assert (
            len(engine.executions)
            == 1
        )

        assert (
            engine.executions[0]
            .context.world["state"]["value"]
            == 1
        )

        assert (
            skills.refresh_calls
            == 1
        )

    run(
        scenario()
    )


def test_core_agent_new_turn_observes_new_module_state():
    """
    Live DataSpace may change between turns.

    Previous turn keeps the old world.
    Next turn receives a new detached snapshot.
    """

    async def scenario():
        modules = FakeModules()
        skills = FakeSkills()

        core, engine = make_core(
            modules,
            skills,
            persona_source=lambda: "persona",
        )

        await core.run(
            "first"
        )

        first_world = (
            engine.executions[0]
            .context.world
        )

        assert (
            first_world["state"]["value"]
            == 1
        )

        modules.space.publish(
            {
                "value": 2,
            }
        )

        await core.run(
            "second"
        )

        second_world = (
            engine.executions[1]
            .context.world
        )

        assert (
            first_world["state"]["value"]
            == 1
        )

        assert (
            second_world["state"]["value"]
            == 2
        )

        assert (
            first_world
            is not second_world
        )

        assert (
            modules.snapshot_calls
            == 2
        )

    run(
        scenario()
    )


def test_core_agent_world_remains_stable_during_live_dataspace_update():
    """
    A live Module update during an Agent turn must not rewrite that
    turn's world.
    """

    async def scenario():
        modules = FakeModules()
        skills = FakeSkills()

        core, engine = make_core(
            modules,
            skills,
            persona_source=lambda: "persona",
        )

        observed = {}

        async def on_execute(
            context,
        ):
            observed["before"] = (
                context.world[
                    "state"
                ]["value"]
            )

            modules.space.publish(
                {
                    "value": 99,
                }
            )

            observed["during"] = (
                context.world[
                    "state"
                ]["value"]
            )

            observed["live"] = (
                modules.space.snapshot()[
                    "value"
                ]
            )

        engine.on_execute = (
            on_execute
        )

        await core.run(
            "turn"
        )

        assert (
            observed["before"]
            == 1
        )

        assert (
            observed["during"]
            == 1
        )

        assert (
            observed["live"]
            == 99
        )

        assert (
            engine.executions[0]
            .context.world[
                "state"
            ]["value"]
            == 1
        )

    run(
        scenario()
    )


# ============================================================================
# CoreAgent turn refresh semantics
# ============================================================================


def test_core_agent_refreshes_skills_at_every_turn_boundary():
    async def scenario():
        modules = FakeModules()
        skills = FakeSkills()

        core, engine = make_core(
            modules,
            skills,
            persona_source=lambda: "persona",
        )

        await core.run(
            "first"
        )

        await core.run(
            "second"
        )

        await core.run(
            "third"
        )

        assert (
            skills.refresh_calls
            == 3
        )

        assert (
            modules.snapshot_calls
            == 3
        )

    run(
        scenario()
    )


def test_core_agent_reloads_persona_at_every_turn_boundary():
    persona = {
        "value": "persona-v1"
    }

    modules = FakeModules()
    skills = FakeSkills()

    core, engine = make_core(
        modules,
        skills,
        persona_source=lambda: persona[
            "value"
        ],
    )

    async def scenario():
        await core.run(
            "first"
        )

        persona[
            "value"
        ] = "persona-v2"

        await core.run(
            "second"
        )

        persona[
            "value"
        ] = "persona-v3"

        await core.run(
            "third"
        )

        assert [
            execution.persona
            for execution
            in engine.executions
        ] == [
            "persona-v1",
            "persona-v2",
            "persona-v3",
        ]

    run(
        scenario()
    )


# ============================================================================
# CoreAgent + AgentLoop integration
# ============================================================================


def test_agent_loop_and_core_agent_share_turn_boundaries():
    """
    Integration-level check:

        AgentLoop
            ->
        CoreAgent.run()
            ->
        exactly one snapshot

    Two AgentLoop turns therefore produce two DataSpace snapshots.
    """

    class LoopAgent:
        def __init__(
            self,
            core,
        ):
            self.core = core
            self.calls = []

            self.loop: AgentLoop | None = None

        async def run(
            self,
            user_input,
        ):
            self.calls.append(
                user_input
            )

            await self.core.run(
                user_input
            )

            if len(self.calls) >= 2:
                assert self.loop is not None
                self.loop.request_stop()

            return SimpleNamespace(
                content="done"
            )

        def has_pending_reports(self):
            return self.core.has_pending_reports()

    async def scenario():
        modules = FakeModules()
        skills = FakeSkills()

        core, engine = make_core(
            modules,
            skills,
            persona_source=lambda: "persona",
        )

        agent = LoopAgent(
            core
        )

        inbox = Inbox(
            maxsize=8
        )

        loop = AgentLoop(
            agent,
            inbox,
            backoff=(0.0,),
        )

        agent.loop = loop

        # First autonomous turn.
        loop_task = asyncio.create_task(
            loop.run_forever()
        )

        try:
            await asyncio.wait_for(
                loop_task,
                timeout=1,
            )

        finally:
            if not loop_task.done():
                loop.request_stop()

                await loop_task

        assert agent.calls == [
            "",
            "",
        ]

        assert (
            modules.snapshot_calls
            == 2
        )

        assert (
            len(engine.executions)
            == 2
        )

        assert (
            engine.executions[0]
            .context.world["state"]["value"]
            == 1
        )

        assert (
            engine.executions[1]
            .context.world["state"]["value"]
            == 1
        )

        assert (
            skills.refresh_calls
            == 2
        )

    run(
        scenario()
    )


def test_agent_loop_delivers_new_input_on_next_turn_after_live_update():
    """
    A message arriving during an active turn becomes input for the
    following turn, where CoreAgent captures the then-current world.
    """

    class BlockingCore:
        def __init__(
            self,
            modules,
        ):
            self.modules = modules
            self.loop: AgentLoop | None = None

            self.started = asyncio.Event()
            self.release = asyncio.Event()

            self.turns = []

        async def run(
            self,
            user_input,
        ):
            world = (
                self.modules.snapshot()
            )

            self.turns.append(
                (
                    user_input,
                    world,
                )
            )

            if len(self.turns) == 1:
                self.started.set()

                await self.release.wait()

            else:
                assert self.loop is not None
                self.loop.request_stop()

            return SimpleNamespace(
                content="done"
            )

        def has_pending_reports(self):
            return False

    async def scenario():
        modules = FakeModules()

        core = BlockingCore(
            modules
        )

        inbox = Inbox(
            maxsize=8
        )

        loop = AgentLoop(
            core,
            inbox,
            backoff=(0.0,),
        )

        core.loop = loop

        task = asyncio.create_task(
            loop.run_forever()
        )

        try:
            await asyncio.wait_for(
                core.started.wait(),
                timeout=1,
            )

            # Turn 1 has already captured world v1.
            assert (
                core.turns[0][1][
                    "state"
                ]["value"]
                == 1
            )

            # Live state changes during turn 1.
            modules.space.publish(
                {
                    "value": 2,
                }
            )

            # New input also arrives during turn 1.
            inbox.put(
                "new input"
            )

            core.release.set()

            await asyncio.wait_for(
                task,
                timeout=1,
            )

        finally:
            if not task.done():
                loop.request_stop()

                await task

        assert (
            core.turns[0][0]
            == ""
        )

        assert (
            core.turns[1][0]
            == "new input"
        )

        # Turn 1 observed v1.
        assert (
            core.turns[0][1][
                "state"
            ]["value"]
            == 1
        )

        # Turn 2 observed the newer live state.
        assert (
            core.turns[1][1][
                "state"
            ]["value"]
            == 2
        )

    run(
        scenario()
    )
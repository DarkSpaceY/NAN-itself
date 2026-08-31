
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from nan_itself.agent.core import CoreAgent
from nan_itself.agent.runtime import AgentRuntime
from nan_itself.modules.model import DataSpace


# ============================================================================
# Async helper
# ============================================================================


def run(coro):
    return asyncio.run(coro)


# ============================================================================
# Fake Module runtime
# ============================================================================


class FakeModules:
    """
    Small Module facade used to model live DataSpace state.

    snapshot() returns a detached world for one Agent turn.
    """

    def __init__(self):
        self.space = DataSpace(
            owner="state",
            initial={
                "value": 1,
            },
        )

        self.snapshot_calls = 0
        self.query_calls = 0

        self.snapshots = []

    def snapshot(self):
        self.snapshot_calls += 1

        snapshot = {
            "state": self.space.snapshot(),
        }

        self.snapshots.append(
            snapshot
        )

        return snapshot

    async def query_snapshot(
        self,
        turn,
        snapshot,
        **kwargs,
    ):
        self.query_calls += 1

        return []

    def deliver_turn(
        self,
        record,
    ):
        pass


# ============================================================================
# Fake Skills / Providers / LLM
# ============================================================================


class FakeSkills:
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


# ============================================================================
# Turn consistency fixtures
# ============================================================================


@dataclass
class CapturedExecution:
    context: object | None = None
    user_input: str | None = None
    persona: str | None = None


class CapturingEngine:
    """
    Replaces StepEngine inside CoreAgent.

    This lets tests inspect the exact AgentContext created at
    each turn boundary without invoking an LLM.
    """

    def __init__(self):
        self.executions = []

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

        if self.on_execute is not None:
            await self.on_execute(
                context
            )

        return SimpleNamespace(
            content="done"
        )


def make_core(
    modules,
    *,
    persona="test-persona",
):
    skills = FakeSkills()

    providers = FakeProviders()

    core = CoreAgent(
        llm=FakeLLM(),
        modules=modules,
        providers=providers,
        skills=skills,
        persona_source=lambda: persona,
    )

    engine = CapturingEngine()

    core.engine = engine

    return (
        core,
        engine,
        skills,
    )


# ============================================================================
# DataSpace semantics
# ============================================================================


def test_dataspace_snapshot_is_detached():
    space = DataSpace(
        owner="state",
        initial={
            "nested": {
                "value": 1,
            }
        },
    )

    snapshot = space.snapshot()

    snapshot["nested"]["value"] = 999

    assert (
        space.snapshot()["nested"]["value"]
        == 1
    )


def test_dataspace_revision_changes_when_published():
    space = DataSpace(
        owner="state",
    )

    assert space.revision == 0

    space.publish(
        {
            "value": 1,
        }
    )

    assert space.revision == 1

    space.publish(
        {
            "value": 2,
        }
    )

    assert space.revision == 2

    assert (
        space.snapshot()["value"]
        == 2
    )


# ============================================================================
# CoreAgent turn snapshot
# ============================================================================


def test_core_captures_exactly_one_world_snapshot_per_turn():
    """
    CoreAgent is responsible for the turn boundary:

        skills.refresh()
        persona_source()
        modules.snapshot()
        engine.execute()

    The fake CapturingEngine deliberately bypasses StepEngine, so
    modules.query_snapshot() is NOT expected to run here.
    """

    async def scenario():
        modules = FakeModules()

        core, engine, skills = make_core(
            modules
        )

        await core.run(
            "hello"
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
            modules.query_calls
            == 0
        )

        assert (
            skills.refresh_calls
            == 1
        )

    run(
        scenario()
    )


def test_each_main_turn_gets_a_new_world_snapshot():
    async def scenario():
        modules = FakeModules()

        core, engine, skills = make_core(
            modules
        )

        await core.run(
            "turn-1"
        )

        first_context = (
            engine.executions[0].context
        )

        assert (
            first_context.world[
                "state"
            ]["value"]
            == 1
        )

        modules.space.publish(
            {
                "value": 2,
            }
        )

        await core.run(
            "turn-2"
        )

        second_context = (
            engine.executions[1].context
        )

        assert (
            first_context.world[
                "state"
            ]["value"]
            == 1
        )

        assert (
            second_context.world[
                "state"
            ]["value"]
            == 2
        )

        assert (
            first_context.world
            is not second_context.world
        )

        assert (
            modules.snapshot_calls
            == 2
        )

        assert (
            skills.refresh_calls
            == 2
        )

    run(
        scenario()
    )


# ============================================================================
# Mid-turn DataSpace mutation
# ============================================================================


def test_world_snapshot_remains_stable_when_dataspace_changes_mid_turn():
    async def scenario():
        modules = FakeModules()

        core, engine, skills = make_core(
            modules
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

            # Mutate the live Module DataSpace while the Agent turn
            # is already executing.
            modules.space.publish(
                {
                    "value": 2,
                }
            )

            observed["after"] = (
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
            "mid-turn"
        )

        assert (
            observed["before"]
            == 1
        )

        # The current turn continues to observe the original snapshot.
        assert (
            observed["after"]
            == 1
        )

        # Live DataSpace has already moved forward.
        assert (
            observed["live"]
            == 2
        )

        context = (
            engine.executions[0].context
        )

        assert (
            context.world[
                "state"
            ]["value"]
            == 1
        )

    run(
        scenario()
    )


# ============================================================================
# Subagent dispatch-tree world identity
# ============================================================================


def test_subagents_share_exact_same_world_object():
    async def scenario():
        runtime = AgentRuntime(
            max_subagent_depth=3
        )

        source_world = {
            "state": {
                "value": 1,
            }
        }

        root = runtime.create_root(
            world=source_world,
            task="root",
        )

        observed = {}

        async def grandchild_worker(
            context,
        ):
            observed["grandchild"] = context
            return "grandchild"

        async def child_worker(
            context,
        ):
            observed["child"] = context

            handle = runtime.dispatch(
                context,
                task="grandchild",
                worker=grandchild_worker,
            )

            return await handle.wait()

        child_handle = runtime.dispatch(
            root,
            task="child",
            worker=child_worker,
        )

        result = (
            await child_handle.wait()
        )

        assert (
            result
            == "grandchild"
        )

        child = observed["child"]
        grandchild = (
            observed["grandchild"]
        )

        # Root creates one frozen world mapping.
        # Every descendant reuses exactly that mapping.
        assert (
            child.world
            is root.world
        )

        assert (
            grandchild.world
            is root.world
        )

        # The runtime-created mapping is not the original source mapping.
        assert (
            root.world
            is not source_world
        )

    run(
        scenario()
    )


def test_subagent_does_not_recapture_live_module_state():
    async def scenario():
        modules = FakeModules()

        runtime = AgentRuntime(
            max_subagent_depth=3
        )

        root = runtime.create_root(
            world=modules.snapshot(),
            task="root",
        )

        observed = {}

        async def child_worker(
            context,
        ):
            observed["child"] = context

            # Change live Module state after child creation.
            modules.space.publish(
                {
                    "value": 99,
                }
            )

            return "done"

        handle = runtime.dispatch(
            root,
            task="child",
            worker=child_worker,
        )

        assert (
            await handle.wait()
            == "done"
        )

        child = (
            observed["child"]
        )

        assert (
            child.world[
                "state"
            ]["value"]
            == 1
        )

        assert (
            modules.space.snapshot()[
                "value"
            ]
            == 99
        )

    run(
        scenario()
    )


# ============================================================================
# Root / child identity
# ============================================================================


def test_child_has_distinct_identity_but_same_world():
    async def scenario():
        runtime = AgentRuntime()

        root = runtime.create_root(
            world={
                "value": 1,
            }
        )

        observed = {}

        async def worker(
            context,
        ):
            observed["child"] = context
            return "done"

        handle = runtime.dispatch(
            root,
            task="child",
            worker=worker,
        )

        await handle.wait()

        child = (
            observed["child"]
        )

        assert (
            child.agent_hash
            != root.agent_hash
        )

        assert (
            child.parent_hash
            == root.agent_hash
        )

        assert (
            child.depth
            == 1
        )

        assert (
            child.world
            is root.world
        )

    run(
        scenario()
    )


# ============================================================================
# Turn generation boundary
# ============================================================================


def test_old_turn_world_is_not_rewritten_by_next_turn():
    async def scenario():
        modules = FakeModules()

        core, engine, skills = make_core(
            modules
        )

        await core.run(
            "first"
        )

        first = (
            engine.executions[0].context
        )

        modules.space.publish(
            {
                "value": 2,
            }
        )

        await core.run(
            "second"
        )

        second = (
            engine.executions[1].context
        )

        modules.space.publish(
            {
                "value": 3,
            }
        )

        assert (
            first.world[
                "state"
            ]["value"]
            == 1
        )

        assert (
            second.world[
                "state"
            ]["value"]
            == 2
        )

        assert (
            modules.space.snapshot()[
                "value"
            ]
            == 3
        )

    run(
        scenario()
    )


def test_turn_snapshot_is_shared_across_recursive_dispatch_tree():
    async def scenario():
        runtime = AgentRuntime(
            max_subagent_depth=3
        )

        source_world = {
            "state": {
                "value": 42,
            }
        }

        root = runtime.create_root(
            world=source_world
        )

        observed = []

        async def level_two(
            context,
        ):
            observed.append(
                context.world
            )

            return "level-two"

        async def level_one(
            context,
        ):
            observed.append(
                context.world
            )

            child = runtime.dispatch(
                context,
                task="level-two",
                worker=level_two,
            )

            return await child.wait()

        level_one_handle = (
            runtime.dispatch(
                root,
                task="level-one",
                worker=level_one,
            )
        )

        result = await level_one_handle.wait()

        assert (
            result
            == "level-two"
        )

        assert (
            len(observed)
            == 2
        )

        # The source mapping is copied into a frozen outer mapping
        # when the root is created.
        assert (
            root.world
            is not source_world
        )

        # Every descendant shares the root's turn world.
        assert (
            observed[0]
            is root.world
        )

        assert (
            observed[1]
            is root.world
        )

    run(
        scenario()
    )


# ============================================================================
# Context immutability / execution state separation
# ============================================================================


def test_agent_context_world_outer_mapping_is_immutable():
    runtime = AgentRuntime()

    root = runtime.create_root(
        world={
            "state": {
                "value": 1,
            }
        }
    )

    with pytest.raises(
        TypeError
    ):
        root.world["other"] = {}


def test_agent_runtime_freezes_only_outer_world_mapping():
    """
    AgentRuntime._freeze_world() deliberately performs only:

        MappingProxyType(dict(world))

    Therefore:
        - the outer mapping is protected
        - nested objects are NOT recursively copied/frozen

    Deep-detached world snapshots are the responsibility of the
    DataSpace / Module snapshot producer.
    """

    runtime = AgentRuntime()

    source = {
        "state": {
            "value": 1,
        }
    }

    root = runtime.create_root(
        world=source
    )

    assert (
        root.world
        is not source
    )

    # Outer mapping is frozen.
    with pytest.raises(
        TypeError
    ):
        root.world["other"] = {}

    # Nested object remains shared by the shallow outer copy.
    source["state"]["value"] = 99

    assert (
        root.world[
            "state"
        ]["value"]
        == 99
    )


def test_dataspace_snapshot_provides_the_deep_detachment_contract():
    """
    The actual Agent world normally originates from:

        modules.snapshot()
            ->
        DataSpace.snapshot()

    DataSpace.snapshot() is responsible for the deep copy.
    """

    modules = FakeModules()

    first = modules.snapshot()

    modules.space.publish(
        {
            "value": 2,
        }
    )

    assert (
        first["state"]["value"]
        == 1
    )


# ============================================================================
# CoreAgent skill / persona / world boundary
# ============================================================================


def test_turn_boundary_refreshes_skill_persona_and_world_together():
    async def scenario():
        modules = FakeModules()

        persona = {
            "value": "persona-v1"
        }

        skills = FakeSkills()

        core = CoreAgent(
            llm=FakeLLM(),
            modules=modules,
            providers=FakeProviders(),
            skills=skills,
            persona_source=lambda: persona[
                "value"
            ],
        )

        engine = CapturingEngine()
        core.engine = engine

        await core.run(
            "first"
        )

        first = (
            engine.executions[0]
        )

        assert (
            first.persona
            == "persona-v1"
        )

        assert (
            first.context.world[
                "state"
            ]["value"]
            == 1
        )

        persona[
            "value"
        ] = "persona-v2"

        modules.space.publish(
            {
                "value": 2,
            }
        )

        await core.run(
            "second"
        )

        second = (
            engine.executions[1]
        )

        assert (
            second.persona
            == "persona-v2"
        )

        assert (
            second.context.world[
                "state"
            ]["value"]
            == 2
        )

        assert (
            skills.refresh_calls
            == 2
        )

        assert (
            modules.snapshot_calls
            == 2
        )

    run(
        scenario()
    )

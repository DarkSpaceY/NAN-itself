
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

import pytest

from nan_itself.agent.core import Agent
from nan_itself.agent.model import SubagentLimitError
from nan_itself.agent.prompts import render_turn
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
# Fake Skills / Providers
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


# ============================================================================
# Turn consistency fixtures
# ============================================================================


@dataclass
class CapturedExecution:
    turn: object

    @property
    def world(self):
        return self.turn.world

    @property
    def persona(self):
        return self.turn.persona


class CapturingEngine:
    """
    Replaces StepEngine inside Agent.

    This lets tests inspect the exact Turn created at each turn
    boundary without invoking an LLM. It mirrors the real engine's
    contract: the assistant's outcome is written back structurally
    (reply), and the Agent chains turns through it.
    """

    def __init__(self):
        self.executions = []

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
            outcome = callback(turn)

            if asyncio.iscoroutine(outcome):
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
    modules,
    **kwargs,
):
    skills = FakeSkills()

    agent = Agent(
        engine=CapturingEngine(),
        modules=modules,
        tools=FakeProviders(),
        skills=skills,
        **kwargs,
    )

    return (
        agent,
        agent.engine,
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
# Agent turn snapshot
# ============================================================================


def test_agent_captures_exactly_one_world_snapshot_per_turn():
    """
    Agent is responsible for the turn boundary:

        skills.refresh()
        persona re-read
        modules.snapshot()
        modules.query_snapshot() (observation assembly)
        engine.step()

    The fake CapturingEngine deliberately bypasses StepEngine, so
    the query_snapshot() call comes from the Agent itself.
    """

    async def scenario():
        modules = FakeModules()

        agent, engine, skills = make_agent(
            modules,
        )

        await agent.run()

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
            == 1
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

        agent, engine, skills = make_agent(
            modules,
        )

        await agent.run()

        first_turn = (
            engine.executions[0].turn
        )

        assert (
            first_turn.world[
                "state"
            ]["value"]
            == 1
        )

        modules.space.publish(
            {
                "value": 2,
            }
        )

        await agent.run()

        second_turn = (
            engine.executions[1].turn
        )

        assert (
            first_turn.world[
                "state"
            ]["value"]
            == 1
        )

        assert (
            second_turn.world[
                "state"
            ]["value"]
            == 2
        )

        assert (
            first_turn.world
            is not second_turn.world
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

        agent, engine, skills = make_agent(
            modules,
        )

        observed = {}

        async def on_execute(
            turn,
        ):
            observed["before"] = (
                turn.world[
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
                turn.world[
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

        await agent.run()

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

        turn = (
            engine.executions[0].turn
        )

        assert (
            turn.world[
                "state"
            ]["value"]
            == 1
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

        agent, engine, skills = make_agent(
            modules,
        )

        await agent.run()

        first = (
            engine.executions[0].turn
        )

        modules.space.publish(
            {
                "value": 2,
            }
        )

        await agent.run()

        second = (
            engine.executions[1].turn
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
# Root identity
# ============================================================================


def test_root_agent_hash_is_stable_across_turns():
    async def scenario():
        agent, engine, skills = make_agent(
            FakeModules(),
        )

        await agent.run()
        await agent.run()

        assert len(engine.executions) == 2

        # Same Agent instance, same identity, no rotation: both
        # turns carry the instance's own hash.
        assert (
            engine.executions[0].turn.agent_hash
            == agent.agent_hash
        )

        assert (
            engine.executions[1].turn.agent_hash
            == agent.agent_hash
        )

        assert engine.executions[0].turn.depth == 0

        assert (
            engine.executions[0].turn.parent_hash
            is None
        )

    run(
        scenario()
    )


# ============================================================================
# Spawn tree (one homogeneous Agent class)
# ============================================================================


def test_spawn_builds_a_homogeneous_child_tree():
    async def scenario():
        modules = FakeModules()

        root, engine, skills = make_agent(
            modules,
        )

        child = root.spawn("child task")

        assert child.depth == 1

        assert child.parent is root

        assert (
            child.parent_hash
            == root.agent_hash
        )

        assert child.task == "child task"

        # Capabilities are shared references, not copies.
        assert child.engine is root.engine
        assert child.modules is root.modules
        assert child.tools is root.tools
        assert child.skills is root.skills

        assert (
            child.agent_hash
            != root.agent_hash
        )

        assert child.max_subagent_depth == (
            root.max_subagent_depth
        )

        assert root.children == [child]

        assert child._task is not None

        grandchild = child.spawn("grandchild task")

        assert grandchild.depth == 2

        assert (
            grandchild.parent_hash
            == child.agent_hash
        )

        assert grandchild.task == "grandchild task"

        assert child.children == [grandchild]

        for agent in (child, grandchild):
            agent._task.cancel()

        await asyncio.gather(
            child._task,
            grandchild._task,
            return_exceptions=True,
        )

    run(
        scenario()
    )


def test_spawn_rejects_beyond_max_subagent_depth():
    async def scenario():
        root, engine, skills = make_agent(
            FakeModules(),
            max_subagent_depth=1,
        )

        child = root.spawn("child task")

        with pytest.raises(SubagentLimitError):
            child.spawn("grandchild task")

        child._task.cancel()

        await asyncio.gather(
            child._task,
            return_exceptions=True,
        )

    run(
        scenario()
    )


# ============================================================================
# Stop cascade (no grace window anywhere)
# ============================================================================


class BlockingEngine:
    """
    Stub engine: the root's first turn spawns one child, and every
    child turn blocks forever inside step() -- an in-flight turn
    that only a task cancellation can end.

    step() no longer receives the Agent, so the test wires the root
    in (the stub is shared by the whole tree).
    """

    def __init__(self):
        self.root = None

        self.child_started = asyncio.Event()

    async def step(
        self,
        turn,
    ):
        if turn.depth == 0:
            if not self.root.children:
                self.root.spawn("child task")

            return replace(
                turn,
                reply="ok",
            )

        self.child_started.set()

        await asyncio.Event().wait()

    def render_turn(
        self,
        turn,
    ):
        return render_turn(turn)


async def cascade_stop():
    """
    Run a root whose first turn spawns a child that then blocks
    inside its own turn; then stop the root and let the teardown
    cascade down the tree.
    """
    engine = BlockingEngine()

    root = Agent(
        engine=engine,
        modules=FakeModules(),
        tools=FakeProviders(),
        skills=FakeSkills(),
        autonomous_interval=0.05,
    )

    engine.root = root

    root_task = asyncio.create_task(
        root.loop()
    )

    await asyncio.wait_for(
        engine.child_started.wait(),
        timeout=1.0,
    )

    child = root.children[0]

    assert child.depth == 1

    assert not child._task.done()

    root.request_stop()

    await asyncio.wait_for(
        root_task,
        timeout=1.0,
    )

    return child


def test_cascade_stop_cancels_a_blocked_child():
    async def scenario():
        child = await cascade_stop()

        assert child.stopping

        assert child._task.done()

    run(
        scenario()
    )


def test_cancelled_child_reports_failure():
    async def scenario():
        child = await cascade_stop()

        assert child.done

        assert child.report is not None

        assert child.report.status == "failed"

        assert (
            child.report.body
            == "error: cancelled"
        )

        assert (
            child.report.task
            == "child task"
        )

    run(
        scenario()
    )


# ============================================================================
# Agent skill / persona / world boundary
# ============================================================================


def test_turn_boundary_refreshes_skill_persona_and_world_together(
    tmp_path,
):
    async def scenario():
        modules = FakeModules()

        (tmp_path / "persona.md").write_text(
            "persona-v1",
            encoding="utf-8",
        )

        agent, engine, skills = make_agent(
            modules,
        )

        await agent.run()

        first = (
            engine.executions[0]
        )

        assert (
            first.persona
            == "persona-v1"
        )

        assert (
            first.turn.world[
                "state"
            ]["value"]
            == 1
        )

        # Persona edits hot-reload: the file is re-read every turn.
        (tmp_path / "persona.md").write_text(
            "persona-v2",
            encoding="utf-8",
        )

        modules.space.publish(
            {
                "value": 2,
            }
        )

        await agent.run()

        second = (
            engine.executions[1]
        )

        assert (
            second.persona
            == "persona-v2"
        )

        assert (
            second.turn.world[
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

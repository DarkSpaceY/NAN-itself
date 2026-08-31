from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nan_itself.agent.runtime import (
    AgentRuntime,
    SubagentLimitError,
)
from nan_itself.agent.verbs import (
    DispatchVerb,
    ExecutionState,
)


def run(coro):
    return asyncio.run(coro)


# ============================================================================
# Root context
# ============================================================================


def test_create_root_assigns_identity_and_freezes_world():
    runtime = AgentRuntime(
        max_subagent_depth=3,
    )

    world = {
        "memory": {
            "value": 1,
        },
        "system": {
            "status": "ok",
        },
    }

    context = runtime.create_root(
        world=world,
        task="root task",
    )

    assert context.depth == 0
    assert context.parent_hash is None
    assert context.task == "root task"
    assert context.skill is None

    assert dict(context.world) == world

    assert context.agent_hash in {
        agent.agent_hash
        for agent in runtime.agents()
    }

    with pytest.raises(TypeError):
        context.world["new"] = {}  # type: ignore[index]


def test_root_world_mapping_isolated_from_original_outer_mapping():
    runtime = AgentRuntime()

    world = {
        "memory": {
            "value": 1,
        }
    }

    context = runtime.create_root(
        world=world,
    )

    world["other"] = {
        "value": 2,
    }

    assert "other" not in context.world


# ============================================================================
# Subagent context inheritance
# ============================================================================


def test_subagent_inherits_world_identity_and_skill():
    async def scenario():
        runtime = AgentRuntime(
            max_subagent_depth=3,
        )

        skill = SimpleNamespace(
            name="research",
        )

        root = runtime.create_root(
            world={
                "memory": {
                    "value": 1,
                }
            },
            skill=skill,
            task="root",
        )

        observed = {}

        async def worker(context):
            observed["context"] = context
            return "done"

        handle = runtime.dispatch(
            root,
            task="child",
            worker=worker,
        )

        result = await handle.wait()

        child = observed["context"]

        assert result == "done"

        assert child.agent_hash != (
            root.agent_hash
        )

        assert child.parent_hash == (
            root.agent_hash
        )

        assert child.depth == 1

        assert child.task == "child"

        assert child.world is root.world

        assert child.skill is skill

    run(
        scenario()
    )


def test_nested_subagent_inherits_same_world():
    async def scenario():
        runtime = AgentRuntime(
            max_subagent_depth=3,
        )

        root = runtime.create_root(
            world={
                "x": 1,
            }
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

        result = await child_handle.wait()

        assert result == "grandchild"

        child = observed["child"]
        grandchild = observed["grandchild"]

        assert child.depth == 1
        assert grandchild.depth == 2

        assert child.world is root.world
        assert grandchild.world is root.world

    run(
        scenario()
    )


# ============================================================================
# Depth limit
# ============================================================================


def test_subagent_depth_limit_is_enforced():
    async def scenario():
        runtime = AgentRuntime(
            max_subagent_depth=1,
        )

        root = runtime.create_root()

        async def worker(context):
            return "done"

        child = runtime.dispatch(
            root,
            task="child",
            worker=worker,
        )

        assert (
            await child.wait()
            == "done"
        )

        child_context = (
            runtime.get_agent(
                child.agent_hash
            )
        )

        assert child_context is not None

        with pytest.raises(
            SubagentLimitError,
            match=(
                "Maximum Subagent depth exceeded"
            ),
        ):
            runtime.dispatch(
                child_context,
                task="grandchild",
                worker=worker,
            )

    run(
        scenario()
    )


def test_zero_depth_limit_disallows_any_child():
    async def scenario():
        runtime = AgentRuntime(
            max_subagent_depth=0,
        )

        root = runtime.create_root()

        async def worker(context):
            return "done"

        with pytest.raises(
            SubagentLimitError
        ):
            runtime.dispatch(
                root,
                task="child",
                worker=worker,
            )

    run(
        scenario()
    )


# ============================================================================
# Dispatch lifecycle
# ============================================================================


def test_dispatch_returns_immediately_without_waiting_for_worker():
    async def scenario():
        runtime = AgentRuntime()

        root = runtime.create_root()

        entered = asyncio.Event()
        release = asyncio.Event()

        async def worker(context):
            entered.set()
            await release.wait()
            return "done"

        handle = runtime.dispatch(
            root,
            task="slow",
            worker=worker,
        )

        await asyncio.wait_for(
            entered.wait(),
            timeout=1,
        )

        assert not handle.done
        assert not handle.cancelled

        release.set()

        assert (
            await handle.wait()
            == "done"
        )

        assert handle.done
        assert not handle.cancelled

    run(
        scenario()
    )


def test_cancelled_subagent_reports_cancelled():
    async def scenario():
        runtime = AgentRuntime()

        root = runtime.create_root()

        entered = asyncio.Event()
        release = asyncio.Event()

        async def worker(context):
            entered.set()
            await release.wait()
            return "done"

        handle = runtime.dispatch(
            root,
            task="cancel me",
            worker=worker,
        )

        await asyncio.wait_for(
            entered.wait(),
            timeout=1,
        )

        assert handle.cancel()

        with pytest.raises(
            asyncio.CancelledError
        ):
            await handle.wait()

        assert handle.done
        assert handle.cancelled

    run(
        scenario()
    )


# ============================================================================
# AgentRuntime wait helpers
# ============================================================================


def test_wait_all_returns_results_in_handle_order():
    async def scenario():
        runtime = AgentRuntime()

        root = runtime.create_root()

        async def first_worker(
            context,
        ):
            await asyncio.sleep(0)
            return "first"

        async def second_worker(
            context,
        ):
            await asyncio.sleep(0)
            return "second"

        first = runtime.dispatch(
            root,
            task="first",
            worker=first_worker,
        )

        second = runtime.dispatch(
            root,
            task="second",
            worker=second_worker,
        )

        result = await runtime.wait_all(
            first,
            second,
        )

        assert result == [
            "first",
            "second",
        ]

    run(
        scenario()
    )


# ============================================================================
# Sleep / interrupt
# ============================================================================


def test_sleep_without_interrupt_event():
    async def scenario():
        runtime = AgentRuntime()

        started = (
            asyncio.get_running_loop().time()
        )

        waited, interrupted = (
            await runtime.sleep(
                0.01
            )
        )

        elapsed = (
            asyncio.get_running_loop().time()
            - started
        )

        assert waited >= 0.0
        assert elapsed >= 0.0
        assert interrupted is False

    run(
        scenario()
    )


def test_sleep_is_interrupted_by_new_input():
    async def scenario():
        runtime = AgentRuntime()

        interrupt = asyncio.Event()

        runtime.set_interrupt_event(
            interrupt
        )

        async def trigger():
            await asyncio.sleep(
                0.01
            )
            interrupt.set()

        trigger_task = asyncio.create_task(
            trigger()
        )

        waited, interrupted = (
            await runtime.sleep(
                5.0
            )
        )

        await trigger_task

        assert interrupted is True
        assert waited < 1.0

    run(
        scenario()
    )


# ============================================================================
# Current Skill inheritance semantics
# ============================================================================


def test_dispatch_verb_passes_current_active_skill_to_child():
    """
    Parent context starts with no Skill.

    Current execution state activates a Skill.

    Dispatch must propagate the current active Skill to the child.
    """

    async def scenario():
        runtime = AgentRuntime()

        skill = SimpleNamespace(
            name="research",
        )

        root = runtime.create_root(
            skill=None,
        )

        state = ExecutionState(
            active_skill=skill,
            persona="test",
        )

        observed = {}

        async def execute_child(
            *,
            context,
            user_input,
            persona,
        ):
            observed["context"] = context

            return SimpleNamespace(
                content="child done"
            )

        engine = SimpleNamespace(
            agent_runtime=runtime,
            execute=execute_child,
        )

        result = await DispatchVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "task": "research child",
                }
            ),
            context=root,
            state=state,
            engine=engine,
        )

        assert (
            "Subagent dispatched."
            in result
        )

        assert len(
            state.children
        ) == 1

        handle = (
            state.children[0].handle
        )

        child_result = (
            await handle.wait()
        )

        assert (
            child_result.content
            == "child done"
        )

        child_context = (
            observed["context"]
        )

        assert (
            child_context.skill
            is skill
        )

        assert (
            child_context.parent_hash
            == root.agent_hash
        )

        assert (
            child_context.world
            is root.world
        )

    run(
        scenario()
    )


def test_dispatch_verb_without_active_skill_child_has_no_skill():
    """
    Explicit regression for the opposite case.

    A parent that has no active Skill must not manufacture one
    for its child.
    """

    async def scenario():
        runtime = AgentRuntime()

        root = runtime.create_root(
            skill=None,
        )

        state = ExecutionState(
            active_skill=None,
            persona="test",
        )

        observed = {}

        async def execute_child(
            *,
            context,
            user_input,
            persona,
        ):
            observed["context"] = context

            return SimpleNamespace(
                content="done"
            )

        engine = SimpleNamespace(
            agent_runtime=runtime,
            execute=execute_child,
        )

        await DispatchVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "task": "child",
                }
            ),
            context=root,
            state=state,
            engine=engine,
        )

        await state.children[
            0
        ].handle.wait()

        assert (
            observed["context"].skill
            is None
        )

    run(
        scenario()
    )
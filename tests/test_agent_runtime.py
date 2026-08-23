from __future__ import annotations

import asyncio

import pytest

from src.nan_itself.agent.runtime import (
    AgentRuntime,
    SubagentLimitError,
)


def test_root_agent_has_depth_zero():
    runtime = AgentRuntime()

    root = runtime.create_root(
        world={
            "foo": 42,
        },
    )

    assert root.depth == 0
    assert root.parent_hash is None
    assert root.agent_hash
    assert root.world["foo"] == 42


@pytest.mark.asyncio
async def test_subagent_runs_asynchronously():
    runtime = AgentRuntime()

    root = runtime.create_root(
        world={
            "value": 42,
        },
    )

    started = asyncio.Event()
    release = asyncio.Event()

    async def worker(context):
        started.set()
        await release.wait()

        return context.world["value"]

    handle = runtime.dispatch(
        root,
        task="do something",
        worker=worker,
    )

    assert handle.done is False

    await started.wait()

    assert handle.done is False

    release.set()

    result = await handle.wait()

    assert result == 42
    assert handle.done is True


@pytest.mark.asyncio
async def test_subagent_gets_same_world_snapshot():
    runtime = AgentRuntime()

    world = {
        "todo": {
            "active": 3,
        },
    }

    root = runtime.create_root(
        world=world,
    )

    observed = []

    async def worker(context):
        observed.append(context.world)
        return None

    first = runtime.dispatch(
        root,
        task="first",
        worker=worker,
    )

    second = runtime.dispatch(
        root,
        task="second",
        worker=worker,
    )

    await asyncio.gather(
        first.wait(),
        second.wait(),
    )

    assert len(observed) == 2
    assert observed[0] is root.world
    assert observed[1] is root.world


@pytest.mark.asyncio
async def test_subagent_has_independent_history():
    runtime = AgentRuntime()

    root = runtime.create_root(
        history=("root-message",),
    )

    observed = []

    async def worker(context):
        observed.append(context.history)

    handle = runtime.dispatch(
        root,
        task="work",
        worker=worker,
        history=("subagent-message",),
    )

    await handle.wait()

    assert root.history == (
        "root-message",
    )

    assert observed == [
        (
            "subagent-message",
        )
    ]


def test_subagent_gets_new_identity():
    runtime = AgentRuntime()

    root = runtime.create_root()

    async def worker(context):
        pass

    async def create():
        handle = runtime.dispatch(
            root,
            task="work",
            worker=worker,
        )

        assert handle.agent_hash != root.agent_hash
        assert handle.parent_hash == root.agent_hash
        assert handle.depth == 1

        await handle.wait()

    asyncio.run(create())


@pytest.mark.asyncio
async def test_subagent_can_use_different_skill():
    runtime = AgentRuntime()

    root = runtime.create_root(
        skill="core",
    )

    observed = []

    async def worker(context):
        observed.append(context.skill)

    handle = runtime.dispatch(
        root,
        task="research",
        worker=worker,
        skill="research",
    )

    await handle.wait()

    assert root.skill == "core"
    assert observed == ["research"]


@pytest.mark.asyncio
async def test_subagent_inherits_parent_skill_by_default():
    runtime = AgentRuntime()

    root = runtime.create_root(
        skill="core",
    )

    observed = []

    async def worker(context):
        observed.append(context.skill)

    handle = runtime.dispatch(
        root,
        task="work",
        worker=worker,
    )

    await handle.wait()

    assert observed == ["core"]


@pytest.mark.asyncio
async def test_subagent_can_dispatch_subagent():
    runtime = AgentRuntime(
        max_subagent_depth=3,
    )

    root = runtime.create_root()

    observed = []

    async def grandchild(context):
        observed.append(
            (
                context.depth,
                context.parent_hash,
            )
        )

    async def child(context):
        assert context.depth == 1

        handle = runtime.dispatch(
            context,
            task="grandchild",
            worker=grandchild,
        )

        await handle.wait()

    handle = runtime.dispatch(
        root,
        task="child",
        worker=child,
    )

    await handle.wait()

    assert observed[0][0] == 2


@pytest.mark.asyncio
async def test_subagent_depth_limit():
    runtime = AgentRuntime(
        max_subagent_depth=1,
    )

    root = runtime.create_root()

    async def child_worker(context):
        with pytest.raises(
            SubagentLimitError,
        ):
            runtime.dispatch(
                context,
                task="too deep",
                worker=child_worker,
            )

    handle = runtime.dispatch(
        root,
        task="child",
        worker=child_worker,
    )

    await handle.wait()


@pytest.mark.asyncio
async def test_subagent_failure_is_visible_through_handle():
    runtime = AgentRuntime()

    root = runtime.create_root()

    async def worker(context):
        raise RuntimeError(
            "subagent failed"
        )

    handle = runtime.dispatch(
        root,
        task="fail",
        worker=worker,
    )

    with pytest.raises(
        RuntimeError,
        match="subagent failed",
    ):
        await handle.wait()


@pytest.mark.asyncio
async def test_sleep_is_independent_of_subagent():
    runtime = AgentRuntime()

    root = runtime.create_root()

    finished = asyncio.Event()

    async def worker(context):
        await asyncio.sleep(0.05)
        finished.set()

    handle = runtime.dispatch(
        root,
        task="delayed",
        worker=worker,
    )

    assert not finished.is_set()

    await runtime.sleep(0.01)

    assert not finished.is_set()

    await handle.wait()

    assert finished.is_set()


@pytest.mark.asyncio
async def test_wait_all():
    runtime = AgentRuntime()

    root = runtime.create_root()

    async def worker(context):
        await asyncio.sleep(
            0.01 * context.depth
        )

        return context.task

    handles = [
        runtime.dispatch(
            root,
            task=f"task-{i}",
            worker=worker,
        )
        for i in range(3)
    ]

    results = await runtime.wait_all(
        *handles,
    )

    assert set(results) == {
        "task-0",
        "task-1",
        "task-2",
    }


@pytest.mark.asyncio
async def test_cancel_subagent():
    runtime = AgentRuntime()

    root = runtime.create_root()

    started = asyncio.Event()

    async def worker(context):
        started.set()

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return "cancelled"

    handle = runtime.dispatch(
        root,
        task="cancel me",
        worker=worker,
    )

    await started.wait()

    assert handle.cancel() is True

    result = await handle.wait()

    assert result == "cancelled"


def test_negative_depth_is_rejected():
    with pytest.raises(ValueError):
        AgentRuntime(
            max_subagent_depth=-1,
        )


@pytest.mark.asyncio
async def test_agent_registry_keeps_created_agents():
    runtime = AgentRuntime()

    root = runtime.create_root()

    async def worker(context):
        return context.agent_hash

    handle = runtime.dispatch(
        root,
        task="child",
        worker=worker,
    )

    child_hash = await handle.wait()

    assert runtime.get_agent(root.agent_hash) is root

    child = runtime.get_agent(
        child_hash
    )

    assert child is not None
    assert child.parent_hash == root.agent_hash
    assert child.depth == 1
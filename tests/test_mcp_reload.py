from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from nan_itself.tools import mcp as mcp_backend
from nan_itself.tools.provider import Provider
from nan_itself.tools.results import text_result
from nan_itself.tools.runtime import ProviderRuntime
from nan_itself.tools.spec import (
    PROVIDER_KIND_MCP,
    ProviderSpec,
)
from nan_itself.tools.view import AgentToolView


TEST_TIMEOUT = 2.0


def run(coro):
    """
    Run one complete async test scenario.

    IMPORTANT:
    A ProviderRuntime/MCP worker lifecycle must stay inside ONE
    event loop. asyncio.run() creates and destroys an event loop,
    so worker Tasks must never be reused across separate run() calls.
    """
    return asyncio.run(coro)


def result_text(result):
    return result.content[0].text


async def wait_event(
    event: asyncio.Event,
    *,
    name: str,
    timeout: float = TEST_TIMEOUT,
) -> None:
    try:
        await asyncio.wait_for(
            event.wait(),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        raise AssertionError(
            f"Timed out waiting for {name}"
        ) from exc


async def wait_until(
    predicate,
    *,
    name: str,
    timeout: float = TEST_TIMEOUT,
) -> None:
    loop = asyncio.get_running_loop()

    deadline = (
        loop.time() + timeout
    )

    while loop.time() < deadline:
        if predicate():
            return

        await asyncio.sleep(0)

    raise AssertionError(
        f"Timed out waiting for {name}"
    )


async def stop_all_workers(
    runtime: ProviderRuntime,
) -> None:
    """
    Clean up all currently active MCP workers.

    Must be called from the same event loop that created them.
    """
    workers = list(
        runtime._mcp_workers.values()
    )

    if workers:
        await runtime._stop_mcp_workers(
            workers
        )


# ============================================================================
# Fake MCP primitives
# ============================================================================


class FakeStack:
    """
    Fake AsyncExitStack-like object.

    It records which Task created/owns it and which Task closes it.
    """

    def __init__(self):
        self.owner_task = (
            asyncio.current_task()
        )

        self.close_task = None

        self.close_started = (
            asyncio.Event()
        )

        self.release_close = (
            asyncio.Event()
        )

        self.block_close = False

    async def aclose(self):
        self.close_task = (
            asyncio.current_task()
        )

        self.close_started.set()

        if self.block_close:
            await self.release_close.wait()


class FakeSession:
    """
    Minimal MCP session used to exercise ProviderRuntime.
    """

    def __init__(
        self,
        *,
        generation: str,
        tools=None,
        block_calls: bool = False,
    ):
        self.generation = generation

        self.tools = list(
            tools
            or [
                SimpleNamespace(
                    name="echo",
                    description=(
                        f"echo-{generation}"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                            }
                        },
                        "required": [
                            "text"
                        ],
                        "additionalProperties": False,
                    },
                )
            ]
        )

        self.block_calls = (
            block_calls
        )

        self.call_started = (
            asyncio.Event()
        )

        self.release_call = (
            asyncio.Event()
        )

        self.calls = []

    async def list_tools(self):
        return SimpleNamespace(
            tools=self.tools
        )

    async def call_tool(
        self,
        name,
        arguments,
    ):
        self.calls.append(
            (
                name,
                dict(arguments),
            )
        )

        self.call_started.set()

        if self.block_calls:
            await self.release_call.wait()

        return text_result(
            f"{self.generation}:{arguments['text']}"
        )


def make_provider(
    spec: ProviderSpec,
    *,
    generation: str,
    stack: FakeStack,
    session: FakeSession,
) -> Provider:
    return Provider(
        spec=spec,
        stack=stack,
        session=session,
        tools={
            tool.name: tool
            for tool in session.tools
        },
    )


def write_mcp_config(
    path: Path,
    *,
    name: str,
    generation: str,
) -> None:
    path.write_text(
        "\n".join(
            [
                f"name: {name}",
                "command: fake-mcp",
                "args:",
                f"  - {generation}",
            ]
        ),
        encoding="utf-8",
    )


# ============================================================================
# MCP worker lifecycle
# ============================================================================


def test_mcp_worker_closes_stack_in_its_own_task(
    tmp_path,
    monkeypatch,
):
    """
    The worker that owns the MCP connection must also close the stack.

    This protects against AnyIO cancel-scope / cross-task shutdown errors.
    """

    async def scenario():
        stack_holder = {}

        async def fake_connect(
            spec,
        ):
            stack = FakeStack()

            session = FakeSession(
                generation="v1"
            )

            stack_holder["stack"] = stack

            return make_provider(
                spec,
                generation="v1",
                stack=stack,
                session=session,
            )

        monkeypatch.setattr(
            mcp_backend,
            "connect",
            fake_connect,
        )

        runtime = ProviderRuntime(
            workspace_mcp_dir=(
                tmp_path / "mcps"
            ),
            workspace_local_dir=(
                tmp_path / "local"
            ),
            builtin_tools=(),
        )

        spec = ProviderSpec(
            name="example",
            kind=PROVIDER_KIND_MCP,
            command="fake-mcp",
            args=("v1",),
            source="<test>",
            origin="workspace",
        )

        try:
            worker = (
                await runtime._start_mcp_worker(
                    spec
                )
            )

            assert worker.task is not None
            assert worker.provider is not None

            stack = stack_holder["stack"]

            assert (
                stack.owner_task
                is worker.task
            )

            await runtime._stop_mcp_worker(
                worker
            )

            assert (
                stack.close_task
                is worker.task
            )

            assert (
                worker.provider.stack
                is None
            )

        finally:
            await stop_all_workers(
                runtime
            )

    run(
        scenario()
    )


# ============================================================================
# MCP reload + in-flight calls
# ============================================================================


def test_mcp_reload_installs_new_provider_while_old_call_drains(
    tmp_path,
    monkeypatch,
):
    """
    Full generation transition:

        v1 worker
            |
            | in-flight call
            v
        reload
            |
            +--> v2 worker installed
            |
            +--> v1 worker stopping
                    |
                    +--> wait for active call
                    |
                    +--> close v1 stack

    Existing calls stay on v1.
    New calls use v2.
    """

    async def scenario():
        source = (
            tmp_path / "example.yaml"
        )

        write_mcp_config(
            source,
            name="example",
            generation="v1",
        )

        stacks = {}
        sessions = {}

        async def fake_connect(
            spec,
        ):
            generation = spec.args[0]

            stack = FakeStack()

            session = FakeSession(
                generation=generation,
                block_calls=(
                    generation == "v1"
                ),
            )

            stacks[generation] = stack
            sessions[generation] = session

            return make_provider(
                spec,
                generation=generation,
                stack=stack,
                session=session,
            )

        monkeypatch.setattr(
            mcp_backend,
            "connect",
            fake_connect,
        )

        runtime = ProviderRuntime(
            workspace_mcp_dir=tmp_path,
            workspace_local_dir=(
                tmp_path / "local"
            ),
            builtin_tools=(),
        )

        try:
            # ----------------------------------------------------------
            # Start v1.
            # ----------------------------------------------------------

            await runtime._reload_workspace_source(
                source,
                (1, 2),
            )

            old_provider = (
                runtime.get_provider(
                    "example"
                )
            )

            assert old_provider is not None

            old_worker = (
                runtime._mcp_workers[
                    "example"
                ]
            )

            assert (
                old_worker.provider
                is old_provider
            )

            # ----------------------------------------------------------
            # Start an in-flight call on v1.
            # ----------------------------------------------------------

            view = AgentToolView(
                runtime,
                active_provider="example",
            )

            old_call = asyncio.create_task(
                view.call_tool(
                    "echo",
                    {
                        "text": "hello",
                    },
                )
            )

            await wait_event(
                sessions["v1"].call_started,
                name="v1 call start",
            )

            assert (
                old_worker.active_calls
                == 1
            )

            # ----------------------------------------------------------
            # Replace workspace source with v2.
            # ----------------------------------------------------------

            write_mcp_config(
                source,
                name="example",
                generation="v2",
            )

            reload_task = asyncio.create_task(
                runtime._reload_workspace_source(
                    source,
                    (2, 2),
                )
            )

            await wait_until(
                lambda: (
                    runtime.get_provider(
                        "example"
                    )
                    is not old_provider
                ),
                name="v2 provider installation",
            )

            new_provider = (
                runtime.get_provider(
                    "example"
                )
            )

            assert new_provider is not None

            assert (
                new_provider
                is not old_provider
            )

            new_worker = (
                runtime._mcp_workers[
                    "example"
                ]
            )

            assert (
                new_worker.provider
                is new_provider
            )

            # Old worker has been removed from the active map,
            # but is still alive and draining the in-flight call.
            assert (
                old_worker
                not in runtime._mcp_workers.values()
            )

            assert old_worker.stopping

            assert (
                old_worker.active_calls
                == 1
            )

            assert (
                not reload_task.done()
            )

            assert (
                not old_call.done()
            )

            # ----------------------------------------------------------
            # Finish old call.
            # ----------------------------------------------------------

            sessions[
                "v1"
            ].release_call.set()

            old_result = (
                await asyncio.wait_for(
                    old_call,
                    timeout=TEST_TIMEOUT,
                )
            )

            assert (
                result_text(old_result)
                == "v1:hello"
            )

            # ----------------------------------------------------------
            # Wait for old worker shutdown and reload completion.
            # ----------------------------------------------------------

            await asyncio.wait_for(
                reload_task,
                timeout=TEST_TIMEOUT,
            )

            assert (
                old_worker.active_calls
                == 0
            )

            assert (
                stacks["v1"].close_task
                is old_worker.task
            )

            assert (
                old_provider.stack
                is None
            )

            # New provider remains live.
            assert (
                runtime._mcp_workers[
                    "example"
                ]
                is new_worker
            )

            assert (
                new_provider.stack
                is not None
            )

            # ----------------------------------------------------------
            # New call must use v2.
            # ----------------------------------------------------------

            new_result = (
                await view.call_tool(
                    "echo",
                    {
                        "text": "world",
                    },
                )
            )

            assert (
                result_text(new_result)
                == "v2:world"
            )

            assert (
                sessions["v2"].calls
                == [
                    (
                        "echo",
                        {
                            "text": "world",
                        },
                    )
                ]
            )

        finally:
            await stop_all_workers(
                runtime
            )

    run(
        scenario()
    )


def test_old_mcp_worker_shutdown_cannot_remove_new_provider(
    tmp_path,
    monkeypatch,
):
    """
    Old worker finalization must not remove a replacement provider
    that already owns the same provider name.
    """

    async def scenario():
        source = (
            tmp_path / "example.yaml"
        )

        write_mcp_config(
            source,
            name="example",
            generation="v1",
        )

        created = {}

        async def fake_connect(
            spec,
        ):
            generation = spec.args[0]

            stack = FakeStack()

            session = FakeSession(
                generation=generation
            )

            if generation == "v1":
                stack.block_close = True

            created[generation] = (
                stack,
                session,
            )

            return make_provider(
                spec,
                generation=generation,
                stack=stack,
                session=session,
            )

        monkeypatch.setattr(
            mcp_backend,
            "connect",
            fake_connect,
        )

        runtime = ProviderRuntime(
            workspace_mcp_dir=tmp_path,
            workspace_local_dir=(
                tmp_path / "local"
            ),
            builtin_tools=(),
        )

        try:
            # ----------------------------------------------------------
            # Start v1.
            # ----------------------------------------------------------

            await runtime._reload_workspace_source(
                source,
                (1, 2),
            )

            old_provider = (
                runtime.get_provider(
                    "example"
                )
            )

            assert old_provider is not None

            old_worker = (
                runtime._mcp_workers[
                    "example"
                ]
            )

            # ----------------------------------------------------------
            # Replace with v2.
            # ----------------------------------------------------------

            write_mcp_config(
                source,
                name="example",
                generation="v2",
            )

            reload_task = asyncio.create_task(
                runtime._reload_workspace_source(
                    source,
                    (2, 2),
                )
            )

            old_stack = created[
                "v1"
            ][0]

            await wait_event(
                old_stack.close_started,
                name="v1 stack close",
            )

            new_provider = (
                runtime.get_provider(
                    "example"
                )
            )

            assert new_provider is not None

            assert (
                new_provider
                is not old_provider
            )

            new_worker = (
                runtime._mcp_workers[
                    "example"
                ]
            )

            assert (
                new_worker.provider
                is new_provider
            )

            assert (
                new_worker
                is not old_worker
            )

            # Old shutdown is still blocked.
            assert (
                not reload_task.done()
            )

            # Release old stack.
            old_stack.release_close.set()

            await asyncio.wait_for(
                reload_task,
                timeout=TEST_TIMEOUT,
            )

            # Old finalizer must not have removed v2.
            assert (
                runtime.get_provider(
                    "example"
                )
                is new_provider
            )

            assert (
                "example"
                in runtime._mcp_workers
            )

            assert (
                runtime._mcp_workers[
                    "example"
                ]
                is new_worker
            )

        finally:
            await stop_all_workers(
                runtime
            )

    run(
        scenario()
    )


# ============================================================================
# Agent view / tool-definition replacement
# ============================================================================


def test_agent_view_uses_new_mcp_tool_definition_after_reload(
    tmp_path,
    monkeypatch,
):
    """
    AgentToolView stores provider name, not Provider identity.

    After reload, the same view observes the replacement provider.
    """

    async def scenario():
        source = (
            tmp_path / "example.yaml"
        )

        write_mcp_config(
            source,
            name="example",
            generation="v1",
        )

        async def fake_connect(
            spec,
        ):
            generation = spec.args[0]

            tool_name = (
                "old_tool"
                if generation == "v1"
                else "new_tool"
            )

            session = FakeSession(
                generation=generation,
                tools=[
                    SimpleNamespace(
                        name=tool_name,
                        description=(
                            f"{generation} tool"
                        ),
                        inputSchema={
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    )
                ],
            )

            return make_provider(
                spec,
                generation=generation,
                stack=FakeStack(),
                session=session,
            )

        monkeypatch.setattr(
            mcp_backend,
            "connect",
            fake_connect,
        )

        runtime = ProviderRuntime(
            workspace_mcp_dir=tmp_path,
            workspace_local_dir=(
                tmp_path / "local"
            ),
            builtin_tools=(),
        )

        try:
            await runtime._reload_workspace_source(
                source,
                (1, 2),
            )

            view = AgentToolView(
                runtime,
                active_provider="example",
            )

            first = (
                await view.list_tools()
            )

            names_v1 = {
                tool.name
                for tool in first
            }

            assert (
                "old_tool"
                in names_v1
            )

            assert (
                "new_tool"
                not in names_v1
            )

            write_mcp_config(
                source,
                name="example",
                generation="v2",
            )

            await runtime._reload_workspace_source(
                source,
                (2, 2),
            )

            second = (
                await view.list_tools()
            )

            names_v2 = {
                tool.name
                for tool in second
            }

            assert (
                "old_tool"
                not in names_v2
            )

            assert (
                "new_tool"
                in names_v2
            )

        finally:
            await stop_all_workers(
                runtime
            )

    run(
        scenario()
    )


# ============================================================================
# Builtin MCP isolation
# ============================================================================


def test_workspace_mcp_cannot_override_builtin_provider(
    tmp_path,
    monkeypatch,
):
    """
    Builtin MCP providers are immutable.

    Workspace reload must reject a workspace provider with the same
    name without disturbing the builtin provider.
    """

    async def scenario():
        async def fake_connect(
            spec,
        ):
            stack = FakeStack()

            session = FakeSession(
                generation=spec.origin
            )

            return make_provider(
                spec,
                generation=spec.origin,
                stack=stack,
                session=session,
            )

        monkeypatch.setattr(
            mcp_backend,
            "connect",
            fake_connect,
        )

        runtime = ProviderRuntime(
            workspace_mcp_dir=tmp_path,
            workspace_local_dir=(
                tmp_path / "local"
            ),
            builtin_tools=(),
        )

        try:
            builtin_spec = ProviderSpec(
                name="example",
                kind=PROVIDER_KIND_MCP,
                command="builtin-mcp",
                source="<builtin>",
                origin="builtin",
            )

            builtin_worker = (
                await runtime._start_mcp_worker(
                    builtin_spec
                )
            )

            builtin_provider = (
                runtime.get_provider(
                    "example"
                )
            )

            assert builtin_provider is not None

            assert (
                builtin_provider.spec.origin
                == "builtin"
            )

            source = (
                tmp_path
                / "example.yaml"
            )

            write_mcp_config(
                source,
                name="example",
                generation="workspace",
            )

            with pytest.raises(
                ValueError,
                match=(
                    "cannot override builtin"
                ),
            ):
                await runtime._reload_workspace_source(
                    source,
                    (1, 2),
                )

            assert (
                runtime.get_provider(
                    "example"
                )
                is builtin_provider
            )

            assert (
                runtime._mcp_workers[
                    "example"
                ]
                is builtin_worker
            )

        finally:
            await stop_all_workers(
                runtime
            )

    run(
        scenario()
    )


# ============================================================================
# Strict 1:1 workspace MCP source contract
# ============================================================================


def test_workspace_mcp_file_must_map_to_exactly_one_provider():
    """
    Architectural contract:

        one workspace file <-> one MCP provider

    Builtin MCP configuration is exempt.

    Workspace files use the single-provider form:

        name: example
        command: ...

    The legacy multi-provider mapping is rejected.
    """

    config = {
        "mcp_servers": {
            "one": {
                "command": "fake-one",
            },
            "two": {
                "command": "fake-two",
            },
        }
    }

    with pytest.raises(
        ValueError,
        match="exactly one provider",
    ):
        mcp_backend.parse_workspace_config(
            config,
            Path(
                "/tmp/multi.yaml"
            ),
        )


# ============================================================================
# MCP source deletion isolation
# ============================================================================


def test_removing_one_mcp_source_does_not_touch_unrelated_provider(
    tmp_path,
    monkeypatch,
):
    """
    Removing one workspace MCP source must remove only its own provider.
    """

    async def scenario():
        alpha = (
            tmp_path / "alpha.yaml"
        )

        beta = (
            tmp_path / "beta.yaml"
        )

        write_mcp_config(
            alpha,
            name="alpha",
            generation="alpha",
        )

        write_mcp_config(
            beta,
            name="beta",
            generation="beta",
        )

        async def fake_connect(
            spec,
        ):
            return make_provider(
                spec,
                generation=spec.name,
                stack=FakeStack(),
                session=FakeSession(
                    generation=spec.name
                ),
            )

        monkeypatch.setattr(
            mcp_backend,
            "connect",
            fake_connect,
        )

        runtime = ProviderRuntime(
            workspace_mcp_dir=tmp_path,
            workspace_local_dir=(
                tmp_path / "local"
            ),
            builtin_tools=(),
        )

        try:
            await runtime._scan_workspace_mcps()

            alpha_before = (
                runtime.get_provider(
                    "alpha"
                )
            )

            beta_before = (
                runtime.get_provider(
                    "beta"
                )
            )

            assert alpha_before is not None
            assert beta_before is not None

            alpha.unlink()

            await runtime._scan_workspace_mcps()

            assert (
                runtime.get_provider(
                    "alpha"
                )
                is None
            )

            assert (
                runtime.get_provider(
                    "beta"
                )
                is beta_before
            )

        finally:
            await stop_all_workers(
                runtime
            )

    run(
        scenario()
    )
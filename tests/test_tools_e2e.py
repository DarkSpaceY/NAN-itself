from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from nan_itself.agent.engine import StepEngine
from nan_itself.agent.runtime import AgentRuntime
from nan_itself.agent.verbs import (
    DispatchVerb,
    ExecutionState,
)
from nan_itself.tools import mcp as mcp_backend
from nan_itself.tools.local import (
    LocalToolProvider,
)
from nan_itself.tools.provider import Provider
from nan_itself.tools.results import (
    text_result,
)
from nan_itself.tools.runtime import (
    ProviderRuntime,
)
from nan_itself.tools.spec import (
    PROVIDER_KIND_LOCAL,
    PROVIDER_KIND_MCP,
    ProviderSpec,
)
from nan_itself.tools.view import (
    AgentToolView,
)
from nan_itself.utils.llm import (
    ToolCall,
)


# ============================================================================
# Async helper
# ============================================================================


def run(coro):
    return asyncio.run(coro)


# ============================================================================
# Fake infrastructure
# ============================================================================


class FakeModules:
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
    def refresh(self):
        pass

    def catalog(self):
        return ()

    def names(self):
        return ()

    def activate(self, name):
        raise AssertionError(
            "Skills are not used in this test"
        )


class FakeLLM:
    model = "fake-model"
    provider = "fake-provider"

    def __init__(
        self,
        responses,
    ):
        self.responses = list(
            responses
        )
        self.requests = []

    async def generate_complete(
        self,
        request,
    ):
        self.requests.append(
            request
        )

        if not self.responses:
            raise AssertionError(
                "LLM received more requests than expected"
            )

        # Give background Subagent / provider work a chance to run.
        await asyncio.sleep(0)

        return self.responses.pop(0)


def make_tool_call(
    call_id,
    name,
    arguments,
):
    return ToolCall(
        id=call_id,
        name=name,
        arguments=arguments,
    )


def make_response(
    *,
    content=None,
    tool_calls=None,
):
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        model="fake-model",
        usage=None,
        provider="fake-provider",
        finish_reason="stop",
    )


def make_engine(
    *,
    llm,
    providers,
    runtime=None,
):
    if runtime is None:
        runtime = AgentRuntime(
            max_subagent_depth=3
        )

    return StepEngine(
        llm=llm,
        modules=FakeModules(),
        providers=providers,
        skills=FakeSkills(),
        agent_runtime=runtime,
    ), runtime


def make_context(
    runtime,
    *,
    task="tool-test",
):
    return runtime.create_root(
        world={
            "state": {
                "value": 1,
            }
        },
        task=task,
    )


# ============================================================================
# Local Tool source helpers
# ============================================================================


def write_local_tool(
    path: Path,
    *,
    body: str,
):
    path.write_text(
        body,
        encoding="utf-8",
    )


LOCAL_ADD_SOURCE = """
# @tool

class Calculator(LocalToolProvider):
    id = "calc"

    @tool
    def add(
        self,
        x: int,
        y: int,
    ) -> int:
        return x + y
"""


LOCAL_FAILURE_SOURCE = """
# @tool

class Calculator(LocalToolProvider):
    id = "calc"

    @tool
    def explode(
        self,
        message: str,
    ) -> str:
        raise RuntimeError(message)
"""


LOCAL_SLOW_SOURCE = """
# @tool

import asyncio

class Calculator(LocalToolProvider):
    id = "calc"

    @tool
    async def slow(
        self,
        seconds: float,
    ) -> str:
        await asyncio.sleep(seconds)
        return "finished"
"""


# ============================================================================
# Local Tool: provider/view level
# ============================================================================


def test_local_tool_can_be_discovered_routed_and_called(
    tmp_path,
):
    async def scenario():
        source = (
            tmp_path / "calc.py"
        )

        write_local_tool(
            source,
            body=LOCAL_ADD_SOURCE,
        )

        runtime = ProviderRuntime(
            workspace_local_dir=tmp_path,
            workspace_mcp_dir=(
                tmp_path / "mcps"
            ),
            builtin_tools=(),
        )

        try:
            await runtime._scan_workspace_locals()

            assert (
                runtime.provider_names()
                == ("calc",)
            )

            provider = (
                runtime.get_provider(
                    "calc"
                )
            )

            assert provider is not None

            assert (
                provider.spec.kind
                == PROVIDER_KIND_LOCAL
            )

            assert (
                set(provider.tools)
                == {"add"}
            )

            view = AgentToolView(
                runtime
            )

            # Before route only `route` is available.
            initial = (
                await view.list_tools()
            )

            assert [
                tool.name
                for tool in initial
            ] == ["route"]

            route_result = (
                await view.call_tool(
                    "route",
                    {
                        "provider_name": "calc",
                    },
                )
            )

            assert (
                "Provider 'calc' activated."
                in route_result.content[0].text
            )

            active = (
                await view.list_tools()
            )

            assert {
                tool.name
                for tool in active
            } == {
                "route",
                "add",
            }

            call_result = (
                await view.call_tool(
                    "add",
                    {
                        "x": 2,
                        "y": 3,
                    },
                )
            )

            assert (
                call_result.isError
                is False
            )

            assert (
                call_result.content[0].text
                == "5"
            )

        finally:
            await runtime.stop()

    run(
        scenario()
    )


def test_local_tool_cannot_be_called_before_provider_is_routed(
    tmp_path,
):
    async def scenario():
        source = (
            tmp_path / "calc.py"
        )

        write_local_tool(
            source,
            body=LOCAL_ADD_SOURCE,
        )

        runtime = ProviderRuntime(
            workspace_local_dir=tmp_path,
            workspace_mcp_dir=(
                tmp_path / "mcps"
            ),
            builtin_tools=(),
        )

        try:
            await runtime._scan_workspace_locals()

            view = AgentToolView(
                runtime
            )

            result = (
                await view.call_tool(
                    "add",
                    {
                        "x": 1,
                        "y": 2,
                    },
                )
            )

            assert (
                result.isError
                is True
            )

            assert (
                "No tool provider is active"
                in result.content[0].text
            )

        finally:
            await runtime.stop()

    run(
        scenario()
    )


def test_local_tool_unknown_tool_returns_error(
    tmp_path,
):
    async def scenario():
        source = (
            tmp_path / "calc.py"
        )

        write_local_tool(
            source,
            body=LOCAL_ADD_SOURCE,
        )

        runtime = ProviderRuntime(
            workspace_local_dir=tmp_path,
            workspace_mcp_dir=(
                tmp_path / "mcps"
            ),
            builtin_tools=(),
        )

        try:
            await runtime._scan_workspace_locals()

            view = AgentToolView(
                runtime
            )

            await view.call_tool(
                "route",
                {
                    "provider_name": "calc",
                },
            )

            result = (
                await view.call_tool(
                    "missing",
                    {},
                )
            )

            assert (
                result.isError
                is True
            )

            assert (
                "not available"
                in result.content[0].text
            )

        finally:
            await runtime.stop()

    run(
        scenario()
    )


# ============================================================================
# Local Tool: schema / validation
# ============================================================================


def test_local_tool_invalid_arguments_are_returned_as_tool_error(
    tmp_path,
):
    async def scenario():
        source = (
            tmp_path / "calc.py"
        )

        write_local_tool(
            source,
            body=LOCAL_ADD_SOURCE,
        )

        runtime = ProviderRuntime(
            workspace_local_dir=tmp_path,
            workspace_mcp_dir=(
                tmp_path / "mcps"
            ),
            builtin_tools=(),
        )

        try:
            await runtime._scan_workspace_locals()

            view = AgentToolView(
                runtime
            )

            await view.call_tool(
                "route",
                {
                    "provider_name": "calc",
                },
            )

            result = (
                await view.call_tool(
                    "add",
                    {
                        "x": "not-an-int",
                        "y": 3,
                    },
                )
            )

            assert (
                result.isError
                is True
            )

            assert (
                "Invalid arguments for tool"
                in result.content[0].text
            )

        finally:
            await runtime.stop()

    run(
        scenario()
    )


def test_local_tool_handler_exception_becomes_tool_error(
    tmp_path,
):
    async def scenario():
        source = (
            tmp_path / "calc.py"
        )

        write_local_tool(
            source,
            body=LOCAL_FAILURE_SOURCE,
        )

        runtime = ProviderRuntime(
            workspace_local_dir=tmp_path,
            workspace_mcp_dir=(
                tmp_path / "mcps"
            ),
            builtin_tools=(),
        )

        try:
            await runtime._scan_workspace_locals()

            view = AgentToolView(
                runtime
            )

            await view.call_tool(
                "route",
                {
                    "provider_name": "calc",
                },
            )

            result = (
                await view.call_tool(
                    "explode",
                    {
                        "message": "boom",
                    },
                )
            )

            assert (
                result.isError
                is True
            )

            assert (
                "RuntimeError: boom"
                in result.content[0].text
            )

        finally:
            await runtime.stop()

    run(
        scenario()
    )


# ============================================================================
# Local Tool: full StepEngine path
# ============================================================================


def test_step_engine_can_route_to_local_tool_then_finish(
    tmp_path,
):
    async def scenario():
        source = (
            tmp_path / "calc.py"
        )

        write_local_tool(
            source,
            body=LOCAL_ADD_SOURCE,
        )

        providers = ProviderRuntime(
            workspace_local_dir=tmp_path,
            workspace_mcp_dir=(
                tmp_path / "mcps"
            ),
            builtin_tools=(),
        )

        await providers._scan_workspace_locals()

        llm = FakeLLM(
            [
                # --------------------------------------------------
                # Step 1:
                # Provider is not active yet.
                # --------------------------------------------------

                make_response(
                    tool_calls=[
                        make_tool_call(
                            "route-1",
                            "route",
                            {
                                "provider_name": "calc",
                            },
                        )
                    ]
                ),

                # --------------------------------------------------
                # Step 2:
                # Provider block is now visible.
                # --------------------------------------------------

                make_response(
                    tool_calls=[
                        make_tool_call(
                            "add-1",
                            "add",
                            {
                                "x": 7,
                                "y": 8,
                            },
                        )
                    ]
                ),

                # --------------------------------------------------
                # Step 3:
                # Final answer.
                # --------------------------------------------------

                make_response(
                    content="15"
                ),
            ]
        )

        engine, agent_runtime = (
            make_engine(
                llm=llm,
                providers=providers,
            )
        )

        context = make_context(
            agent_runtime
        )

        try:
            result = await engine.execute(
                context=context,
                user_input=(
                    "calculate 7 + 8"
                ),
                persona="test",
            )

            assert (
                result.content
                == "15"
            )

            assert (
                len(llm.requests)
                == 3
            )

            first_tools = {
                tool.name
                for tool
                in llm.requests[0].tools
            }

            second_tools = {
                tool.name
                for tool
                in llm.requests[1].tools
            }

            assert (
                "route"
                in first_tools
            )

            assert (
                "add"
                not in first_tools
            )

            assert (
                "add"
                in second_tools
            )

            # The actual provider call happened before the final
            # model response.
            tool_messages = [
                message
                for request
                in llm.requests
                for message
                in request.messages
                if getattr(
                    message,
                    "role",
                    None,
                )
                == "tool"
            ]

            assert tool_messages

            assert any(
                "15"
                in (
                    message.content
                    or ""
                )
                for message
                in tool_messages
            )

        finally:
            await agent_runtime.shutdown()
            await providers.stop()

    run(
        scenario()
    )


# ============================================================================
# MCP helpers
# ============================================================================


class FakeMCPStack:
    def __init__(self):
        self.owner = (
            asyncio.current_task()
        )
        self.closed_by = None

    async def aclose(self):
        self.closed_by = (
            asyncio.current_task()
        )


class FakeMCPSession:
    def __init__(
        self,
        *,
        value="mcp-result",
        delay=0.0,
    ):
        self.value = value
        self.delay = delay

        self.calls = []

    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="lookup",
                    description=(
                        "Lookup something."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                            }
                        },
                        "required": [
                            "key"
                        ],
                        "additionalProperties": False,
                    },
                )
            ]
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

        if self.delay > 0:
            await asyncio.sleep(
                self.delay
            )

        return text_result(
            f"{self.value}:{arguments['key']}"
        )


def write_mcp_source(
    path: Path,
    *,
    name="example",
):
    path.write_text(
        "\n".join(
            [
                f"name: {name}",
                "command: fake-mcp",
            ]
        ),
        encoding="utf-8",
    )


# ============================================================================
# MCP Tool: provider/view level
# ============================================================================


def test_mcp_tool_can_be_discovered_routed_and_called(
    tmp_path,
    monkeypatch,
):
    async def scenario():
        source = (
            tmp_path / "example.yaml"
        )

        write_mcp_source(
            source
        )

        created = {}

        async def fake_connect(
            spec,
        ):
            stack = FakeMCPStack()

            session = FakeMCPSession(
                value="lookup"
            )

            provider = Provider(
                spec=spec,
                stack=stack,
                session=session,
                tools={
                    tool.name: tool
                    for tool
                    in (
                        await session.list_tools()
                    ).tools
                },
            )

            created["provider"] = provider

            return provider

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

            provider = (
                runtime.get_provider(
                    "example"
                )
            )

            assert provider is not None

            assert (
                provider.spec.kind
                == PROVIDER_KIND_MCP
            )

            view = AgentToolView(
                runtime
            )

            initial = (
                await view.list_tools()
            )

            assert [
                tool.name
                for tool in initial
            ] == ["route"]

            route_result = (
                await view.call_tool(
                    "route",
                    {
                        "provider_name": "example",
                    },
                )
            )

            assert (
                "Provider 'example' activated."
                in route_result.content[0].text
            )

            active = (
                await view.list_tools()
            )

            assert {
                tool.name
                for tool in active
            } == {
                "route",
                "lookup",
            }

            call_result = (
                await view.call_tool(
                    "lookup",
                    {
                        "key": "abc",
                    },
                )
            )

            assert (
                call_result.content[0].text
                == "lookup:abc"
            )

            assert (
                created["provider"]
                .session.calls
                == [
                    (
                        "lookup",
                        {
                            "key": "abc",
                        },
                    )
                ]
            )

        finally:
            await runtime.stop()

    run(
        scenario()
    )


# ============================================================================
# MCP Tool: full StepEngine path
# ============================================================================


def test_step_engine_can_route_to_mcp_tool_then_finish(
    tmp_path,
    monkeypatch,
):
    async def scenario():
        source = (
            tmp_path / "example.yaml"
        )

        write_mcp_source(
            source
        )

        sessions = {}

        async def fake_connect(
            spec,
        ):
            stack = FakeMCPStack()

            session = FakeMCPSession(
                value="mcp",
            )

            sessions["example"] = (
                session
            )

            tools = {
                tool.name: tool
                for tool
                in (
                    await session.list_tools()
                ).tools
            }

            return Provider(
                spec=spec,
                stack=stack,
                session=session,
                tools=tools,
            )

        monkeypatch.setattr(
            mcp_backend,
            "connect",
            fake_connect,
        )

        providers = ProviderRuntime(
            workspace_mcp_dir=tmp_path,
            workspace_local_dir=(
                tmp_path / "local"
            ),
            builtin_tools=(),
        )

        await providers._scan_workspace_mcps()

        llm = FakeLLM(
            [
                make_response(
                    tool_calls=[
                        make_tool_call(
                            "route-1",
                            "route",
                            {
                                "provider_name": "example",
                            },
                        )
                    ]
                ),
                make_response(
                    tool_calls=[
                        make_tool_call(
                            "lookup-1",
                            "lookup",
                            {
                                "key": "hello",
                            },
                        )
                    ]
                ),
                make_response(
                    content="mcp:hello"
                ),
            ]
        )

        engine, agent_runtime = (
            make_engine(
                llm=llm,
                providers=providers,
            )
        )

        context = make_context(
            agent_runtime
        )

        try:
            result = await engine.execute(
                context=context,
                user_input=(
                    "lookup hello"
                ),
                persona="test",
            )

            assert (
                result.content
                == "mcp:hello"
            )

            assert (
                sessions["example"].calls
                == [
                    (
                        "lookup",
                        {
                            "key": "hello",
                        },
                    )
                ]
            )

            assert (
                len(llm.requests)
                == 3
            )

            first_tools = {
                tool.name
                for tool
                in llm.requests[0].tools
            }

            second_tools = {
                tool.name
                for tool
                in llm.requests[1].tools
            }

            assert (
                "route"
                in first_tools
            )

            assert (
                "lookup"
                not in first_tools
            )

            assert (
                "lookup"
                in second_tools
            )

            tool_messages = [
                message
                for request
                in llm.requests
                for message
                in request.messages
                if getattr(
                    message,
                    "role",
                    None,
                )
                == "tool"
            ]

            assert tool_messages

            assert any(
                "mcp:hello"
                in (
                    message.content
                    or ""
                )
                for message
                in tool_messages
            )

        finally:
            await agent_runtime.shutdown()
            await providers.stop()

    run(
        scenario()
    )


# ============================================================================
# MCP Tool: timeout
# ============================================================================


def test_mcp_tool_timeout_is_returned_as_error(
    tmp_path,
    monkeypatch,
):
    async def scenario():
        source = (
            tmp_path / "example.yaml"
        )

        write_mcp_source(
            source
        )

        async def fake_connect(
            spec,
        ):
            stack = FakeMCPStack()

            session = FakeMCPSession(
                value="slow",
                delay=0.1,
            )

            tools = {
                tool.name: tool
                for tool
                in (
                    await session.list_tools()
                ).tools
            }

            return Provider(
                spec=spec,
                stack=stack,
                session=session,
                tools=tools,
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
            tool_timeout=0.01,
        )

        try:
            await runtime._scan_workspace_mcps()

            view = AgentToolView(
                runtime
            )

            await view.call_tool(
                "route",
                {
                    "provider_name": "example",
                },
            )

            result = (
                await view.call_tool(
                    "lookup",
                    {
                        "key": "timeout",
                    },
                )
            )

            assert (
                result.isError
                is True
            )

            assert (
                "timed out"
                in result.content[0].text
            )

        finally:
            await runtime.stop()

    run(
        scenario()
    )


# ============================================================================
# Provider isolation
# ============================================================================


def test_agent_tool_view_active_provider_is_per_view():
    async def scenario():
        class Runtime:
            def __init__(self):
                self.providers = {
                    "a": SimpleNamespace(
                        tools={
                            "tool_a": SimpleNamespace(
                                name="tool_a",
                                description="a",
                                inputSchema={
                                    "type": "object",
                                    "properties": {},
                                    "additionalProperties": False,
                                },
                            )
                        }
                    ),
                    "b": SimpleNamespace(
                        tools={
                            "tool_b": SimpleNamespace(
                                name="tool_b",
                                description="b",
                                inputSchema={
                                    "type": "object",
                                    "properties": {},
                                    "additionalProperties": False,
                                },
                            )
                        }
                    ),
                }

            def provider_names(self):
                return tuple(
                    self.providers
                )

            def get_provider(
                self,
                name,
            ):
                return self.providers.get(
                    name
                )

            async def refresh_provider_tools(
                self,
                name,
            ):
                pass

        runtime = Runtime()

        first = AgentToolView(
            runtime
        )

        second = AgentToolView(
            runtime
        )

        await first.call_tool(
            "route",
            {
                "provider_name": "a",
            },
        )

        await second.call_tool(
            "route",
            {
                "provider_name": "b",
            },
        )

        first_tools = {
            tool.name
            for tool
            in await first.list_tools()
        }

        second_tools = {
            tool.name
            for tool
            in await second.list_tools()
        }

        assert (
            "tool_a"
            in first_tools
        )

        assert (
            "tool_b"
            not in first_tools
        )

        assert (
            "tool_b"
            in second_tools
        )

        assert (
            "tool_a"
            not in second_tools
        )

    run(
        scenario()
    )


# ============================================================================
# Tool definition refresh
# ============================================================================

def test_mcp_view_refreshes_live_tool_definition(
    tmp_path,
    monkeypatch,
):
    """
    AgentToolView must refresh the tool definition table from the
    currently active MCP session.

    This test does NOT reload the MCP provider generation itself.
    It verifies the live `list_tools()` refresh contract.

        generation 1
            -> old_tool

        refresh

        generation 2
            -> new_tool
    """

    async def scenario():
        source = (
            tmp_path / "example.yaml"
        )

        write_mcp_source(
            source
        )

        generation = {
            "value": 1,
        }

        class RefreshableSession:
            def __init__(self):
                self.calls = []

            async def list_tools(
                self,
            ):
                name = (
                    "old_tool"
                    if generation["value"] == 1
                    else "new_tool"
                )

                return SimpleNamespace(
                    tools=[
                        SimpleNamespace(
                            name=name,
                            description=name,
                            inputSchema={
                                "type": "object",
                                "properties": {},
                                "additionalProperties": False,
                            },
                        )
                    ]
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

                return text_result(
                    name
                )

        async def fake_connect(
            spec,
        ):
            stack = FakeMCPStack()

            session = RefreshableSession()

            tools_result = (
                await session.list_tools()
            )

            provider = Provider(
                spec=spec,
                stack=stack,
                session=session,
                tools={
                    tool.name: tool
                    for tool in tools_result.tools
                },
            )

            return provider

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

            provider = (
                runtime.get_provider(
                    "example"
                )
            )

            assert provider is not None

            view = AgentToolView(
                runtime,
                active_provider="example",
            )

            # ------------------------------------------------------
            # Generation 1.
            # ------------------------------------------------------

            first = (
                await view.list_tools()
            )

            assert {
                tool.name
                for tool in first
            } == {
                "route",
                "old_tool",
            }

            # ------------------------------------------------------
            # Live MCP session now reports a different tool table.
            # ------------------------------------------------------

            generation["value"] = 2

            second = (
                await view.list_tools()
            )

            assert {
                tool.name
                for tool in second
            } == {
                "route",
                "new_tool",
            }

            assert (
                "old_tool"
                not in {
                    tool.name
                    for tool in second
                }
            )

        finally:
            await runtime.stop()

    run(
        scenario()
    )


# ============================================================================
# Composite E2E: Skill + Tool inside Subagent
# ============================================================================


def test_subagent_can_activate_skill_route_tool_and_report():
    """
    Full Subagent capability chain:

        dispatch_subagent
            ->
        child StepEngine
            ->
        activate_skill
            ->
        route
            ->
        add
            ->
        final result
            ->
        parent formats report
    """

    class CalculatorProvider:
        def __init__(self):
            self.tools = {
                "add": SimpleNamespace(
                    name="add",
                    description="Add two numbers.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "number",
                            },
                            "y": {
                                "type": "number",
                            },
                        },
                        "required": [
                            "x",
                            "y",
                        ],
                        "additionalProperties": False,
                    },
                )
            }

    class Providers:
        def __init__(self):
            self.providers = {
                "calc": CalculatorProvider(),
            }

            self.calls = []

        def provider_names(self):
            return tuple(
                self.providers
            )

        def get_provider(
            self,
            name,
        ):
            return self.providers.get(
                name
            )

        async def refresh_provider_tools(
            self,
            name,
        ):
            pass

        async def call_tool(
            self,
            provider_name,
            tool_name,
            arguments,
        ):
            self.calls.append(
                (
                    provider_name,
                    tool_name,
                    dict(arguments),
                )
            )

            return text_result(
                str(
                    arguments["x"]
                    + arguments["y"]
                )
            )

    class Skill:
        def __init__(self):
            self.name = "research"
            self.description = (
                "Research skill"
            )
            self.instructions = (
                "Work carefully and verify results."
            )
            self.generation = 1

            self.metadata = SimpleNamespace(
                name="research",
                description="Research skill",
                origin="workspace",
                source=None,
                frontmatter={},
            )

            self.resources = []
            self.scripts = ()
            self.references = ()
            self.assets = ()

    class Skills:
        def __init__(self):
            self.skill = Skill()
            self.activate_calls = []

        def catalog(self):
            return (
                self.skill.metadata,
            )

        def names(self):
            return (
                "research",
            )

        def activate(
            self,
            name,
        ):
            self.activate_calls.append(
                name
            )

            assert (
                name
                == "research"
            )

            return self.skill

    class Modules:
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

    class LLM:
        model = "fake-model"
        provider = "fake-provider"

        def __init__(self):
            self.requests = []
            self.responses = []

        async def generate_complete(
            self,
            request,
        ):
            self.requests.append(
                request
            )

            await asyncio.sleep(0)

            if not self.responses:
                raise AssertionError(
                    "Child LLM requested more responses "
                    "than expected."
                )

            return self.responses.pop(0)

    async def scenario():
        modules = Modules()
        providers = Providers()
        skills = Skills()

        runtime = AgentRuntime(
            max_subagent_depth=3
        )

        child_llm = LLM()

        # ----------------------------------------------------------
        # Child model sequence:
        #
        #   step 1 -> activate_skill
        #   step 2 -> route
        #   step 3 -> add
        #   step 4 -> final
        # ----------------------------------------------------------

        child_llm.responses = [
            make_response(
                tool_calls=[
                    make_tool_call(
                        "skill-1",
                        "activate_skill",
                        {
                            "name": "research",
                        },
                    )
                ]
            ),
            make_response(
                tool_calls=[
                    make_tool_call(
                        "route-1",
                        "route",
                        {
                            "provider_name": "calc",
                        },
                    )
                ]
            ),
            make_response(
                tool_calls=[
                    make_tool_call(
                        "add-1",
                        "add",
                        {
                            "x": 4,
                            "y": 5,
                        },
                    )
                ]
            ),
            make_response(
                content="child complete",
            ),
        ]

        child_engine = StepEngine(
            llm=child_llm,
            modules=modules,
            providers=providers,
            skills=skills,
            agent_runtime=runtime,
        )

        parent = runtime.create_root(
            world={
                "state": {
                    "value": 1,
                }
            },
            task="parent",
        )

        observed = {}

        # ----------------------------------------------------------
        # This is the actual engine.execute contract required by
        # DispatchVerb.
        #
        # DispatchVerb creates a worker internally that invokes:
        #
        #     engine.execute(
        #         context=...,
        #         user_input=...,
        #         persona=...
        #     )
        #
        # Therefore this function must accept exactly that interface.
        # ----------------------------------------------------------

        async def execute_child(
            *,
            context,
            user_input,
            persona,
        ):
            observed["context"] = context

            return await child_engine.execute(
                context=context,
                user_input=user_input,
                persona=persona,
            )

        engine = SimpleNamespace(
            agent_runtime=runtime,
            skills=skills,
            execute=execute_child,
        )

        state = ExecutionState(
            active_skill=None,
            persona="child persona",
        )

        try:
            dispatch_result = (
                await DispatchVerb().execute(
                    call=SimpleNamespace(
                        arguments={
                            "task": (
                                "research and calculate"
                            )
                        }
                    ),
                    context=parent,
                    state=state,
                    engine=engine,
                )
            )

            assert (
                "Subagent dispatched."
                in dispatch_result
            )

            assert (
                len(state.children)
                == 1
            )

            child = state.children[0]

            child_result = (
                await child.handle.wait()
            )

            assert (
                child_result.content
                == "child complete"
            )

            # ------------------------------------------------------
            # Skill activation.
            # ------------------------------------------------------

            assert (
                skills.activate_calls
                == ["research"]
            )

            # ------------------------------------------------------
            # Tool execution.
            # ------------------------------------------------------

            assert (
                providers.calls
                == [
                    (
                        "calc",
                        "add",
                        {
                            "x": 4,
                            "y": 5,
                        },
                    )
                ]
            )

            # ------------------------------------------------------
            # Child context.
            # ------------------------------------------------------

            child_context = (
                observed["context"]
            )

            assert (
                child_context.depth
                == 1
            )

            assert (
                child_context.parent_hash
                == parent.agent_hash
            )

            assert (
                child_context.world
                is parent.world
            )

            # The child inherited the current execution Skill
            # only because DispatchVerb explicitly supplied it.
            #
            # In this particular test the parent had no Skill, so
            # the child starts without one and activates it itself.
            assert (
                child_context.skill
                is None
            )

            # ------------------------------------------------------
            # Child really executed four model steps.
            # ------------------------------------------------------

            assert (
                len(child_llm.requests)
                == 4
            )

            # ------------------------------------------------------
            # The Skill becomes visible from the next step.
            # ------------------------------------------------------

            first_tools = {
                tool.name
                for tool in child_llm.requests[0].tools
            }

            second_tools = {
                tool.name
                for tool in child_llm.requests[1].tools
            }

            third_tools = {
                tool.name
                for tool in child_llm.requests[2].tools
            }

            assert (
                "activate_skill"
                in first_tools
            )

            assert (
                "route"
                in second_tools
            )

            assert (
                "research"
                in (
                    child_llm.requests[1]
                    .messages[0]
                    .content
                )
            )

            assert (
                "add"
                not in second_tools
            )

            assert (
                "add"
                in third_tools
            )

            # ------------------------------------------------------
            # Report.
            # ------------------------------------------------------

            from nan_itself.agent.reports import (
                format_child_report,
            )

            report = (
                await format_child_report(
                    child
                )
            )

            assert (
                "[Subagent Report]"
                in report
            )

            assert (
                "id:"
                in report
            )

            assert (
                "task: research and calculate"
                in report
            )

            assert (
                "status: completed"
                in report
            )

            assert (
                "child complete"
                in report
            )

        finally:
            await runtime.shutdown()

    run(
        scenario()
    )


# ============================================================================
# Composite E2E: Parent Tool + Subagent + Report
# ============================================================================


def test_parent_can_use_tool_and_subagent_before_final_answer():
    """
    Full Main Agent orchestration:

        parent
          |
          +--> route
          |
          +--> add
          |
          +--> dispatch_subagent
                  |
                  +--> child StepEngine
                  |
                  +--> child final result
          |
          +--> [Subagent Report]
          |
          +--> final answer

    Parent and child are real StepEngine executions.
    They share the same AgentRuntime but get independent
    AgentToolView instances.
    """

    class ProviderRuntimeDouble:
        def __init__(self):
            self.providers = {
                "calc": SimpleNamespace(
                    tools={
                        "add": SimpleNamespace(
                            name="add",
                            description="Add numbers.",
                            inputSchema={
                                "type": "object",
                                "properties": {
                                    "x": {
                                        "type": "number",
                                    },
                                    "y": {
                                        "type": "number",
                                    },
                                },
                                "required": [
                                    "x",
                                    "y",
                                ],
                                "additionalProperties": False,
                            },
                        )
                    }
                )
            }

            self.calls = []

        def provider_names(self):
            return (
                "calc",
            )

        def get_provider(
            self,
            name,
        ):
            return self.providers.get(
                name
            )

        async def refresh_provider_tools(
            self,
            name,
        ):
            return None

        async def call_tool(
            self,
            provider_name,
            tool_name,
            arguments,
        ):
            self.calls.append(
                (
                    provider_name,
                    tool_name,
                    dict(arguments),
                )
            )

            return text_result(
                str(
                    arguments["x"]
                    + arguments["y"]
                )
            )

    class Modules:
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

    class Skills:
        def refresh(self):
            pass

        def catalog(self):
            return ()

        def names(self):
            return ()

        def activate(
            self,
            name,
        ):
            raise AssertionError(
                "No Skill should be activated in this test"
            )

    class LLM:
        model = "fake-model"
        provider = "fake-provider"

        def __init__(self):
            self.requests = []

            self.parent_steps = 0
            self.child_calls = 0

        async def generate_complete(
            self,
            request,
        ):
            self.requests.append(
                request
            )

            # Simulate the asynchronous boundary of a real LLM request.
            await asyncio.sleep(0)

            user_inputs = [
                message.content
                for message in request.messages
                if getattr(
                    message,
                    "role",
                    None,
                )
                == "user"
                and isinstance(
                    getattr(
                        message,
                        "content",
                        None,
                    ),
                    str,
                )
            ]

            # ----------------------------------------------------------
            # Child detection.
            #
            # Do NOT search for "child task" anywhere in the complete
            # prompt, because the parent's DispatchVerb result itself
            # legitimately contains the child task text.
            #
            # A real child execution starts with:
            #
            #     user_input == "child task"
            #
            # while the parent starts with:
            #
            #     user_input == "parent task"
            # ----------------------------------------------------------

            if (
                user_inputs
                and user_inputs[0]
                == "child task"
            ):
                self.child_calls += 1

                return make_response(
                    content="child completed"
                )

            # ----------------------------------------------------------
            # Parent must start from its own user input.
            # ----------------------------------------------------------

            assert (
                user_inputs
                and user_inputs[0]
                == "parent task"
            ), (
                "Unexpected LLM request:\n"
                + "\n".join(user_inputs)
            )

            has_report = any(
                (
                    "[Subagent Report]"
                    in (
                        message.content
                        or ""
                    )
                )
                for message in request.messages
                if getattr(
                    message,
                    "role",
                    None,
                )
                == "user"
                and isinstance(
                    getattr(
                        message,
                        "content",
                        None,
                    ),
                    str,
                )
            )

            # ----------------------------------------------------------
            # Once the report has arrived, finish.
            # ----------------------------------------------------------

            if has_report:
                return make_response(
                    content=(
                        "completed orchestration"
                    )
                )

            self.parent_steps += 1

            # ----------------------------------------------------------
            # Parent step 1:
            # route provider.
            # ----------------------------------------------------------

            if (
                self.parent_steps
                == 1
            ):
                return make_response(
                    tool_calls=[
                        make_tool_call(
                            "route-1",
                            "route",
                            {
                                "provider_name": "calc",
                            },
                        )
                    ]
                )

            # ----------------------------------------------------------
            # Parent step 2:
            # call provider tool.
            # ----------------------------------------------------------

            if (
                self.parent_steps
                == 2
            ):
                return make_response(
                    tool_calls=[
                        make_tool_call(
                            "add-1",
                            "add",
                            {
                                "x": 10,
                                "y": 20,
                            },
                        )
                    ]
                )

            # ----------------------------------------------------------
            # Parent step 3:
            # dispatch exactly one child.
            # ----------------------------------------------------------

            if (
                self.parent_steps
                == 3
            ):
                return make_response(
                    tool_calls=[
                        make_tool_call(
                            "dispatch-1",
                            "dispatch_subagent",
                            {
                                "task": "child task",
                            },
                        )
                    ]
                )

            # ----------------------------------------------------------
            # Parent step 4:
            # yield through the real SleepVerb.
            #
            # The child gets a chance to complete while the parent is
            # yielding. The following StepEngine iteration will call
            # collect_finished_children() and inject the report.
            # ----------------------------------------------------------

            if (
                self.parent_steps
                == 4
            ):
                return make_response(
                    tool_calls=[
                        make_tool_call(
                            "sleep-1",
                            "sleep",
                            {
                                "seconds": 0,
                            },
                        )
                    ]
                )

            raise AssertionError(
                "Parent advanced without receiving "
                "the Subagent report."
            )

    async def scenario():
        modules = Modules()
        skills = Skills()
        providers = (
            ProviderRuntimeDouble()
        )

        runtime = AgentRuntime(
            max_subagent_depth=3
        )

        llm = LLM()

        engine = StepEngine(
            llm=llm,
            modules=modules,
            providers=providers,
            skills=skills,
            agent_runtime=runtime,
        )

        parent = runtime.create_root(
            world={
                "state": {
                    "value": 1,
                }
            },
            task="parent task",
        )

        try:
            result = await engine.execute(
                context=parent,
                user_input="parent task",
                persona="parent persona",
            )

            # ------------------------------------------------------
            # Parent final response.
            # ------------------------------------------------------

            assert (
                result.content
                == "completed orchestration"
            )

            # ------------------------------------------------------
            # Tool path really happened.
            # ------------------------------------------------------

            assert (
                providers.calls
                == [
                    (
                        "calc",
                        "add",
                        {
                            "x": 10,
                            "y": 20,
                        },
                    )
                ]
            )

            # ------------------------------------------------------
            # Exactly one child.
            # ------------------------------------------------------

            children = [
                agent
                for agent in runtime.agents()
                if (
                    agent.parent_hash
                    == parent.agent_hash
                )
            ]

            assert (
                len(children)
                == 1
            )

            child = children[0]

            assert (
                child.task
                == "child task"
            )

            assert (
                child.depth
                == 1
            )

            assert (
                child.world
                is parent.world
            )

            # ------------------------------------------------------
            # Child actually executed.
            # ------------------------------------------------------

            assert (
                llm.child_calls
                == 1
            )

            # Parent performed:
            #
            #   route
            #   add
            #   dispatch
            #   sleep
            #
            assert (
                llm.parent_steps
                == 4
            )

            # ------------------------------------------------------
            # Parent received the Subagent Report.
            # ------------------------------------------------------

            report_requests = []

            for request in llm.requests:
                has_report = any(
                    (
                        "[Subagent Report]"
                        in (
                            message.content
                            or ""
                        )
                    )
                    for message in request.messages
                    if getattr(
                        message,
                        "role",
                        None,
                    )
                    == "user"
                    and isinstance(
                        getattr(
                            message,
                            "content",
                            None,
                        ),
                        str,
                    )
                )

                if has_report:
                    report_requests.append(
                        request
                    )

            assert (
                report_requests
            )

            final_report_text = "\n".join(
                message.content
                for message in (
                    report_requests[-1].messages
                )
                if getattr(
                    message,
                    "role",
                    None,
                )
                == "user"
                and isinstance(
                    getattr(
                        message,
                        "content",
                        None,
                    ),
                    str,
                )
            )

            assert (
                "[Subagent Report]"
                in final_report_text
            )

            assert (
                "child completed"
                in final_report_text
            )

            # No fire-and-forget Subagent remains.
            assert (
                runtime.active_subagent_count
                == 0
            )

        finally:
            await runtime.shutdown()

    run(
        scenario()
    )
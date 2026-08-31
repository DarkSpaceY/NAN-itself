from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from nan_itself.agent.engine import StepEngine
from nan_itself.agent.runtime import AgentRuntime
from nan_itself.agent.verbs import (
    ActivateSkillVerb,
    DispatchVerb,
    ExecutionState,
    SleepVerb,
)
from nan_itself.skills import UnknownSkillError
from nan_itself.tools import mcp as mcp_backend
from nan_itself.tools.provider import Provider
from nan_itself.tools.results import text_result
from nan_itself.tools.runtime import ProviderRuntime
from nan_itself.tools.view import AgentToolView
from nan_itself.utils.llm import ToolCall


# ============================================================================
# Async helpers
# ============================================================================


def run(coro):
    return asyncio.run(coro)


# ============================================================================
# Generic test doubles
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
    def __init__(
        self,
        skills=None,
    ):
        self._skills = dict(
            skills or {}
        )

        self.activate_calls = []
        self.refresh_calls = 0

    def refresh(self):
        self.refresh_calls += 1

    def catalog(self):
        return tuple(
            getattr(
                skill,
                "metadata",
                None,
            )
            for skill in self._skills.values()
            if getattr(
                skill,
                "metadata",
                None,
            ) is not None
        )

    def names(self):
        return tuple(
            self._skills
        )

    def activate(
        self,
        name,
    ):
        self.activate_calls.append(
            name
        )

        if name not in self._skills:
            raise UnknownSkillError(
                name
            )

        return self._skills[name]


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

        await asyncio.sleep(0)

        if not self.responses:
            raise AssertionError(
                "FakeLLM received more requests than expected."
            )

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
    providers=None,
    skills=None,
    runtime=None,
):
    if runtime is None:
        runtime = AgentRuntime(
            max_subagent_depth=3
        )

    if providers is None:
        providers = FakeProviders()

    if skills is None:
        skills = FakeSkills()

    engine = StepEngine(
        llm=llm,
        modules=FakeModules(),
        providers=providers,
        skills=skills,
        agent_runtime=runtime,
    )

    return (
        engine,
        runtime,
        skills,
    )


def unique_tool_messages(
    llm,
):
    """
    Return each actual tool execution exactly once.

    StepEngine carries the complete accumulated message history into
    later LLM requests. Therefore the same tool message may appear in
    multiple requests.

    `tool_call_id` identifies the actual execution and is the correct
    deduplication key for these tests.
    """
    by_id = {}

    for request in llm.requests:
        for message in request.messages:
            if (
                getattr(
                    message,
                    "role",
                    None,
                )
                != "tool"
            ):
                continue

            tool_call_id = getattr(
                message,
                "tool_call_id",
                None,
            )

            if tool_call_id is None:
                continue

            by_id[
                tool_call_id
            ] = message

    return list(
        by_id.values()
    )


# ============================================================================
# Local tool source helpers
# ============================================================================


def write_local_tool(
    path: Path,
    body: str,
):
    path.write_text(
        body,
        encoding="utf-8",
    )


LOCAL_CALCULATOR = """
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


LOCAL_FAILURE = """
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


LOCAL_SLOW = """
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


async def make_local_runtime(
    tmp_path,
    body,
):
    source = (
        tmp_path / "calc.py"
    )

    write_local_tool(
        source,
        body,
    )

    runtime = ProviderRuntime(
        workspace_local_dir=tmp_path,
        workspace_mcp_dir=(
            tmp_path / "mcps"
        ),
        builtin_tools=(),
    )

    await runtime._scan_workspace_locals()

    return runtime


# ============================================================================
# SleepVerb failure paths
# ============================================================================


def test_sleep_rejects_missing_seconds():
    async def scenario():
        runtime = AgentRuntime()

        state = ExecutionState(
            active_skill=None,
            persona="test",
        )

        result = await SleepVerb().execute(
            call=SimpleNamespace(
                arguments={}
            ),
            context=None,
            state=state,
            engine=SimpleNamespace(
                agent_runtime=runtime
            ),
        )

        assert result == (
            "sleep requires 'seconds'."
        )

        await runtime.shutdown()

    run(
        scenario()
    )


def test_sleep_rejects_non_numeric_seconds():
    async def scenario():
        runtime = AgentRuntime()

        state = ExecutionState(
            active_skill=None,
            persona="test",
        )

        result = await SleepVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "seconds": "later",
                }
            ),
            context=None,
            state=state,
            engine=SimpleNamespace(
                agent_runtime=runtime
            ),
        )

        assert result == (
            "'seconds' must be a number."
        )

        await runtime.shutdown()

    run(
        scenario()
    )


def test_sleep_rejects_negative_seconds():
    async def scenario():
        runtime = AgentRuntime()

        state = ExecutionState(
            active_skill=None,
            persona="test",
        )

        result = await SleepVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "seconds": -1,
                }
            ),
            context=None,
            state=state,
            engine=SimpleNamespace(
                agent_runtime=runtime
            ),
        )

        assert result == (
            "'seconds' must be >= 0."
        )

        await runtime.shutdown()

    run(
        scenario()
    )


# ============================================================================
# ActivateSkillVerb failure paths
# ============================================================================


def test_main_agent_cannot_activate_skill():
    """
    Main Agent is depth=0.

    The verb may be hallucinated by the model, but RolePolicy blocks
    execution before SkillRuntime.activate() is reached.
    """

    skill = SimpleNamespace(
        name="research",
        metadata=SimpleNamespace(
            name="research",
            description="Research",
            origin="workspace",
        ),
        instructions="Research carefully.",
        resources=[],
        scripts=(),
        references=(),
        assets=(),
        generation=1,
    )

    skills = FakeSkills(
        {
            "research": skill,
        }
    )

    llm = FakeLLM(
        [
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
                content="continued",
            ),
        ]
    )

    engine, runtime, _ = make_engine(
        llm=llm,
        skills=skills,
    )

    async def scenario():
        context = runtime.create_root(
            task="main",
        )

        try:
            result = await engine.execute(
                context=context,
                user_input="use research",
                persona="test",
            )

            assert (
                result.content
                == "continued"
            )

            assert (
                skills.activate_calls
                == []
            )

            tool_messages = unique_tool_messages(
                llm
            )

            assert tool_messages

            assert any(
                "only available to Subagents"
                in (
                    message.content
                    or ""
                )
                for message in tool_messages
            )

        finally:
            await runtime.shutdown()

    run(
        scenario()
    )


def test_activate_unknown_skill_preserves_current_skill():
    async def scenario():
        original = SimpleNamespace(
            name="original"
        )

        skills = FakeSkills(
            {
                "research": SimpleNamespace(
                    name="research"
                )
            }
        )

        state = ExecutionState(
            active_skill=original,
            persona="test",
        )

        result = await ActivateSkillVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "name": "missing",
                }
            ),
            context=None,
            state=state,
            engine=SimpleNamespace(
                skills=skills
            ),
        )

        assert (
            "Unknown Skill 'missing'."
            in result
        )

        assert (
            state.active_skill
            is original
        )

    run(
        scenario()
    )


# ============================================================================
# DispatchVerb failure paths
# ============================================================================


def test_dispatch_rejects_empty_task_without_creating_child():
    async def scenario():
        runtime = AgentRuntime()

        parent = runtime.create_root()

        state = ExecutionState(
            active_skill=None,
            persona="test",
        )

        engine = SimpleNamespace(
            agent_runtime=runtime,
            execute=None,
        )

        try:
            result = await DispatchVerb().execute(
                call=SimpleNamespace(
                    arguments={
                        "task": "   ",
                    }
                ),
                context=parent,
                state=state,
                engine=engine,
            )

            assert (
                "non-empty 'task'"
                in result
            )

            assert (
                state.children
                == []
            )

            assert (
                runtime.active_subagent_count
                == 0
            )

        finally:
            await runtime.shutdown()

    run(
        scenario()
    )


def test_dispatch_respects_depth_limit_without_creating_child():
    async def scenario():
        runtime = AgentRuntime(
            max_subagent_depth=0
        )

        parent = runtime.create_root()

        state = ExecutionState(
            active_skill=None,
            persona="test",
        )

        async def execute_child(
            *,
            context,
            user_input,
            persona,
        ):
            raise AssertionError(
                "Child must not be created"
            )

        engine = SimpleNamespace(
            agent_runtime=runtime,
            execute=execute_child,
        )

        try:
            result = await DispatchVerb().execute(
                call=SimpleNamespace(
                    arguments={
                        "task": "too deep",
                    }
                ),
                context=parent,
                state=state,
                engine=engine,
            )

            assert (
                "Maximum Subagent depth exceeded"
                in result
            )

            assert (
                state.children
                == []
            )

            assert (
                runtime.active_subagent_count
                == 0
            )

        finally:
            await runtime.shutdown()

    run(
        scenario()
    )


def test_failed_child_is_reported_without_raising_to_parent():
    async def scenario():
        from nan_itself.agent.reports import (
            format_child_report,
        )

        runtime = AgentRuntime()

        parent = runtime.create_root()

        state = ExecutionState(
            active_skill=None,
            persona="test",
        )

        async def execute_child(
            *,
            context,
            user_input,
            persona,
        ):
            raise RuntimeError(
                "child exploded"
            )

        engine = SimpleNamespace(
            agent_runtime=runtime,
            execute=execute_child,
        )

        try:
            dispatch_result = (
                await DispatchVerb().execute(
                    call=SimpleNamespace(
                        arguments={
                            "task": "failing child",
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

            report = (
                await format_child_report(
                    child
                )
            )

            assert (
                "status: failed"
                in report
            )

            assert (
                "child exploded"
                in report
            )

            assert (
                runtime.active_subagent_count
                == 0
            )

        finally:
            await runtime.shutdown()

    run(
        scenario()
    )


# ============================================================================
# Local Tool failures
# ============================================================================


def test_local_tool_rejects_invalid_arguments(
    tmp_path,
):
    async def scenario():
        runtime = await make_local_runtime(
            tmp_path,
            LOCAL_CALCULATOR,
        )

        try:
            view = AgentToolView(
                runtime
            )

            await view.call_tool(
                "route",
                {
                    "provider_name": "calc",
                },
            )

            result = await view.call_tool(
                "add",
                {
                    "x": "wrong",
                    "y": 2,
                },
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
        runtime = await make_local_runtime(
            tmp_path,
            LOCAL_FAILURE,
        )

        try:
            view = AgentToolView(
                runtime
            )

            await view.call_tool(
                "route",
                {
                    "provider_name": "calc",
                },
            )

            result = await view.call_tool(
                "explode",
                {
                    "message": "boom",
                },
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


def test_tool_call_before_route_returns_error(
    tmp_path,
):
    """
    The model may hallucinate a provider tool before routing.

    AgentToolView must return an error instead of throwing, and the
    StepEngine must remain able to continue.
    """

    async def scenario():
        runtime_provider = await make_local_runtime(
            tmp_path,
            LOCAL_CALCULATOR,
        )

        llm = FakeLLM(
            [
                make_response(
                    tool_calls=[
                        make_tool_call(
                            "bad-1",
                            "add",
                            {
                                "x": 1,
                                "y": 2,
                            },
                        )
                    ]
                ),
                make_response(
                    content="recovered",
                ),
            ]
        )

        runtime = AgentRuntime()

        engine = StepEngine(
            llm=llm,
            modules=FakeModules(),
            providers=runtime_provider,
            skills=FakeSkills(),
            agent_runtime=runtime,
        )

        context = runtime.create_root(
            task="test",
        )

        try:
            result = await engine.execute(
                context=context,
                user_input="add",
                persona="test",
            )

            assert (
                result.content
                == "recovered"
            )

            tool_messages = unique_tool_messages(
                llm
            )

            assert tool_messages

            assert any(
                "No tool provider is active"
                in (
                    message.content
                    or ""
                )
                for message in tool_messages
            )

        finally:
            await runtime.shutdown()
            await runtime_provider.stop()

    run(
        scenario()
    )


def test_unknown_tool_after_route_returns_error_and_agent_continues(
    tmp_path,
):
    async def scenario():
        provider_runtime = await make_local_runtime(
            tmp_path,
            LOCAL_CALCULATOR,
        )

        llm = FakeLLM(
            [
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
                            "missing-1",
                            "does_not_exist",
                            {},
                        )
                    ]
                ),
                make_response(
                    content="recovered",
                ),
            ]
        )

        runtime = AgentRuntime()

        engine = StepEngine(
            llm=llm,
            modules=FakeModules(),
            providers=provider_runtime,
            skills=FakeSkills(),
            agent_runtime=runtime,
        )

        context = runtime.create_root(
            task="unknown tool",
        )

        try:
            result = await engine.execute(
                context=context,
                user_input="use missing tool",
                persona="test",
            )

            assert (
                result.content
                == "recovered"
            )

            tool_messages = unique_tool_messages(
                llm
            )

            assert any(
                "does_not_exist"
                in (
                    message.content
                    or ""
                )
                for message in tool_messages
            )

            assert (
                len(llm.requests)
                == 3
            )

        finally:
            await runtime.shutdown()
            await provider_runtime.stop()

    run(
        scenario()
    )


def test_local_tool_error_does_not_prevent_next_model_step(
    tmp_path,
):
    """
    Tool validation failure is returned to the model as a tool result,
    not raised through StepEngine.
    """

    async def scenario():
        provider_runtime = await make_local_runtime(
            tmp_path,
            LOCAL_CALCULATOR,
        )

        llm = FakeLLM(
            [
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
                                "x": "wrong",
                                "y": 2,
                            },
                        )
                    ]
                ),
                make_response(
                    content="tool error handled",
                ),
            ]
        )

        runtime = AgentRuntime()

        engine = StepEngine(
            llm=llm,
            modules=FakeModules(),
            providers=provider_runtime,
            skills=FakeSkills(),
            agent_runtime=runtime,
        )

        context = runtime.create_root(
            task="invalid tool call",
        )

        try:
            result = await engine.execute(
                context=context,
                user_input="calculate",
                persona="test",
            )

            assert (
                result.content
                == "tool error handled"
            )

            assert (
                len(llm.requests)
                == 3
            )

            tool_messages = unique_tool_messages(
                llm
            )

            assert (
                len(tool_messages)
                == 2
            )

            assert any(
                "Invalid arguments for tool"
                in (
                    message.content
                    or ""
                )
                for message in tool_messages
            )

        finally:
            await runtime.shutdown()
            await provider_runtime.stop()

    run(
        scenario()
    )


# ============================================================================
# Tool disappearance / routing failures
# ============================================================================


def test_active_provider_disappearing_returns_error(
    tmp_path,
):
    async def scenario():
        runtime = await make_local_runtime(
            tmp_path,
            LOCAL_CALCULATOR,
        )

        try:
            view = AgentToolView(
                runtime
            )

            route_result = await view.call_tool(
                "route",
                {
                    "provider_name": "calc",
                },
            )

            assert (
                route_result.isError
                is False
            )

            assert (
                view.active_provider
                == "calc"
            )

            runtime.providers.pop(
                "calc",
                None,
            )

            result = await view.call_tool(
                "add",
                {
                    "x": 1,
                    "y": 2,
                },
            )

            assert (
                result.isError
                is True
            )

            assert (
                "no longer available"
                in result.content[0].text
            )

            assert (
                view.active_provider
                is None
            )

        finally:
            await runtime.stop()

    run(
        scenario()
    )


def test_route_unknown_provider_returns_error():
    async def scenario():
        class Runtime:
            providers = {}

            def provider_names(self):
                return ()

            def get_provider(
                self,
                name,
            ):
                return None

        view = AgentToolView(
            Runtime()
        )

        result = await view.call_tool(
            "route",
            {
                "provider_name": "missing",
            },
        )

        assert (
            result.isError
            is True
        )

        assert (
            "Unknown tool provider"
            in result.content[0].text
        )

    run(
        scenario()
    )


# ============================================================================
# MCP failures
# ============================================================================


class FakeMCPStack:
    def __init__(self):
        self.closed = False
        self.closed_by = None

    async def aclose(self):
        self.closed = True
        self.closed_by = (
            asyncio.current_task()
        )


class FailingMCPSession:
    def __init__(
        self,
        *,
        delay=0.0,
        error=None,
    ):
        self.delay = delay
        self.error = error
        self.calls = []

    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="lookup",
                    description="Lookup",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                            }
                        },
                        "required": [
                            "key",
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

        if self.error is not None:
            raise self.error

        return text_result(
            f"ok:{arguments['key']}"
        )


def write_mcp_config(
    path: Path,
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


def test_mcp_timeout_returns_error(
    tmp_path,
    monkeypatch,
):
    async def scenario():
        source = (
            tmp_path / "example.yaml"
        )

        write_mcp_config(
            source
        )

        async def fake_connect(
            spec,
        ):
            stack = FakeMCPStack()

            session = FailingMCPSession(
                delay=0.1
            )

            result = (
                await session.list_tools()
            )

            return Provider(
                spec=spec,
                stack=stack,
                session=session,
                tools={
                    tool.name: tool
                    for tool in result.tools
                },
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

            result = await view.call_tool(
                "lookup",
                {
                    "key": "slow",
                },
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


def test_mcp_backend_exception_becomes_error_result(
    tmp_path,
    monkeypatch,
):
    async def scenario():
        source = (
            tmp_path / "example.yaml"
        )

        write_mcp_config(
            source
        )

        async def fake_connect(
            spec,
        ):
            stack = FakeMCPStack()

            session = FailingMCPSession(
                error=RuntimeError(
                    "mcp exploded"
                )
            )

            result = (
                await session.list_tools()
            )

            return Provider(
                spec=spec,
                stack=stack,
                session=session,
                tools={
                    tool.name: tool
                    for tool in result.tools
                },
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

            view = AgentToolView(
                runtime
            )

            await view.call_tool(
                "route",
                {
                    "provider_name": "example",
                },
            )

            result = await view.call_tool(
                "lookup",
                {
                    "key": "boom",
                },
            )

            assert (
                result.isError
                is True
            )

            assert (
                "mcp exploded"
                in result.content[0].text
            )

        finally:
            await runtime.stop()

    run(
        scenario()
    )


# ============================================================================
# Subagent failure propagation to parent
# ============================================================================


def test_parent_can_continue_after_failed_subagent():
    """
    A failed child becomes a normal [Subagent Report] failure message.
    Parent execution continues and can produce a final answer.
    """

    class LLM:
        model = "fake-model"
        provider = "fake-provider"

        def __init__(self):
            self.requests = []

        async def generate_complete(
            self,
            request,
        ):
            self.requests.append(
                request
            )

            await asyncio.sleep(0)

            user_inputs = [
                message.content
                for message in request.messages
                if getattr(
                    message,
                    "role",
                    None,
                ) == "user"
                and isinstance(
                    getattr(
                        message,
                        "content",
                        None,
                    ),
                    str,
                )
            ]

            if not user_inputs:
                raise AssertionError(
                    "No user input found."
                )

            # ------------------------------------------------------
            # Child.
            # ------------------------------------------------------

            if (
                user_inputs[0]
                == "failing child"
            ):
                raise RuntimeError(
                    "child exploded"
                )

            # ------------------------------------------------------
            # Parent.
            # ------------------------------------------------------

            assert (
                user_inputs[0]
                == "parent task"
            )

            has_report = any(
                "[Subagent Report]"
                in (
                    message.content
                    or ""
                )
                for message in request.messages
                if getattr(
                    message,
                    "role",
                    None,
                ) == "user"
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
                return make_response(
                    content=(
                        "parent recovered "
                        "from child failure"
                    )
                )

            # First parent step: dispatch child.
            if not hasattr(
                self,
                "_dispatched",
            ):
                self._dispatched = True

                return make_response(
                    tool_calls=[
                        make_tool_call(
                            "dispatch-1",
                            "dispatch_subagent",
                            {
                                "task": (
                                    "failing child"
                                )
                            },
                        )
                    ]
                )

            # Give the child a chance to finish.
            if not hasattr(
                self,
                "_yielded",
            ):
                self._yielded = True

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
                "the failed-child report."
            )

    async def scenario():
        runtime = AgentRuntime(
            max_subagent_depth=3
        )

        llm = LLM()

        engine = StepEngine(
            llm=llm,
            modules=FakeModules(),
            providers=FakeProviders(),
            skills=FakeSkills(),
            agent_runtime=runtime,
        )

        parent = runtime.create_root(
            task="parent task",
        )

        try:
            result = await engine.execute(
                context=parent,
                user_input="parent task",
                persona="parent persona",
            )

            assert (
                result.content
                == (
                    "parent recovered "
                    "from child failure"
                )
            )

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

            assert (
                children[0].task
                == "failing child"
            )

            assert (
                runtime.active_subagent_count
                == 0
            )

            assert any(
                "[Subagent Report]"
                in (
                    message.content
                    or ""
                )
                and "child exploded"
                in (
                    message.content
                    or ""
                )
                for request in llm.requests
                for message in request.messages
                if getattr(
                    message,
                    "role",
                    None,
                ) == "user"
                and isinstance(
                    getattr(
                        message,
                        "content",
                        None,
                    ),
                    str,
                )
            )

        finally:
            await runtime.shutdown()

    run(
        scenario()
    )


# ============================================================================
# Combined failure: one tool call fails, another succeeds
# ============================================================================


def test_multiple_tool_calls_continue_after_one_failure(
    tmp_path,
):
    """
    One model response may contain multiple tool calls.

    The first tool call fails validation; the second succeeds.

    StepEngine must execute both calls rather than aborting the whole
    response after the first tool error.
    """

    async def scenario():
        provider_runtime = await make_local_runtime(
            tmp_path,
            LOCAL_CALCULATOR,
        )

        llm = FakeLLM(
            [
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
                            "bad-add",
                            "add",
                            {
                                "x": "bad",
                                "y": 2,
                            },
                        ),
                        make_tool_call(
                            "good-add",
                            "add",
                            {
                                "x": 3,
                                "y": 4,
                            },
                        ),
                    ]
                ),
                make_response(
                    content="one failed, one succeeded",
                ),
            ]
        )

        runtime = AgentRuntime()

        engine = StepEngine(
            llm=llm,
            modules=FakeModules(),
            providers=provider_runtime,
            skills=FakeSkills(),
            agent_runtime=runtime,
        )

        context = runtime.create_root(
            task="multiple tools",
        )

        try:
            result = await engine.execute(
                context=context,
                user_input="calculate",
                persona="test",
            )

            assert (
                result.content
                == "one failed, one succeeded"
            )

            tool_messages = unique_tool_messages(
                llm
            )

            assert (
                len(tool_messages)
                == 3
            )

            assert any(
                "Invalid arguments for tool"
                in (
                    message.content
                    or ""
                )
                for message in tool_messages
            )

            assert any(
                (
                    message.content
                    or ""
                )
                == (
                    '{"meta": null, '
                    '"content": [{"type": "text", '
                    '"text": "7", '
                    '"annotations": null, '
                    '"meta": null}], '
                    '"structuredContent": null, '
                    '"isError": false}'
                )
                for message in tool_messages
            )

        finally:
            await runtime.shutdown()
            await provider_runtime.stop()

    run(
        scenario()
    )
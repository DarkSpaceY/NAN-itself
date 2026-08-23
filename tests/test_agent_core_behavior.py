from __future__ import annotations

import asyncio
from pathlib import Path
from types import MappingProxyType

import pytest
from mcp.types import Tool

from src.nan_itself.agent.core import CoreAgent
from src.nan_itself.agent.runtime import AgentContext
from src.nan_itself.skills.facade import (
    Skill,
    SkillMetadata,
)
from src.nan_itself.tools.facade import (
    MCPProvider,
    MCPProviderSpec,
    MCPRuntime,
)
from src.nan_itself.utils.llm import (
    LLMResponse,
    Message,
    ToolCall,
    Usage,
)


# ============================================================================
# Fakes
# ============================================================================


def make_skill(
    name: str = "core",
    instructions: str | None = None,
) -> Skill:
    metadata = SkillMetadata(
        name=name,
        description=f"{name} skill",
        source=Path(
            f"/tmp/{name}/SKILL.md"
        ),
        origin="builtin",
        frontmatter=MappingProxyType({}),
    )

    return Skill(
        metadata=metadata,
        instructions=(
            instructions
            if instructions is not None
            else f"Use {name} workflow."
        ),
        scripts=(),
        references=(),
        assets=(),
    )


class FakeSkills:
    def __init__(
        self,
        skills: dict[str, Skill],
    ):
        self._skills = skills

    def names(self) -> tuple[str, ...]:
        return tuple(
            sorted(self._skills)
        )

    def activate(
        self,
        name: str,
    ) -> Skill:
        return self._skills[name]


class FakeModules:
    def __init__(
        self,
        context: list[str] | None = None,
    ):
        self.world = {
            "todo": {
                "active": 3,
            },
            "value": 123,
        }

        self.context = (
            context
            if context is not None
            else [
                "There are 3 active TODO items."
            ]
        )

        self.queries: list[
            tuple[str, int, object]
        ] = []

    def snapshot(self):
        return self.world

    async def query_snapshot(
        self,
        turn,
        snapshot,
    ):
        self.queries.append(
            (
                turn.agent_hash,
                turn.depth,
                snapshot,
            )
        )

        return list(self.context)


class FakeSession:
    def __init__(
        self,
        tools: list[Tool],
        handler=None,
    ):
        self.tools = {
            tool.name: tool
            for tool in tools
        }

        self.handler = handler
        self.calls: list[
            tuple[str, dict]
        ] = []

    async def list_tools(self):
        return type(
            "ToolListResult",
            (),
            {
                "tools": list(
                    self.tools.values()
                )
            },
        )()

    async def call_tool(
        self,
        name,
        arguments,
    ):
        self.calls.append(
            (
                name,
                arguments,
            )
        )

        if self.handler is not None:
            return await self.handler(
                name,
                arguments,
            )

        return {
            "tool": name,
            "arguments": arguments,
        }


class FakeStack:
    async def aclose(self):
        pass


def install_fake_mcp(
    runtime: MCPRuntime,
    name: str,
    tools: list[Tool],
    handler=None,
) -> FakeSession:
    session = FakeSession(
        tools=tools,
        handler=handler,
    )

    spec = MCPProviderSpec(
        name=name,
        command="fake",
        source="<test>",
        origin="builtin",
    )

    runtime.providers[name] = MCPProvider(
        spec=spec,
        stack=FakeStack(),
        session=session,
        tools={
            tool.name: tool
            for tool in tools
        },
    )

    return session


class SequenceLLM:
    """
    Returns one preconfigured response per LLM call.
    """

    def __init__(
        self,
        responses: list[LLMResponse],
    ):
        self.responses = list(responses)
        self.requests = []

    async def generate_complete(
        self,
        request,
    ):
        self.requests.append(request)

        if not self.responses:
            raise AssertionError(
                "SequenceLLM ran out of responses"
            )

        return self.responses.pop(0)


def response(
    *,
    text: str | None = None,
    calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=list(calls or []),
        model="fake",
        usage=Usage(),
        provider="openai",
        finish_reason=finish_reason,
    )


def tool_call(
    *,
    call_id: str,
    name: str,
    arguments: dict,
) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=name,
        arguments=arguments,
    )


def echo_tool() -> Tool:
    return Tool(
        name="echo",
        description="Echo a value.",
        inputSchema={
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                },
            },
            "required": ["value"],
        },
    )


def complete_tool() -> Tool:
    return Tool(
        name="complete",
        description="Complete something.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    )


# ============================================================================
# Main Agent
# ============================================================================


@pytest.mark.asyncio
async def test_main_agent_uses_core_skill_only():
    core = make_skill(
        "core",
        "CORE SYSTEM PROMPT",
    )

    llm = SequenceLLM([
        response(
            text="done",
        ),
    ])

    agent = CoreAgent(
        llm=llm,
        modules=FakeModules(),
        mcp=MCPRuntime(),
        skills=FakeSkills({
            "core": core,
            "research": make_skill(
                "research",
                "RESEARCH SYSTEM PROMPT",
            ),
        }),
        core_skill=core,
    )

    result = await agent.run(
        "hello"
    )

    assert result.content == "done"

    system = llm.requests[0].messages[0]

    assert system.role == "system"
    assert "CORE SYSTEM PROMPT" in (
        system.content or ""
    )
    assert "RESEARCH SYSTEM PROMPT" not in (
        system.content or ""
    )


@pytest.mark.asyncio
async def test_main_agent_receives_ambient_module_context():
    core = make_skill(
        instructions="CORE",
    )

    llm = SequenceLLM([
        response(
            text="answer",
        ),
    ])

    modules = FakeModules(
        context=[
            "Memory: user prefers concise answers.",
            "Todo: report is due today.",
        ],
    )

    agent = CoreAgent(
        llm=llm,
        modules=modules,
        mcp=MCPRuntime(),
        skills=FakeSkills({
            "core": core,
        }),
        core_skill=core,
    )

    await agent.run(
        "What should I do?"
    )

    system = llm.requests[0].messages[0]

    assert (
        "Memory: user prefers concise answers."
        in (system.content or "")
    )

    assert (
        "Todo: report is due today."
        in (system.content or "")
    )


@pytest.mark.asyncio
async def test_main_agent_keeps_only_recent_turns():
    core = make_skill()

    llm = SequenceLLM([
        response(text="answer-0"),
        response(text="answer-1"),
        response(text="answer-2"),
        response(text="answer-3"),
        response(text="answer-4"),
        response(text="answer-5"),
    ])

    agent = CoreAgent(
        llm=llm,
        modules=FakeModules(),
        mcp=MCPRuntime(),
        skills=FakeSkills({
            "core": core,
        }),
        core_skill=core,
        history_turns=3,
    )

    for index in range(6):
        await agent.run(
            f"user-{index}"
        )

    assert len(
        agent._main_history
    ) == 3


# ============================================================================
# World snapshot
# ============================================================================


@pytest.mark.asyncio
async def test_main_and_subagent_share_exact_same_snapshot():
    core = make_skill()

    llm = SequenceLLM([
        response(
            calls=[
                tool_call(
                    call_id="dispatch",
                    name="dispatch_subagent",
                    arguments={
                        "task": "inspect world",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="main done",
        ),
        response(
            text="sub done",
        ),
    ])

    modules = FakeModules()

    agent = CoreAgent(
        llm=llm,
        modules=modules,
        mcp=MCPRuntime(),
        skills=FakeSkills({
            "core": core,
        }),
        core_skill=core,
    )

    result = await agent.run(
        "do both",
    )

    assert result.content == "main done"

    for _ in range(100):
        if len(modules.queries) >= 2:
            break

        await asyncio.sleep(0.01)

    assert len(modules.queries) >= 2

    main_query = modules.queries[0]
    sub_query = modules.queries[1]

    assert main_query[2] is sub_query[2]


@pytest.mark.asyncio
async def test_module_can_distinguish_main_and_subagent():
    core = make_skill()

    class IdentityAwareModules(
        FakeModules
    ):
        async def query_snapshot(
            self,
            turn,
            snapshot,
        ):
            self.queries.append(
                (
                    turn.agent_hash,
                    turn.depth,
                    snapshot,
                )
            )

            if turn.depth == 0:
                return [
                    "MAIN CONTEXT"
                ]

            return [
                "SUBAGENT CONTEXT"
            ]

    modules = IdentityAwareModules()

    llm = SequenceLLM([
        response(
            calls=[
                tool_call(
                    call_id="dispatch",
                    name="dispatch_subagent",
                    arguments={
                        "task": "sub task",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="main",
        ),
        response(
            text="sub",
        ),
    ])

    agent = CoreAgent(
        llm=llm,
        modules=modules,
        mcp=MCPRuntime(),
        skills=FakeSkills({
            "core": core,
        }),
        core_skill=core,
    )

    await agent.run(
        "start",
    )

    for _ in range(100):
        if len(modules.queries) >= 2:
            break

        await asyncio.sleep(0.01)

    assert len(modules.queries) == 2

    assert {
        query[1]
        for query in modules.queries
    } == {0, 1}


# ============================================================================
# MCP / Tool loop
# ============================================================================


@pytest.mark.asyncio
async def test_tool_call_round_trips_through_model():
    core = make_skill()

    session = None

    async def handler(
        name,
        arguments,
    ):
        return {
            "result": (
                f"echo:{arguments['value']}"
            )
        }

    mcp = MCPRuntime()

    session = install_fake_mcp(
        mcp,
        "test",
        [
            echo_tool(),
        ],
        handler=handler,
    )

    llm = SequenceLLM([
        response(
            calls=[
                tool_call(
                    call_id="route",
                    name="route_mcp",
                    arguments={
                        "mcp_name": "test",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            calls=[
                tool_call(
                    call_id="echo",
                    name="echo",
                    arguments={
                        "value": "hello",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="Echo completed.",
        ),
    ])

    agent = CoreAgent(
        llm=llm,
        modules=FakeModules(),
        mcp=mcp,
        skills=FakeSkills({
            "core": core,
        }),
        core_skill=core,
    )

    result = await agent.run(
        "echo hello",
    )

    assert result.content == (
        "Echo completed."
    )

    assert session.calls == [
        (
            "echo",
            {
                "value": "hello",
            },
        )
    ]

    final_request = llm.requests[-1]

    assert any(
        message.role == "tool"
        for message in final_request.messages
    )


@pytest.mark.asyncio
async def test_route_mcp_is_agent_local_during_core_loop():
    core = make_skill()

    mcp = MCPRuntime()

    install_fake_mcp(
        mcp,
        "files",
        [echo_tool()],
    )

    install_fake_mcp(
        mcp,
        "browser",
        [complete_tool()],
    )

    llm = SequenceLLM([
        response(
            calls=[
                tool_call(
                    call_id="route-files",
                    name="route_mcp",
                    arguments={
                        "mcp_name": "files",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="done",
        ),
    ])

    agent = CoreAgent(
        llm=llm,
        modules=FakeModules(),
        mcp=mcp,
        skills=FakeSkills({
            "core": core,
        }),
        core_skill=core,
    )

    await agent.run(
        "files task",
    )

    first_request = llm.requests[0]

    first_tool_names = {
        tool.name
        for tool in first_request.tools
    }

    assert (
        "route_mcp"
        in first_tool_names
    )

    assert (
        "echo"
        not in first_tool_names
    )

    # Route result is processed internally;
    # the next model call receives the routed tool.
    second_request = llm.requests[1]

    second_tool_names = {
        tool.name
        for tool in second_request.tools
    }

    assert "echo" in second_tool_names
    assert "complete" not in second_tool_names


# ============================================================================
# Subagent dispatch
# ============================================================================


@pytest.mark.asyncio
async def test_subagent_dispatch_returns_immediately():
    core = make_skill()

    started = asyncio.Event()
    release = asyncio.Event()

    class SlowModules(FakeModules):
        pass

    class ControlledCore(
        CoreAgent
    ):
        async def _run_agent(
            self,
            *,
            context,
            user_input,
            skill,
            history,
        ):
            if context.depth == 1:
                started.set()
                await release.wait()

                return type(
                    "Result",
                    (),
                    {
                        "content": "sub done",
                        "messages": (),
                        "response": None,
                    },
                )()

            return await super()._run_agent(
                context=context,
                user_input=user_input,
                skill=skill,
                history=history,
            )

    llm = SequenceLLM([
        response(
            calls=[
                tool_call(
                    call_id="dispatch",
                    name="dispatch_subagent",
                    arguments={
                        "task": "slow work",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="main continues",
        ),
    ])

    agent = ControlledCore(
        llm=llm,
        modules=SlowModules(),
        mcp=MCPRuntime(),
        skills=FakeSkills({
            "core": core,
        }),
        core_skill=core,
    )

    result = await agent.run(
        "parallel",
    )

    assert result.content == (
        "main continues"
    )

    await asyncio.wait_for(
        started.wait(),
        timeout=1.0,
    )

    assert any(
        not handle.done
        for handle in agent._subagents.values()
    )

    release.set()

    await asyncio.gather(
        *(
            handle.wait()
            for handle in agent._subagents.values()
        )
    )


@pytest.mark.asyncio
async def test_main_can_wait_for_subagent():
    core = make_skill()

    started = asyncio.Event()
    release = asyncio.Event()

    class ControlledCore(
        CoreAgent
    ):
        async def _run_agent(
            self,
            *,
            context,
            user_input,
            skill,
            history,
        ):
            if context.depth == 1:
                started.set()
                await release.wait()

                return type(
                    "Result",
                    (),
                    {
                        "content": "subagent result",
                        "messages": (),
                        "response": None,
                    },
                )()

            return await super()._run_agent(
                context=context,
                user_input=user_input,
                skill=skill,
                history=history,
            )

    # First dispatches.
    # Second asks for a specific handle via sleep.
    # But the model doesn't yet know the handle at authoring time,
    # so this test focuses on the underlying handle plumbing by
    # injecting it after dispatch.
    llm = SequenceLLM([
        response(
            calls=[
                tool_call(
                    call_id="dispatch",
                    name="dispatch_subagent",
                    arguments={
                        "task": "slow work",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="main waiting",
        ),
    ])

    agent = ControlledCore(
        llm=llm,
        modules=FakeModules(),
        mcp=MCPRuntime(),
        skills=FakeSkills({
            "core": core,
        }),
        core_skill=core,
    )

    result = await agent.run(
        "dispatch and wait",
    )

    assert result.content == (
        "main waiting"
    )

    await started.wait()

    handles = list(
        agent._subagents.values()
    )

    assert len(handles) == 1

    handle = handles[0]

    assert handle.done is False

    release.set()

    child_result = await handle.wait()

    assert child_result.content == (
        "subagent result"
    )


# ============================================================================
# Skill switching
# ============================================================================


@pytest.mark.asyncio
async def test_subagent_receives_selected_skill():
    core = make_skill(
        "core",
        "CORE",
    )

    research = make_skill(
        "research",
        "RESEARCH",
    )

    llm = SequenceLLM([
        response(
            calls=[
                tool_call(
                    call_id="dispatch",
                    name="dispatch_subagent",
                    arguments={
                        "task": "research this",
                        "skill": "research",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="main",
        ),
        response(
            text="sub",
        ),
    ])

    agent = CoreAgent(
        llm=llm,
        modules=FakeModules(),
        mcp=MCPRuntime(),
        skills=FakeSkills({
            "core": core,
            "research": research,
        }),
        core_skill=core,
    )

    await agent.run(
        "delegate",
    )

    for _ in range(100):
        if len(llm.requests) >= 3:
            break

        await asyncio.sleep(0.01)

    assert len(llm.requests) >= 3

    subagent_request = (
        llm.requests[2]
    )

    system = (
        subagent_request
        .messages[0]
        .content
        or ""
    )

    assert "RESEARCH" in system
    assert "CORE" not in system


@pytest.mark.asyncio
async def test_subagent_can_explicitly_use_core_skill():
    core = make_skill(
        "core",
        "CORE",
    )

    research = make_skill(
        "research",
        "RESEARCH",
    )

    llm = SequenceLLM([
        response(
            calls=[
                tool_call(
                    call_id="dispatch",
                    name="dispatch_subagent",
                    arguments={
                        "task": "switch to core",
                        "skill": "core",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="main",
        ),
        response(
            text="child",
        ),
    ])

    agent = CoreAgent(
        llm=llm,
        modules=FakeModules(),
        mcp=MCPRuntime(),
        skills=FakeSkills({
            "core": core,
            "research": research,
        }),
        core_skill=core,
    )

    await agent.run(
        "delegate",
    )

    for _ in range(100):
        if len(llm.requests) >= 3:
            break

        await asyncio.sleep(0.01)

    subagent_system = (
        llm.requests[2]
        .messages[0]
        .content
        or ""
    )

    assert "CORE" in subagent_system
    assert "RESEARCH" not in subagent_system


# ============================================================================
# Recursive Subagent
# ============================================================================


@pytest.mark.asyncio
async def test_subagent_can_dispatch_sub_subagent():
    core = make_skill()

    llm = SequenceLLM([
        # Main dispatch child.
        response(
            calls=[
                tool_call(
                    call_id="child",
                    name="dispatch_subagent",
                    arguments={
                        "task": "child task",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        # Main finishes.
        response(
            text="main",
        ),
        # Child dispatches grandchild.
        response(
            calls=[
                tool_call(
                    call_id="grandchild",
                    name="dispatch_subagent",
                    arguments={
                        "task": "grandchild task",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        # Child finishes.
        response(
            text="child",
        ),
        # Grandchild finishes.
        response(
            text="grandchild",
        ),
    ])

    agent = CoreAgent(
        llm=llm,
        modules=FakeModules(),
        mcp=MCPRuntime(),
        skills=FakeSkills({
            "core": core,
        }),
        core_skill=core,
        max_subagent_depth=2,
    )

    result = await agent.run(
        "recursive",
    )

    assert result.content == "main"

    for _ in range(100):
        if len(agent._subagents) >= 2:
            break

        await asyncio.sleep(0.01)

    assert len(agent._subagents) >= 2

    depths = sorted(
        handle.depth
        for handle in agent._subagents.values()
    )

    assert depths == [
        1,
        2,
    ]


@pytest.mark.asyncio
async def test_subagent_depth_limit_is_exposed_to_agent():
    core = make_skill()

    llm = SequenceLLM([
        # Main dispatches one child.
        response(
            calls=[
                tool_call(
                    call_id="child",
                    name="dispatch_subagent",
                    arguments={
                        "task": "child",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="main",
        ),
        # Child tries to dispatch another child.
        response(
            calls=[
                tool_call(
                    call_id="grandchild",
                    name="dispatch_subagent",
                    arguments={
                        "task": "grandchild",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="child",
        ),
    ])

    agent = CoreAgent(
        llm=llm,
        modules=FakeModules(),
        mcp=MCPRuntime(),
        skills=FakeSkills({
            "core": core,
        }),
        core_skill=core,
        max_subagent_depth=1,
    )

    await agent.run(
        "depth",
    )

    for _ in range(100):
        if len(llm.requests) >= 4:
            break

        await asyncio.sleep(0.01)

    # Grandchild was not actually created.
    assert len(agent._subagents) == 1

    child_request = llm.requests[2]

    assert any(
        message.role == "tool"
        and "Maximum Subagent depth"
        in (message.content or "")
        for message in llm.requests[3].messages
    ) or any(
        "Maximum Subagent depth"
        in (message.content or "")
        for message in child_request.messages
        if message.role == "tool"
    )


# ============================================================================
# Sleep
# ============================================================================


@pytest.mark.asyncio
async def test_agent_can_sleep_and_resume():
    core = make_skill()

    llm = SequenceLLM([
        response(
            calls=[
                tool_call(
                    call_id="sleep",
                    name="sleep",
                    arguments={
                        "seconds": 0.01,
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="awake",
        ),
    ])

    agent = CoreAgent(
        llm=llm,
        modules=FakeModules(),
        mcp=MCPRuntime(),
        skills=FakeSkills({
            "core": core,
        }),
        core_skill=core,
    )

    result = await agent.run(
        "wait",
    )

    assert result.content == "awake"
    assert len(llm.requests) == 2


# ============================================================================
# Tool environment remains global while Skill changes
# ============================================================================


@pytest.mark.asyncio
async def test_changing_skill_does_not_remove_global_tools():
    core = make_skill(
        "core",
        "CORE",
    )

    research = make_skill(
        "research",
        "RESEARCH",
    )

    mcp = MCPRuntime()

    install_fake_mcp(
        mcp,
        "test",
        [echo_tool()],
    )

    llm = SequenceLLM([
        response(
            calls=[
                tool_call(
                    call_id="route",
                    name="route_mcp",
                    arguments={
                        "mcp_name": "test",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            calls=[
                tool_call(
                    call_id="dispatch",
                    name="dispatch_subagent",
                    arguments={
                        "task": "research",
                        "skill": "research",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="main",
        ),
        response(
            text="sub",
        ),
    ])

    agent = CoreAgent(
        llm=llm,
        modules=FakeModules(),
        mcp=mcp,
        skills=FakeSkills({
            "core": core,
            "research": research,
        }),
        core_skill=core,
    )

    await agent.run(
        "start",
    )

    for _ in range(100):
        if len(llm.requests) >= 4:
            break

        await asyncio.sleep(0.01)

    assert len(llm.requests) >= 4

    # Main's routed MCP tool remains available on the next model call.
    main_after_route = llm.requests[1]

    assert any(
        tool.name == "echo"
        for tool in main_after_route.tools
    )

    # Subagent has a fresh view.
    # The global MCP provider exists, but routing is agent-local.
    subagent_request = llm.requests[3]

    assert any(
        tool.name == "route_mcp"
        for tool in subagent_request.tools
    )
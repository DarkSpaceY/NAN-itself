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
    Provider,
    ProviderSpec,
    ProviderRuntime,
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

    def catalog(self):
        return [
            self._skills[name].metadata
            for name in sorted(self._skills)
        ]


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
    runtime: ProviderRuntime,
    name: str,
    tools: list[Tool],
    handler=None,
) -> FakeSession:
    session = FakeSession(
        tools=tools,
        handler=handler,
    )

    spec = ProviderSpec(
        name=name,
        command="fake",
        source="<test>",
        origin="builtin",
    )

    runtime.providers[name] = Provider(
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
        providers=ProviderRuntime(),
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
        providers=ProviderRuntime(),
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
async def test_main_history_respects_token_budget():
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
        providers=ProviderRuntime(),
        skills=FakeSkills({
            "core": core,
        }),
        core_skill=core,
        history_token_budget=45,
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
        providers=ProviderRuntime(),
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
        providers=ProviderRuntime(),
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

    providers = ProviderRuntime()

    session = install_fake_mcp(
        providers,
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
                    name="route",
                    arguments={
                        "provider_name": "test",
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
        providers=providers,
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


# ============================================================================
# Skill switching
# ============================================================================


@pytest.mark.asyncio
async def test_subagent_activates_skill_itself():
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
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="main",
        ),
        response(
            calls=[
                tool_call(
                    call_id="switch",
                    name="activate_skill",
                    arguments={
                        "name": "research",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="sub",
        ),
    ])

    agent = CoreAgent(
        llm=llm,
        modules=FakeModules(),
        providers=ProviderRuntime(),
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
        if len(llm.requests) >= 4:
            break

        await asyncio.sleep(0.01)

    assert len(llm.requests) >= 4

    first_child_request = (
        llm.requests[2]
    )

    switched_request = (
        llm.requests[3]
    )

    first_system = (
        first_child_request
        .messages[0]
        .content
        or ""
    )

    # The child starts on the inherited core skill and
    # sees the catalog of switchable skills.
    assert "CORE" in first_system
    assert "RESEARCH" not in first_system
    assert "[Available Skills]" in first_system

    switched_system = (
        switched_request
        .messages[0]
        .content
        or ""
    )

    assert "RESEARCH" in switched_system


@pytest.mark.asyncio
async def test_subagent_inherits_parent_skill_by_default():
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
                        "task": "plain task",
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
        providers=ProviderRuntime(),
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


@pytest.mark.asyncio
async def test_main_agent_never_sees_activate_skill():
    core = make_skill(
        "core",
        "CORE",
    )

    research = make_skill(
        "research",
        "RESEARCH",
    )

    llm = SequenceLLM([
        # The Main Agent hallucinates the Subagent-only tool.
        response(
            calls=[
                tool_call(
                    call_id="switch",
                    name="activate_skill",
                    arguments={
                        "name": "research",
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
        providers=ProviderRuntime(),
        skills=FakeSkills({
            "core": core,
            "research": research,
        }),
        core_skill=core,
    )

    await agent.run(
        "try to switch",
    )

    assert len(llm.requests) == 2

    first_request = llm.requests[0]

    # The Main Agent never sees activate_skill.
    assert not any(
        tool.name == "activate_skill"
        for tool in first_request.tools
    )

    second_request = llm.requests[1]

    tool_messages = [
        message
        for message in second_request.messages
        if message.role == "tool"
    ]

    assert any(
        "only available to Subagents"
        in (message.content or "")
        for message in tool_messages
    )

    # The Main Agent is still pinned to the core skill.
    system = (
        second_request.messages[0].content
        or ""
    )

    assert "CORE" in system
    assert "RESEARCH" not in system


@pytest.mark.asyncio
async def test_activate_skill_unknown_name_returns_error():
    core = make_skill(
        "core",
        "CORE",
    )

    llm = SequenceLLM([
        response(
            calls=[
                tool_call(
                    call_id="dispatch",
                    name="dispatch_subagent",
                    arguments={
                        "task": "task",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="main",
        ),
        response(
            calls=[
                tool_call(
                    call_id="switch",
                    name="activate_skill",
                    arguments={
                        "name": "missing",
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
        providers=ProviderRuntime(),
        skills=FakeSkills({
            "core": core,
        }),
        core_skill=core,
    )

    await agent.run(
        "delegate",
    )

    for _ in range(100):
        if len(llm.requests) >= 4:
            break

        await asyncio.sleep(0.01)

    assert len(llm.requests) >= 4

    switched_request = (
        llm.requests[3]
    )

    # The Subagent does see activate_skill.
    assert any(
        tool.name == "activate_skill"
        for tool in switched_request.tools
    )

    tool_messages = [
        message
        for message in switched_request.messages
        if message.role == "tool"
    ]

    assert any(
        "Unknown Skill 'missing'" in (message.content or "")
        and "Available Skills:" in (message.content or "")
        for message in tool_messages
    )

    # The failed activation leaves the inherited skill active.
    system = (
        switched_request.messages[0].content
        or ""
    )

    assert "CORE" in system


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
        providers=ProviderRuntime(),
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
        if len(agent.agent_runtime.agents()) >= 3:
            break

        await asyncio.sleep(0.01)

    depths = sorted(
        context.depth
        for context in agent.agent_runtime.agents()
        if context.depth > 0
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
        providers=ProviderRuntime(),
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

    created = [
        context
        for context in agent.agent_runtime.agents()
        if context.depth > 0
    ]

    # Grandchild was not actually created.
    assert len(created) == 1

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
        providers=ProviderRuntime(),
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
async def test_routed_tools_do_not_leak_into_subagent_view():
    core = make_skill(
        "core",
        "CORE",
    )

    research = make_skill(
        "research",
        "RESEARCH",
    )

    providers = ProviderRuntime()

    install_fake_mcp(
        providers,
        "test",
        [echo_tool()],
    )

    llm = SequenceLLM([
        response(
            calls=[
                tool_call(
                    call_id="route",
                    name="route",
                    arguments={
                        "provider_name": "test",
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
        providers=providers,
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
        tool.name == "route"
        for tool in subagent_request.tools
    )
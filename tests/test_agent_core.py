from __future__ import annotations

import asyncio

import pytest

from src.nan_itself.agent.core import (
    CoreAgent,
)
from src.nan_itself.agent.runtime import (
    AgentContext,
)
from src.nan_itself.skills.facade import (
    Skill,
    SkillMetadata,
)
from src.nan_itself.tools.facade import (
    ProviderRuntime,
    Provider,
    ProviderSpec,
)
from src.nan_itself.utils.llm import (
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
    Usage,
)
from types import MappingProxyType
from pathlib import Path


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def generate_complete(self, request):
        self.requests.append(request)

        if not self.responses:
            raise RuntimeError(
                "FakeLLM has no more responses"
            )

        response = self.responses.pop(0)

        return response


class FakeModules:
    def __init__(self):
        self.snapshot_value = {
            "todo": {
                "active": 3,
            },
        }

        self.queries = []

    def snapshot(self):
        return self.snapshot_value

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

        return [
            "There are 3 active TODO items."
        ]


def make_skill(
    name="core",
):
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
            f"You are using the {name} skill."
        ),
        scripts=(),
        references=(),
        assets=(),
    )


class FakeSkills:
    def __init__(self, skills):
        self.skills = skills

    def names(self):
        return tuple(self.skills)

    def activate(self, name):
        return self.skills[name]

    def catalog(self):
        return [
            self.skills[name].metadata
            for name in sorted(self.skills)
        ]


def make_provider_runtime():
    runtime = ProviderRuntime()

    return runtime


@pytest.mark.asyncio
async def test_core_agent_produces_final_response():
    llm = FakeLLM([
        LLMResponse(
            content="Hello!",
            tool_calls=[],
            model="fake",
            usage=Usage(),
            provider="openai",
            finish_reason="stop",
        )
    ])

    modules = FakeModules()
    providers = make_provider_runtime()

    core_skill = make_skill()

    agent = CoreAgent(
        llm=llm,
        modules=modules,
        providers=providers,
        skills=FakeSkills({
            "core": core_skill,
        }),
        core_skill=core_skill,
    )

    result = await agent.run(
        "Hello"
    )

    assert result.content == "Hello!"

    assert len(llm.requests) == 1

    request = llm.requests[0]

    assert request.messages[0].role == "system"
    assert "core skill" in (
        request.messages[0].content or ""
    )

    assert (
        "There are 3 active TODO items."
        in request.messages[0].content
    )


@pytest.mark.asyncio
async def test_core_agent_executes_mcp_tool_then_continues():
    from mcp.types import Tool

    tool = Tool(
        name="echo",
        description="Echo input.",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                }
            },
            "required": ["text"],
        },
    )

    class FakeSession:
        async def list_tools(self):
            return type(
                "Result",
                (),
                {"tools": [tool]},
            )()

        async def call_tool(
            self,
            name,
            arguments,
        ):
            return {
                "echoed": arguments["text"]
            }

    class FakeStack:
        async def aclose(self):
            pass

    runtime = ProviderRuntime()

    spec = ProviderSpec(
        name="test",
        command="fake",
        origin="builtin",
        source="<test>",
    )

    runtime.providers["test"] = Provider(
        spec=spec,
        stack=FakeStack(),
        session=FakeSession(),
        tools={
            "echo": tool,
        },
    )

    llm = FakeLLM([
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="route",
                    arguments={
                        "provider_name": "test",
                    },
                )
            ],
            model="fake",
            usage=Usage(),
            provider="openai",
            finish_reason="tool_calls",
        ),
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call-2",
                    name="echo",
                    arguments={
                        "text": "hello",
                    },
                )
            ],
            model="fake",
            usage=Usage(),
            provider="openai",
            finish_reason="tool_calls",
        ),
        LLMResponse(
            content="done",
            tool_calls=[],
            model="fake",
            usage=Usage(),
            provider="openai",
            finish_reason="stop",
        ),
    ])

    modules = FakeModules()
    core_skill = make_skill()

    agent = CoreAgent(
        llm=llm,
        modules=modules,
        providers=runtime,
        skills=FakeSkills({
            "core": core_skill,
        }),
        core_skill=core_skill,
    )

    result = await agent.run(
        "echo hello"
    )

    assert result.content == "done"

    assert len(llm.requests) == 3

    # The final request contains the tool result.
    final_request = llm.requests[-1]

    assert any(
        message.role == "tool"
        for message in final_request.messages
    )


@pytest.mark.asyncio
async def test_subagent_is_parallel_and_shares_snapshot():
    responses = [
        # Main dispatches.
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="dispatch-1",
                    name="dispatch_subagent",
                    arguments={
                        "task": "do work",
                    },
                )
            ],
            model="fake",
            usage=Usage(),
            provider="openai",
            finish_reason="tool_calls",
        ),
        # Main final response.
        LLMResponse(
            content="main finished",
            tool_calls=[],
            model="fake",
            usage=Usage(),
            provider="openai",
            finish_reason="stop",
        ),
    ]

    llm = FakeLLM(responses)

    modules = FakeModules()
    core_skill = make_skill()

    agent = CoreAgent(
        llm=llm,
        modules=modules,
        providers=ProviderRuntime(),
        skills=FakeSkills({
            "core": core_skill,
        }),
        core_skill=core_skill,
    )

    result = await agent.run(
        "dispatch something"
    )

    assert result.content == "main finished"

    # Two queries:
    # Main + Subagent.
    for _ in range(100):
        if len(modules.queries) >= 2:
            break

        await asyncio.sleep(0.01)

    assert len(modules.queries) == 2

    main_query = modules.queries[0]
    sub_query = modules.queries[1]

    assert main_query[2] is sub_query[2]

    assert main_query[0] != sub_query[0]


@pytest.mark.asyncio
async def test_subagent_can_switch_its_own_skill():
    llm = FakeLLM([
        # Main dispatches a child (no skill argument exists).
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="dispatch",
                    name="dispatch_subagent",
                    arguments={
                        "task": "research",
                    },
                )
            ],
            model="fake",
            usage=Usage(),
            provider="openai",
            finish_reason="tool_calls",
        ),
        LLMResponse(
            content="main",
            tool_calls=[],
            model="fake",
            usage=Usage(),
            provider="openai",
            finish_reason="stop",
        ),
        # The child switches itself to the research skill.
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="switch",
                    name="activate_skill",
                    arguments={
                        "name": "research",
                    },
                )
            ],
            model="fake",
            usage=Usage(),
            provider="openai",
            finish_reason="tool_calls",
        ),
        LLMResponse(
            content="subagent",
            tool_calls=[],
            model="fake",
            usage=Usage(),
            provider="openai",
            finish_reason="stop",
        ),
    ])

    modules = FakeModules()
    core = make_skill("core")
    research = make_skill("research")

    agent = CoreAgent(
        llm=llm,
        modules=modules,
        providers=ProviderRuntime(),
        skills=FakeSkills({
            "core": core,
            "research": research,
        }),
        core_skill=core,
    )

    result = await agent.run(
        "delegate research"
    )

    assert result.content == "main"

    for _ in range(100):
        if len(llm.requests) >= 4:
            break

        await asyncio.sleep(0.01)

    first_child_request = llm.requests[2]
    switched_request = llm.requests[3]

    # The child starts on the inherited core skill...
    first_system = (
        first_child_request.messages[0].content
        or ""
    )

    assert (
        "You are using the research skill."
        not in first_system
    )

    assert (
        "You are using the core skill."
        in first_system
    )

    # ...and sees the skill catalog for switching.
    assert (
        "[Available Skills]"
        in first_system
    )

    # After its own activate_skill call, the research skill
    # instructions take effect.
    switched_system = (
        switched_request.messages[0].content
        or ""
    )

    assert (
        "You are using the research skill."
        in switched_system
    )


@pytest.mark.asyncio
async def test_recent_history_is_bounded():
    llm = FakeLLM([
        LLMResponse(
            content=f"answer-{index}",
            tool_calls=[],
            model="fake",
            usage=Usage(),
            provider="openai",
            finish_reason="stop",
        )
        for index in range(6)
    ])

    modules = FakeModules()
    core = make_skill()

    agent = CoreAgent(
        llm=llm,
        modules=modules,
        providers=ProviderRuntime(),
        skills=FakeSkills({
            "core": core,
        }),
        core_skill=core,
        history_token_budget=55,
    )

    for index in range(6):
        await agent.run(
            f"user-{index}"
        )

    assert len(agent._main_history) == 4


@pytest.mark.asyncio
async def test_sleep_tool_blocks_only_current_agent_execution():
    llm = FakeLLM([
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="sleep",
                    name="sleep",
                    arguments={
                        "seconds": 0.01,
                    },
                )
            ],
            model="fake",
            usage=Usage(),
            provider="openai",
            finish_reason="tool_calls",
        ),
        LLMResponse(
            content="awake",
            tool_calls=[],
            model="fake",
            usage=Usage(),
            provider="openai",
            finish_reason="stop",
        ),
    ])

    modules = FakeModules()
    core = make_skill()

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
        "wait a moment"
    )

    assert result.content == "awake"
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pytest

from src.nan_itself.agent.core import (
    CoreAgent,
)
from src.nan_itself.agent.engine import (
    StepEngine,
)
from src.nan_itself.skills import (
    UnknownSkillError,
    Skill,
    SkillMetadata,
)
from src.nan_itself.tools import (
    ProviderRuntime,
)
from src.nan_itself.utils.llm import (
    LLMResponse,
    ToolCall,
    Usage,
)

_REPORT_PREFIX = "[Subagent Report]"


# ============================================================================
# Helpers
# ============================================================================


def make_skill(
    name: str = "core",
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
            f"You are using the {name} skill."
        ),
        scripts=(),
        references=(),
        assets=(),
    )


class FakeModules:

    def __init__(self, *args, **kwargs):
        self.turn_records: list = []

    def deliver_turn(self, record):
        self.turn_records.append(record)

    def snapshot(self):
        return {}

    async def query_snapshot(
        self,
        turn,
        snapshot,
    ):
        return []


class FakeSkills:
    def __init__(self, skills):
        self.skills = skills

    def names(self):
        return tuple(self.skills)

    def activate(self, name):
        try:
            return self.skills[name]
        except KeyError:
            raise UnknownSkillError(f"Unknown Skill: {name}")

    def catalog(self):
        return [
            self.skills[name].metadata
            for name in sorted(self.skills)
        ]

    def refresh(self):
        self.refresh_calls = (
            getattr(self, "refresh_calls", 0) + 1
        )


@dataclass
class FakeResult:
    content: str
    messages: tuple = ()
    response: object = None


class SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def generate_complete(
        self,
        request,
    ):
        # Yield twice so a dispatched child already sitting in
        # the ready queue deterministically overtakes the parent
        # before the next response is popped (asyncio's ready
        # queue is FIFO and sleep(0) requeues at the end).
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.requests.append(request)

        if not self.responses:
            raise AssertionError(
                "SequenceLLM ran out of responses"
            )

        return self.responses.pop(0)


def response(
    *,
    text=None,
    calls=None,
    finish_reason="stop",
):
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
    call_id,
    name,
    arguments,
):
    return ToolCall(
        id=call_id,
        name=name,
        arguments=arguments,
    )


def make_agent(
    llm,
    **kwargs,
):
    core = make_skill()

    return CoreAgent(
        llm=llm,
        modules=FakeModules(),
        providers=ProviderRuntime(),
        skills=FakeSkills({
            "core": core,
        }),
        persona_source=lambda: "CORE",
        **kwargs,
    )


def user_reports(messages):
    return [
        message
        for message in messages
        if message.role == "user"
        and _REPORT_PREFIX
        in (message.content or "")
    ]


@pytest.mark.asyncio
async def test_report_is_injected_at_next_step_boundary():
    llm = SequenceLLM([
        # Main dispatches and then sleeps within the same
        # response. The real sleep suspends the parent, so the
        # child deterministically finishes before the next step.
        response(
            calls=[
                tool_call(
                    call_id="d",
                    name="dispatch_subagent",
                    arguments={
                        "task": "quick work",
                    },
                ),
                tool_call(
                    call_id="s",
                    name="sleep",
                    arguments={
                        "seconds": 0.05,
                    },
                ),
            ],
            finish_reason="tool_calls",
        ),
        # Child finishes on its own.
        response(
            text="quick result",
        ),
        response(
            text="wrapped up",
        ),
    ])

    agent = make_agent(llm)

    result = await agent.run("delegate")

    assert result.content == "wrapped up"

    assert len(llm.requests) == 3

    final_request = llm.requests[2]

    reports = user_reports(
        final_request.messages
    )

    assert len(reports) == 1

    content = reports[0].content or ""

    assert "quick result" in content
    assert "status: completed" in content


@pytest.mark.asyncio
async def test_failed_child_delivers_failed_report():
    core = make_skill()

    llm = SequenceLLM([
        response(
            calls=[
                tool_call(
                    call_id="d",
                    name="dispatch_subagent",
                    arguments={
                        "task": "doomed work",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        # Give the doomed child a beat to actually fail.
        response(
            calls=[
                tool_call(
                    call_id="s",
                    name="sleep",
                    arguments={"seconds": 0.05},
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="handled failure",
        ),
    ])

    class DoomedEngine(StepEngine):
        async def execute(self, **kwargs):
            if kwargs["context"].depth == 1:
                raise RuntimeError("boom")

            return await super().execute(**kwargs)

    agent = make_agent(llm)

    agent.engine = DoomedEngine(
        llm=llm,
        modules=agent.modules,
        providers=agent.providers,
        skills=agent.skills,
        agent_runtime=agent.agent_runtime,
    )

    result = await agent.run("turn")

    assert result.content == "handled failure"

    reports = user_reports(llm.requests[2].messages)

    assert len(reports) == 1

    body = reports[0].content or ""

    assert "status: failed" in body
    assert "boom" in body


@pytest.mark.asyncio
async def test_late_report_arrives_next_turn():
    core = make_skill()

    started = asyncio.Event()

    release = asyncio.Event()

    llm = SequenceLLM([
        # Turn 1: dispatch then answer without waiting.
        response(
            calls=[
                tool_call(
                    call_id="d",
                    name="dispatch_subagent",
                    arguments={
                        "task": "slow work",
                    },
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="main done",
        ),
        # Turn 2.
        response(
            text="second turn",
        ),
    ])

    class GatedEngine(StepEngine):
        async def execute(self, **kwargs):
            if kwargs["context"].depth == 1:
                started.set()

                await release.wait()

                return FakeResult(
                    content="late sub work",
                )

            return await super().execute(**kwargs)

    agent = make_agent(llm)

    agent.engine = GatedEngine(
        llm=llm,
        modules=agent.modules,
        providers=agent.providers,
        skills=agent.skills,
        agent_runtime=agent.agent_runtime,
    )

    first = await agent.run("turn one")

    assert first.content == "main done"

    await asyncio.wait_for(
        started.wait(),
        timeout=2.0,
    )

    # Turn 1 saw no report: the child was still running.
    assert user_reports(
        llm.requests[1].messages
    ) == []

    release.set()

    for _ in range(200):
        if agent.has_pending_reports():
            break

        await asyncio.sleep(0.01)

    assert agent.has_pending_reports()

    second = await agent.run("turn two")

    assert second.content == "second turn"

    second_request = llm.requests[2]

    reports = user_reports(
        second_request.messages
    )

    assert len(reports) == 1

    report_index = (
        second_request.messages.index(
            reports[0]
        )
    )

    input_index = (
        second_request.messages.index(
            [
                message
                for message in (
                    second_request.messages
                )
                if message.role == "user"
                and message.content
                == "turn two"
            ][0]
        )
    )

    # The late report precedes this turn's user input.
    assert report_index < input_index

    assert "late sub work" in (
        reports[0].content or ""
    )

    # Delivered exactly once.
    assert not agent.has_pending_reports()


# ============================================================================
# Tool surface
# ============================================================================


@pytest.mark.asyncio
async def test_agent_tool_surface_after_removal():
    llm = SequenceLLM([
        response(text="hi"),
    ])

    agent = make_agent(llm)

    await agent.run("hello")

    tools = {
        tool.name: tool
        for tool in llm.requests[0].tools
    }

    assert set(tools) == {
        "route",
        "sleep",
        "dispatch_subagent",
    }

    sleep_schema = tools["sleep"].input_schema

    assert set(
        sleep_schema["properties"]
    ) == {"seconds"}
    assert sleep_schema["required"] == [
        "seconds",
    ]


# ============================================================================
# Token estimation
# ============================================================================



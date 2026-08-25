from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pytest

from src.nan_itself.agent.core import (
    CoreAgent,
    _estimate_tokens,
)
from src.nan_itself.skills.facade import (
    Skill,
    SkillMetadata,
)
from src.nan_itself.tools.facade import (
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
        return self.skills[name]

    def catalog(self):
        return [
            self.skills[name].metadata
            for name in sorted(self.skills)
        ]


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
        core_skill=core,
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


# ============================================================================
# Barrier
# ============================================================================


@pytest.mark.asyncio
async def test_barrier_returns_all_reports_at_once():
    core = make_skill()

    started = [
        asyncio.Event(),
        asyncio.Event(),
    ]

    release = asyncio.Event()

    class BlockedCore(CoreAgent):
        async def _run_agent(
            self,
            *,
            context,
            user_input,
            history,
            seed_reports=None,
        ):
            if context.depth == 1:
                index = (
                    0
                    if user_input == "work A"
                    else 1
                )

                started[index].set()

                await release.wait()

                return FakeResult(
                    content=f"sub::{user_input}",
                )

            return await super()._run_agent(
                context=context,
                user_input=user_input,
                history=history,
                seed_reports=seed_reports,
            )

    llm = SequenceLLM([
        # One response fans out two children at once.
        response(
            calls=[
                tool_call(
                    call_id="d1",
                    name="dispatch_subagent",
                    arguments={
                        "task": "work A",
                    },
                ),
                tool_call(
                    call_id="d2",
                    name="dispatch_subagent",
                    arguments={
                        "task": "work B",
                    },
                ),
            ],
            finish_reason="tool_calls",
        ),
        # Model uses the barrier.
        response(
            calls=[
                tool_call(
                    call_id="barrier",
                    name="await_subagents",
                    arguments={},
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="all done",
        ),
    ])

    agent = BlockedCore(
        llm=llm,
        modules=FakeModules(),
        providers=ProviderRuntime(),
        skills=FakeSkills({
            "core": core,
        }),
        core_skill=core,
    )

    async def releaser():
        for event in started:
            await asyncio.wait_for(
                event.wait(),
                timeout=2.0,
            )

        await asyncio.sleep(0.01)

        release.set()

    asyncio.create_task(releaser())

    result = await agent.run(
        "fan out",
    )

    assert result.content == "all done"

    final_messages = (
        llm.requests[2].messages
    )

    tool_text = "\n".join(
        message.content or ""
        for message in final_messages
        if message.role == "tool"
    )

    assert (
        "2 subagent(s) finished"
        in tool_text
    )
    assert "sub::work A" in tool_text
    assert "sub::work B" in tool_text

    # Reports delivered through the barrier are not
    # duplicated as injected user messages afterwards.
    assert user_reports(final_messages) == []


@pytest.mark.asyncio
async def test_barrier_without_outstanding_children_is_immediate():
    llm = SequenceLLM([
        response(text="done"),
    ])

    agent = make_agent(llm)

    result = await agent.run("hello")

    assert result.content == "done"

    barrier_result = (
        await agent._execute_await_subagents([])
    )

    assert barrier_result == (
        "No outstanding subagents."
    )


# ============================================================================
# Automatic step-boundary delivery
# ============================================================================


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

    started = asyncio.Event()

    release = asyncio.Event()

    class BoomCore(CoreAgent):
        async def _run_agent(
            self,
            *,
            context,
            user_input,
            history,
            seed_reports=None,
        ):
            if context.depth == 1:
                started.set()

                await release.wait()

                raise RuntimeError("boom")

            return await super()._run_agent(
                context=context,
                user_input=user_input,
                history=history,
                seed_reports=seed_reports,
            )

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
        response(
            calls=[
                tool_call(
                    call_id="b",
                    name="await_subagents",
                    arguments={},
                )
            ],
            finish_reason="tool_calls",
        ),
        response(
            text="handled failure",
        ),
    ])

    agent = BoomCore(
        llm=llm,
        modules=FakeModules(),
        providers=ProviderRuntime(),
        skills=FakeSkills({
            "core": core,
        }),
        core_skill=core,
    )

    async def releaser():
        await asyncio.wait_for(
            started.wait(),
            timeout=2.0,
        )

        await asyncio.sleep(0.01)

        release.set()

    asyncio.create_task(releaser())

    result = await agent.run("delegate")

    assert result.content == "handled failure"

    final_messages = (
        llm.requests[2].messages
    )

    tool_text = "\n".join(
        message.content or ""
        for message in final_messages
        if message.role == "tool"
    )

    assert "status: failed" in tool_text
    assert "boom" in tool_text


# ============================================================================
# Late delivery across turns
# ============================================================================


@pytest.mark.asyncio
async def test_late_report_arrives_next_turn():
    core = make_skill()

    started = asyncio.Event()

    release = asyncio.Event()

    class SlowCore(CoreAgent):
        async def _run_agent(
            self,
            *,
            context,
            user_input,
            history,
            seed_reports=None,
        ):
            if context.depth == 1:
                started.set()

                await release.wait()

                return FakeResult(
                    content="late sub work",
                )

            return await super()._run_agent(
                context=context,
                user_input=user_input,
                history=history,
                seed_reports=seed_reports,
            )

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

    agent = SlowCore(
        llm=llm,
        modules=FakeModules(),
        providers=ProviderRuntime(),
        skills=FakeSkills({
            "core": core,
        }),
        core_skill=core,
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
        if agent._late_reports:
            break

        await asyncio.sleep(0.01)

    assert len(agent._late_reports) == 1

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
    assert len(agent._late_reports) == 0


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
        "await_subagents",
    }

    sleep_schema = tools["sleep"].input_schema

    assert set(
        sleep_schema["properties"]
    ) == {"seconds"}
    assert sleep_schema["required"] == [
        "seconds",
    ]

    barrier_schema = (
        tools["await_subagents"].input_schema
    )

    assert barrier_schema["properties"] == {}


# ============================================================================
# Token estimation
# ============================================================================


def test_token_estimator_shapes():
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("abcd") == 1
    assert _estimate_tokens("一二三") == 3

    # Two CJK characters plus four ASCII ones.
    assert _estimate_tokens("十二abcd") == 3

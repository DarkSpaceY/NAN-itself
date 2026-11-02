"""The turn engine: inject/think/act phases and stream emission."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from nan_fakes import FakeLLM, FakeStreamEvent

from nan.agent.engine import TurnEngine
from nan.events import sink
from nan.llm.types import (
    Message,
    ThinkingBlock,
    ToolCall,
    ToolDefinition,
    Usage,
)
from nan.turn import DeltaEvent, InboxMessage, Injection, Turn

pytestmark = pytest.mark.asyncio


class StreamingLLM:
    """A provider whose stream is exactly the scripted event list."""

    model = "fake"

    def __init__(self, events: list[FakeStreamEvent]) -> None:
        self.events = events

    async def generate(self, request: Any) -> AsyncIterator[FakeStreamEvent]:
        for event in self.events:
            yield event


def _turn(**kwargs: Any) -> Turn:
    return Turn(agent_id="a1", parent_id=None, depth=0, **kwargs)


async def test_inject_appends_service_deltas_and_marks_the_phase(no_sink: Any) -> None:
    turn = _turn(delta=(DeltaEvent(section="agent", line="paused"),))

    contributions = (Injection(service_id="clock", text="09:00", started_at=1.0, ended_at=1.5),)
    deltas = [DeltaEvent(section="service", line="+clock online")]

    async def inject(_turn: Turn) -> tuple[tuple[Injection, ...], list[DeltaEvent]]:
        return contributions, deltas

    injected = await TurnEngine(FakeLLM()).inject(turn, inject=inject)

    assert injected.phase == "injected"
    assert injected.injections == contributions
    assert injected.delta == (*turn.delta, *deltas)

    collected = next(event for event in sink.history() if event["t"] == "injections_collected")

    assert collected["content"]["services"] == [
        {"service_id": "clock", "ok": True, "duration_s": 0.5, "text_len": 5}
    ]


async def test_think_builds_persona_history_then_observation() -> None:
    llm = FakeLLM()

    engine = TurnEngine(llm)

    turn = _turn(inbox=(InboxMessage(kind="user", source="user", text="now"),))

    tools = [ToolDefinition(name="sleep", description="wait", input_schema={})]

    thought = await engine.think(
        turn,
        persona="PERSONA",
        history=[Message(role="assistant", content="past")],
        tools=tools,
        retries=2,
    )

    request = llm.requests[0]

    assert [message.role for message in request.messages] == ["system", "assistant", "user"]
    assert request.messages[0].content == "PERSONA"
    assert request.messages[1].content == "past"
    assert "now" in (request.messages[2].content or "")
    assert request.tools == tools

    assert thought.phase == "thought"
    assert thought.reply == "ok"
    assert thought.llm is not None
    assert thought.llm.retries == 2
    assert thought.llm.finish_reason == "stop"


async def test_think_records_calls_and_thinking() -> None:
    llm = FakeLLM()

    llm.thinking = "REASONING"

    llm.script("tool-me", [("tool", "sleep", {"seconds": 0})])

    turn = _turn(inbox=(InboxMessage(kind="user", source="user", text="tool-me"),))

    thought = await TurnEngine(llm).think(
        turn,
        persona="P",
        history=[],
        tools=[],
        retries=0,
    )

    assert thought.reply == ""
    assert [call.name for call in thought.calls] == ["sleep"]
    assert [block.text for block in thought.thinking] == ["REASONING"]


async def test_act_runs_calls_in_order_and_records_each_result() -> None:
    first = ToolCall(id="a", name="sleep", arguments={"seconds": 0})
    second = ToolCall(id="b", name="send", arguments={"text": "hi"})

    turn = _turn(phase="thought", calls=(first, second))

    executed: list[str] = []
    recorded: list[list[str]] = []

    async def execute(call: ToolCall) -> str:
        executed.append(call.id)

        return f"result-{call.id}"

    async def record(current: Turn) -> None:
        recorded.append([result.call_id for result in current.results])

    acted = await TurnEngine(FakeLLM()).act(turn, execute=execute, record=record)

    assert executed == ["a", "b"]
    assert recorded == [["a"], ["a", "b"]]
    assert [result.output for result in acted.results] == ["result-a", "result-b"]
    assert acted.phase == "acted"


async def test_generate_emits_the_stream_lifecycle(no_sink: Any) -> None:
    llm = StreamingLLM(
        [
            FakeStreamEvent("thinking", thinking_text="why"),
            FakeStreamEvent("text", text="hello"),
            FakeStreamEvent(
                "done",
                usage=Usage(prompt_tokens=3, completion_tokens=4),
                finish_reason="stop",
                thinking=[ThinkingBlock(text="why")],
            ),
        ]
    )

    response = await TurnEngine(llm).generate(agent_id="a1", depth=0, request=type("R", (), {})())

    kinds = [event["t"] for event in sink.history()]

    # thinking_done closes after the stream ends: the answer's output
    # frames are already in flight when reasoning is finished.
    assert kinds == [
        "thinking_started",
        "thinking_delta",
        "output_started",
        "output_delta",
        "thinking_done",
        "output_done",
    ]
    assert sink.history()[1]["content"]["text"] == "why"
    assert sink.history()[3]["content"]["text"] == "hello"

    assert response.content == "hello"
    assert [block.text for block in response.thinking] == ["why"]
    assert response.usage == Usage(prompt_tokens=3, completion_tokens=4)
    assert response.finish_reason == "stop"


async def test_generate_tool_answer_cancels_the_output_stream(no_sink: Any) -> None:
    call = ToolCall(id="c1", name="sleep", arguments={})

    llm = StreamingLLM(
        [
            FakeStreamEvent("text", text="thinking out loud"),
            FakeStreamEvent("tool_call", tool_call=call),
            FakeStreamEvent("done", usage=Usage(), finish_reason="tool_calls"),
        ]
    )

    response = await TurnEngine(llm).generate(agent_id="a1", depth=0, request=type("R", (), {})())

    assert [event["t"] for event in sink.history()] == [
        "output_started",
        "output_delta",
        "output_cancelled",
    ]
    assert response.tool_calls == [call]
    assert response.finish_reason == "tool_calls"


async def test_generate_tool_only_answer_streams_no_output(no_sink: Any) -> None:
    call = ToolCall(id="c1", name="sleep", arguments={})

    llm = StreamingLLM(
        [FakeStreamEvent("tool_call", tool_call=call), FakeStreamEvent("done", usage=Usage())]
    )

    response = await TurnEngine(llm).generate(agent_id="a1", depth=0, request=type("R", (), {})())

    assert sink.history() == []
    assert response.content is None
    assert response.tool_calls == [call]
    assert response.finish_reason == "unknown"


async def test_generate_synthesizes_thinking_blocks_from_deltas(no_sink: Any) -> None:
    llm = StreamingLLM(
        [
            FakeStreamEvent("thinking", thinking_text="a"),
            FakeStreamEvent("thinking", thinking_text="b"),
            FakeStreamEvent("done", usage=Usage(), finish_reason="stop"),
        ]
    )

    response = await TurnEngine(llm).generate(agent_id="a1", depth=0, request=type("R", (), {})())

    assert [block.text for block in response.thinking] == ["ab"]


async def test_generate_with_a_real_request_type(no_sink: Any) -> None:
    """The provider receives the DTO it was handed, unchanged."""
    from nan.llm.types import LLMRequest

    llm = StreamingLLM([FakeStreamEvent("done", usage=Usage())])

    request = LLMRequest(messages=[Message(role="user", content="hi")])

    await TurnEngine(llm).generate(agent_id="a1", depth=0, request=request)

    assert request.messages[0].content == "hi"


async def test_act_with_no_calls_still_advances_the_phase() -> None:
    acted = await TurnEngine(FakeLLM()).act(
        _turn(phase="thought"), execute=lambda _call: asyncio.sleep(0), record=lambda _turn: None
    )

    assert acted.phase == "acted"
    assert acted.results == ()


async def test_think_reports_the_model_name_and_finish_reason() -> None:
    llm = FakeLLM()

    thought = await TurnEngine(llm).think(_turn(), persona="P", history=[], tools=[], retries=0)

    assert thought.llm is not None
    assert thought.llm.model == llm.model
    assert thought.llm.finish_reason == "stop"

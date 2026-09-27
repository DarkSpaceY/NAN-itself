"""
Rendering determinism of the prompt assembly.

The engine is the only place a Turn becomes model messages, and
the render must be byte-stable: each turn's request extends the
previous turn's request exactly, which is what keeps the
provider's message prefix cacheable.
"""

from __future__ import annotations

from dataclasses import replace

from nan_itself.agent.model import Report
from nan_itself.agent.prompts import (
    build_messages,
    render_turn,
)
from nan_itself.modules.model import Turn
from nan_itself.utils.llm import ToolCall


def make_turn(
    **overrides,
):
    turn = Turn(
        agent_hash="hash",
        parent_hash=None,
        depth=1,
        task="do it",
        world={},
        persona="persona",
    )

    return replace(turn, **overrides)


def test_render_is_deterministic():
    turn = make_turn(
        ambient=("module line",),
        reply="hi",
    )

    assert (
        build_messages(turn)
        == build_messages(turn)
    )

    assert (
        render_turn(turn)
        == render_turn(turn)
    )


def test_in_flight_turn_renders_observation_only():
    turn = make_turn(
        ambient=("a",),
        reports=(
            Report(
                agent_id="abc12345",
                task="child task",
                status="completed",
                body="child result",
            ),
        ),
    )

    rendered = render_turn(turn)

    # No flow yet: the observation alone.
    assert [
        m.role for m in rendered
    ] == ["user"]

    content = rendered[0].content

    assert "<task>\ndo it\n</task>" in content

    assert "<subagent_report>" in content

    assert "id: abc12345" in content

    assert "child result" in content


def test_failed_turn_renders_observation_only():
    # A failed or cancelled turn never got a flow: same as
    # in-flight.
    turn = make_turn(
        error="RuntimeError: boom",
    )

    assert [
        m.role for m in render_turn(turn)
    ] == ["user"]


def test_reply_only_turn_renders_assistant_text():
    turn = make_turn(
        ambient=("a",),
        reply="done",
    )

    rendered = render_turn(turn)

    assert [
        m.role for m in rendered
    ] == ["user", "assistant"]

    assert rendered[1].content == "done"


def test_calls_render_positional_tool_messages():
    calls = (
        ToolCall(
            id="c1",
            name="invoke_tool",
            arguments={},
        ),
        ToolCall(
            id="c2",
            name="sleep",
            arguments={},
        ),
    )

    turn = make_turn(
        reply="",
        calls=calls,
        results=("one", "two"),
    )

    rendered = render_turn(turn)

    assert [
        m.role for m in rendered
    ] == ["user", "assistant", "tool", "tool"]

    assistant = rendered[1]

    assert assistant.content == ""

    assert assistant.tool_calls == list(calls)

    assert rendered[2].tool_call_id == "c1"

    assert rendered[2].content == "one"

    assert rendered[3].tool_call_id == "c2"

    assert rendered[3].content == "two"


def test_partial_results_render_truncated():
    # A verb that raised leaves the results unwritten (or
    # shorter than the calls); rendering truncates to what
    # exists instead of inventing messages.
    turn = make_turn(
        reply="",
        calls=(
            ToolCall(
                id="c1",
                name="invoke_tool",
                arguments={},
            ),
            ToolCall(
                id="c2",
                name="invoke_tool",
                arguments={},
            ),
        ),
        results=("one",),
    )

    rendered = render_turn(turn)

    assert [
        m.role for m in rendered
    ] == ["user", "assistant", "tool"]


def test_next_request_extends_previous_exactly():
    """
    The cacheability lock: the next turn's full request is the
    previous turn's full request plus exactly this turn's
    rendering -- byte for byte.
    """
    first = make_turn(
        ambient=("a1",),
        reply="",
        calls=(
            ToolCall(
                id="c1",
                name="invoke_tool",
                arguments={},
            ),
        ),
        results=("tool ok",),
    )

    second = make_turn(
        history=render_turn(first),
        ambient=("a2",),
        reply="done",
    )

    first_request = build_messages(first)

    second_request = build_messages(second)

    assert second_request[
        : len(first_request)
    ] == first_request

    assert [
        m.role for m in second_request
    ] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
    ]

    assert second_request[-1].content == "done"

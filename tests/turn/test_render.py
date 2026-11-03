"""Rendering: the observation schema and the assistant/tool run."""

from __future__ import annotations

from dataclasses import replace

from nan.llm.types import ThinkingBlock, VerbCall
from nan.turn import (
    INTERRUPTED,
    DeltaEvent,
    InboxMessage,
    Injection,
    Turn,
    VerbResult,
    freeze_turn,
    render_turn,
)


def _turn(**kwargs: object) -> Turn:
    return Turn(agent_id="a1", parent_id=None, depth=0, **kwargs)


def test_empty_observation_is_explicit() -> None:
    messages = render_turn(_turn())

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "(no input this turn)"


def test_delta_groups_by_section_in_first_appearance_order() -> None:
    turn = _turn(
        delta=(
            DeltaEvent(section="service", line="+echo online"),
            DeltaEvent(section="subagent", line="+c1 spawned"),
            DeltaEvent(section="service", line="-echo offline"),
        )
    )

    content = render_turn(turn)[0].content

    assert content == (
        "<delta>\n"
        "<service>+echo online\n-echo offline</service>\n"
        "<subagent>+c1 spawned</subagent>\n"
        "</delta>"
    )


def test_injection_renders_service_blocks_and_errors() -> None:
    turn = _turn(
        injections=(
            Injection(service_id="clock", text="09:00", started_at=1.0, ended_at=1.1),
            Injection(
                service_id="broken", text="timed out", started_at=2.0, ended_at=2.1, ok=False
            ),
        )
    )

    content = render_turn(turn)[0].content

    assert '<service id="clock">\n09:00\n</service>' in content
    assert '<service id="broken">\n[service error] timed out\n</service>' in content


def test_inbox_tags_the_sender_edge_and_event() -> None:
    turn = _turn(
        inbox=(
            InboxMessage(kind="user", source="user", text="hello"),
            InboxMessage(kind="subagent", source="c1", text="report", event="finish"),
        )
    )

    content = render_turn(turn)[0].content

    assert "<user>\nhello\n</user>" in content
    assert '<subagent id="c1" event="finish">\nreport\n</subagent>' in content


def test_observed_turn_renders_only_the_observation() -> None:
    turn = _turn(
        inbox=(InboxMessage(kind="user", source="user", text="hi"),),
        reply="ignored until thought",
        phase="injected",
    )

    messages = render_turn(turn)

    assert len(messages) == 1
    assert messages[0].role == "user"


def test_thought_turn_renders_assistant_then_tools_by_call_id() -> None:
    call = VerbCall(id="c1", name="sleep", arguments={"seconds": 1})

    turn = _turn(
        phase="acted",
        reply="done",
        thinking=(ThinkingBlock(text="reasoning"),),
        verb_calls=(call,),
        verb_results=(
            VerbResult(call_id="c1", name="sleep", output="ok", started_at=1.0, ended_at=2.0),
        ),
    )

    messages = render_turn(turn)

    assert [message.role for message in messages] == ["user", "assistant", "tool"]

    assistant = messages[1]

    assert assistant.content == "done"
    assert [block.text for block in assistant.thinking] == ["reasoning"]
    assert assistant.tool_calls == [call]

    assert messages[2].tool_call_id == "c1"
    assert messages[2].content == "ok"


def test_a_call_without_a_result_renders_interrupted() -> None:
    call = VerbCall(id="c1", name="sleep", arguments={})

    turn = _turn(phase="thought", verb_calls=(call,), verb_results=())

    tool = render_turn(turn)[2]

    assert tool.role == "tool"
    assert tool.tool_call_id == "c1"
    assert tool.content == INTERRUPTED


def test_freeze_is_idempotent_and_carries_the_run() -> None:
    turn = _turn(
        inbox=(InboxMessage(kind="user", source="user", text="hi"),),
        reply="answer",
        phase="acted",
        verb_calls=(),
    )

    frozen = freeze_turn(turn)

    assert frozen.rendered == render_turn(turn)
    assert freeze_turn(frozen) is frozen


def test_render_is_byte_stable() -> None:
    turn = _turn(
        delta=(DeltaEvent(section="agent", line="paused"),),
        inbox=(InboxMessage(kind="user", source="user", text="hi"),),
        phase="committed",
        reply="answer",
        ended_at=1.0,
    )

    assert render_turn(turn) == render_turn(replace(turn))

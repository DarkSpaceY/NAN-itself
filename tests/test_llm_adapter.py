from __future__ import annotations

from src.nan_itself.utils.llm import (
    LLMProvider,
    Message,
    ToolCall,
)


def test_adjacent_user_messages_are_merged():
    messages = [
        Message(
            role="user",
            content="first",
        ),
        Message(
            role="user",
            content="[Subagent Report]\nstatus: completed",
        ),
    ]

    system, result = (
        LLMProvider._anthropic_messages(
            messages
        )
    )

    assert system is None

    # Anthropic requires alternating roles; consecutive user
    # messages coalesce into one multi-block message.
    assert len(result) == 1

    blocks = result[0]["content"]

    assert [block["text"] for block in blocks] == [
        "first",
        "[Subagent Report]\nstatus: completed",
    ]


def test_report_after_tool_result_merges_with_tool_result_first():
    messages = [
        Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="dispatch_subagent",
                    arguments={"task": "x"},
                )
            ],
        ),
        Message(
            role="tool",
            tool_call_id="t1",
            content="Subagent dispatched.",
        ),
        Message(
            role="user",
            content="[Subagent Report]\nid: abc",
        ),
    ]

    _, result = LLMProvider._anthropic_messages(
        messages
    )

    # assistant + merged user(tool_result, text)
    assert len(result) == 2

    merged = result[1]

    assert merged["role"] == "user"

    blocks = merged["content"]

    assert blocks[0]["type"] == "tool_result"
    assert (
        blocks[0]["tool_use_id"] == "t1"
    )
    assert blocks[1]["type"] == "text"
    assert "[Subagent Report]" in (
        blocks[1]["text"]
    )


def test_single_user_message_becomes_one_block():
    messages = [
        Message(
            role="user",
            content="hello",
        ),
    ]

    _, result = LLMProvider._anthropic_messages(
        messages
    )

    assert len(result) == 1

    assert result[0] == {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "hello",
            }
        ],
    }


def test_three_consecutive_users_merge_in_order():
    messages = [
        Message(role="user", content="a"),
        Message(role="user", content="b"),
        Message(role="user", content="c"),
    ]

    _, result = LLMProvider._anthropic_messages(
        messages
    )

    assert len(result) == 1

    texts = [
        block["text"]
        for block in result[0]["content"]
    ]

    assert texts == ["a", "b", "c"]


def test_empty_user_message_is_skipped_in_anthropic():
    messages = [
        Message(
            role="user",
            content="[Subagent Report]\nstatus: completed",
        ),
        Message(role="user", content=""),
    ]

    _, result = LLMProvider._anthropic_messages(
        messages
    )

    # The empty input contributes nothing visible.
    assert len(result) == 1

    blocks = result[0]["content"]

    assert [block["text"] for block in blocks] == [
        "[Subagent Report]\nstatus: completed",
    ]


def test_empty_user_message_is_skipped_in_openai():
    message = Message(role="user", content="")

    assert (
        LLMProvider._openai_message(message)
        is None
    )

    keep = Message(role="user", content="hello")

    assert LLMProvider._openai_message(keep) == {
        "role": "user",
        "content": "hello",
    }

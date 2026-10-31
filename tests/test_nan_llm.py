"""Provider mapping: reasoning is captured and replayed for strict prefix caching."""

from __future__ import annotations

from typing import Any

from nan.llm import Message, ThinkingBlock
from nan.llm.provider import LLMProvider


def test_openai_replays_structured_reasoning() -> None:
    message = Message(
        role="assistant",
        content="answer",
        thinking=[ThinkingBlock(text="reasoning", signature="sig", format="anthropic")],
    )

    mapped = LLMProvider._openai_message(message)

    assert mapped is not None
    assert mapped["reasoning_details"] == [
        {
            "type": "reasoning.text",
            "text": "reasoning",
            "signature": "sig",
            "format": "anthropic",
        }
    ]


def test_openai_replays_plain_reasoning() -> None:
    message = Message(
        role="assistant",
        content="answer",
        thinking=[ThinkingBlock(text="reasoning")],
    )

    mapped = LLMProvider._openai_message(message)

    assert mapped is not None
    assert mapped["reasoning_content"] == "reasoning"


def test_anthropic_replays_thinking_blocks_first() -> None:
    _system, messages = LLMProvider._anthropic_messages(
        [
            Message(
                role="assistant",
                content="answer",
                thinking=[
                    ThinkingBlock(text="thinking", signature="sig", format="anthropic"),
                    ThinkingBlock(type="encrypted", data="payload"),
                ],
            )
        ]
    )

    blocks: list[dict[str, Any]] = messages[0]["content"]

    assert blocks[0] == {
        "type": "thinking",
        "thinking": "thinking",
        "signature": "sig",
    }
    assert blocks[1] == {"type": "redacted_thinking", "data": "payload"}
    assert blocks[2] == {"type": "text", "text": "answer"}


def test_anthropic_skips_unsigned_thinking() -> None:
    _system, messages = LLMProvider._anthropic_messages(
        [
            Message(
                role="assistant",
                content="answer",
                thinking=[ThinkingBlock(text="unsigned")],
            )
        ]
    )

    assert messages[0]["content"] == [{"type": "text", "text": "answer"}]


def test_merge_reasoning_details_accumulates_by_index() -> None:
    blocks: dict[int, ThinkingBlock] = {}

    assert (
        LLMProvider._merge_reasoning_detail(
            blocks, 0, {"type": "reasoning.text", "text": "a", "format": "f"}
        )
        == "a"
    )
    assert (
        LLMProvider._merge_reasoning_detail(
            blocks, 0, {"type": "reasoning.text", "text": "b", "signature": "sig"}
        )
        == "b"
    )
    assert blocks[0].text == "ab"
    assert blocks[0].signature == "sig"
    assert blocks[0].format == "f"

    assert (
        LLMProvider._merge_reasoning_detail(
            blocks, 1, {"type": "reasoning.encrypted", "data": "payload"}
        )
        is None
    )
    assert blocks[1].type == "encrypted"
    assert blocks[1].data == "payload"

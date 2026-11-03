"""Provider mapping: reasoning is captured and replayed for strict prefix caching."""

from __future__ import annotations

from typing import Any

from nan.llm import Message, ThinkingBlock
from nan.llm.provider import LLMProvider


def _provider() -> LLMProvider:
    """The mapping seams need no SDK clients, so skip the constructor."""
    return LLMProvider.__new__(LLMProvider)


def test_a_well_formed_tool_call_carries_no_error() -> None:
    call = _provider()._build_verb_call(
        tool_id="k1", name="shell", raw_arguments='{"command": "ls"}'
    )

    assert call.id == "k1"
    assert call.name == "shell"
    assert call.arguments == {"command": "ls"}
    assert call.error is None

    # The error is a run-time marker only: snapshots keep their shape.
    assert "error" not in call.model_dump()


def test_broken_json_is_returned_as_the_call_error() -> None:
    call = _provider()._build_verb_call(tool_id="k1", name="shell", raw_arguments="{oops")

    assert call.arguments == {}
    assert call.error is not None and "not valid JSON" in call.error


def test_the_error_echo_is_bounded() -> None:
    blob = "{" + "x" * 1000

    call = _provider()._build_verb_call(tool_id="k1", name="shell", raw_arguments=blob)

    assert call.error is not None
    assert len(call.error) < LLMProvider._MAX_ARGUMENT_ECHO + 100


def test_non_object_arguments_are_an_error() -> None:
    call = _provider()._build_verb_call(tool_id="k1", name="shell", raw_arguments="[1, 2]")

    assert call.arguments == {}
    assert call.error is not None and "JSON object" in call.error


def test_missing_id_and_name_are_normalized() -> None:
    provider = _provider()

    unnamed = provider._build_verb_call(tool_id="", name="", raw_arguments="")

    # An id is the provider's to supply, so it is synthesized silently;
    # a nameless call is the model's mistake and is reported.
    assert unnamed.id.startswith("call-")
    assert unnamed.name == LLMProvider._INVALID_TOOL_NAME
    assert unnamed.error is not None and "without a name" in unnamed.error


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

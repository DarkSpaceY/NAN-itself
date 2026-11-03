"""
LLM protocol vocabulary: the shared DTOs every layer speaks --
messages, verb calls, usage, the streaming event contract. Pure
data, no I/O.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

FinishReason = Literal[
    "stop",
    "tool_calls",
    "length",
    "unknown",
]


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class VerbDefinition(BaseModel):
    """
    Model-facing verb definition.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


class VerbCall(BaseModel):
    """
    One complete model-generated verb call.

    `error` carries the adapter's reason when parsing fails: the
    runtime returns that text as this call's result. It lives one
    run and is excluded from snapshots.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    error: str | None = Field(default=None, exclude=True)


class ThinkingBlock(BaseModel):
    """
    One piece of model reasoning, preserved verbatim.

    text       visible reasoning text
    signature  provider signature proving provenance (Anthropic)
    data       encrypted reasoning payload (redacted / encrypted)
    format     provider format tag when one exists (OpenRouter)
    """

    type: Literal["text", "encrypted"] = "text"
    text: str = ""
    signature: str | None = None
    data: str | None = None
    format: str | None = None


class Message(BaseModel):
    """
    Provider-neutral conversation message.

    assistant:
        content may be text
        thinking carries the reasoning that preceded it
        tool_calls may contain model-generated calls

    tool:
        tool_call_id identifies the invocation being answered
        content contains the tool result
    """

    role: Literal[
        "system",
        "user",
        "assistant",
        "tool",
    ]

    content: str | None = None

    thinking: list[ThinkingBlock] = Field(
        default_factory=list,
    )

    tool_calls: list[VerbCall] = Field(
        default_factory=list,
    )

    tool_call_id: str | None = None


class LLMRequest(BaseModel):
    messages: list[Message]

    temperature: float = 0.7

    tools: list[VerbDefinition] = Field(
        default_factory=list,
    )


class LLMResponse(BaseModel):
    """
    Normalized complete model response.
    """

    content: str | None = None

    thinking: list[ThinkingBlock] = Field(
        default_factory=list,
    )

    tool_calls: list[VerbCall] = Field(
        default_factory=list,
    )

    model: str
    usage: Usage

    finish_reason: FinishReason = "unknown"


class LLMStreamEvent(BaseModel):
    """
    Provider-neutral streaming event.

    kind:
        thinking
        text
        tool_call
        done
    """

    kind: Literal[
        "thinking",
        "text",
        "tool_call",
        "done",
    ]

    thinking_text: str | None = None

    thinking: list[ThinkingBlock] | None = None

    text: str | None = None

    tool_call: VerbCall | None = None

    usage: Usage | None = None

    finish_reason: FinishReason | None = None

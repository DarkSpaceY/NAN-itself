"""
LLM protocol vocabulary: the shared DTOs every layer speaks --
messages, tool calls, usage, the streaming event contract. Pure
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


class ToolDefinition(BaseModel):
    """
    Model-facing tool definition.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    """
    One complete model-generated Tool call.
    """

    id: str
    name: str
    arguments: dict[str, Any]


class Message(BaseModel):
    """
    Provider-neutral conversation message.

    assistant:
        content may be text
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

    tool_calls: list[ToolCall] = Field(
        default_factory=list,
    )

    tool_call_id: str | None = None


class LLMRequest(BaseModel):
    messages: list[Message]

    temperature: float = 0.7

    tools: list[ToolDefinition] = Field(
        default_factory=list,
    )


class LLMResponse(BaseModel):
    """
    Normalized complete model response.
    """

    content: str | None = None

    tool_calls: list[ToolCall] = Field(
        default_factory=list,
    )

    model: str
    usage: Usage

    finish_reason: FinishReason = "unknown"


class LLMStreamEvent(BaseModel):
    """
    Provider-neutral streaming event.

    kind:
        text
        tool_call
        done
    """

    kind: Literal[
        "text",
        "tool_call",
        "done",
    ]

    text: str | None = None

    tool_call: ToolCall | None = None

    usage: Usage | None = None

    finish_reason: FinishReason | None = None

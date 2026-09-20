from __future__ import annotations

import json
from typing import Any, AsyncIterator, Literal

from anthropic import AsyncAnthropic
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


ProviderName = Literal["openai", "anthropic"]

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

    This is intentionally provider-neutral.
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

    # OpenAI-compatible JSON mode ({"type": "json_object"}).
    # Used by structured machine-facing calls such as the memory
    # module's REVIEW/MERGE operations; None for normal turns.
    response_format: dict | None = None


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

    provider: ProviderName

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


class LLMProvider:
    """
    Provider-neutral streaming LLM adapter.

    Responsibilities:
        - talk to OpenAI / Anthropic
        - translate provider-specific messages
        - translate provider-specific tool definitions
        - normalize tool calls
        - normalize usage
        - expose one streaming protocol to Agent Runtime

    Agent Runtime should not know which provider is being used.
    """

    def __init__(
        self,
        provider: ProviderName,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float,
        max_retries: int,
    ) -> None:
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        logger.info(
            "Initializing LLMProvider: "
            f"provider={provider}, "
            f"model={model}, "
            f"base_url={base_url}, "
            f"timeout={timeout}, "
            f"max_retries={max_retries}",
        )

        if provider == "openai":
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            self.client = AsyncAnthropic(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
            )

        self._closed = False

    # ==================================================================
    # Public API
    # ==================================================================

    async def generate(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[LLMStreamEvent]:
        """
        Stream normalized model events.
        """
        if self._closed:
            raise RuntimeError(
                "LLMProvider is already closed"
            )

        if self.provider == "openai":
            async for event in self._stream_openai(request):
                yield event
        else:
            async for event in self._stream_anthropic(request):
                yield event

    async def generate_complete(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Collect the normalized stream into one response.
        """
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        usage = Usage()
        finish_reason: FinishReason = "unknown"

        async for event in self.generate(request):
            if event.kind == "text":
                if event.text:
                    text_parts.append(event.text)

            elif event.kind == "tool_call":
                if event.tool_call is not None:
                    tool_calls.append(
                        event.tool_call
                    )

            elif event.kind == "done":
                if event.usage is not None:
                    usage = event.usage

                if event.finish_reason is not None:
                    finish_reason = (
                        event.finish_reason
                    )

        content = "".join(text_parts)

        return LLMResponse(
            content=content or None,
            tool_calls=tool_calls,
            model=self.model,
            usage=usage,
            provider=self.provider,
            finish_reason=finish_reason,
        )

    async def close(self) -> None:
        if self._closed:
            return

        logger.info(
            "Closing LLMProvider: "
            f"provider={self.provider}, "
            f"model={self.model}",
        )

        await self.client.close()

        self._closed = True

    # ==================================================================
    # OpenAI
    # ==================================================================

    async def _stream_openai(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[LLMStreamEvent]:
        messages = [
            mapped
            for mapped in (
                self._openai_message(message)
                for message in request.messages
            )
            if mapped is not None
        ]

        logger.warning(
            "\n========== OPENAI REQUEST {} ==========\n{}",
            len(messages),
            json.dumps(
                messages,
                ensure_ascii=False,
                indent=2,
            ),
        )

        tools = (
            [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in request.tools
            ]
            if request.tools
            else None
        )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }

        if tools:
            kwargs["tools"] = tools

        if request.response_format is not None:
            kwargs["response_format"] = (
                request.response_format
            )

        stream = await self.client.chat.completions.create(
            **kwargs
        )

        tool_buffers: dict[
            int,
            dict[str, Any],
        ] = {}

        usage = Usage()
        finish_reason: FinishReason = "unknown"

        async for chunk in stream:
            if chunk.usage is not None:
                usage = Usage(
                    prompt_tokens=(
                        chunk.usage.prompt_tokens
                    ),
                    completion_tokens=(
                        chunk.usage.completion_tokens
                    ),
                )

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                yield LLMStreamEvent(
                    kind="text",
                    text=delta.content,
                )

            if delta.tool_calls:
                for tool_delta in delta.tool_calls:
                    index = (
                        tool_delta.index
                    )

                    buffer = tool_buffers.setdefault(
                        index,
                        {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        },
                    )

                    if tool_delta.id:
                        buffer["id"] = (
                            tool_delta.id
                        )

                    if (
                        tool_delta.function
                        and tool_delta.function.name
                    ):
                        buffer["name"] = (
                            tool_delta.function.name
                        )

                    if (
                        tool_delta.function
                        and tool_delta.function.arguments
                    ):
                        buffer["arguments"] += (
                            tool_delta.function.arguments
                        )

            if choice.finish_reason:
                finish_reason = (
                    self._normalize_openai_finish_reason(
                        choice.finish_reason
                    )
                )

        for index in sorted(tool_buffers):
            buffer = tool_buffers[index]

            yield LLMStreamEvent(
                kind="tool_call",
                tool_call=self._build_tool_call(
                    tool_id=buffer["id"],
                    name=buffer["name"],
                    raw_arguments=buffer["arguments"],
                ),
            )

        yield LLMStreamEvent(
            kind="done",
            usage=usage,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _openai_message(
        message: Message,
    ) -> dict[str, Any]:
        if message.role == "system":
            return {
                "role": "system",
                "content": message.content or "",
            }

        if message.role == "user":
            text = message.content or ""

            if not text:
                # Empty inputs carry no information; local chat
                # templates handle them poorly.
                return None

            return {
                "role": "user",
                "content": text,
            }

        if message.role == "assistant":
            if not message.content and not message.tool_calls:
                return None

            result: dict[str, Any] = {
                "role": "assistant",
            }

            if message.content is not None:
                result["content"] = message.content
            else:
                result["content"] = None

            if message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(
                                call.arguments,
                                ensure_ascii=False,
                            ),
                        },
                    }
                    for call in message.tool_calls
                ]

            return result

        if message.role == "tool":
            if message.tool_call_id is None:
                raise ValueError(
                    "tool message requires "
                    "tool_call_id"
                )

            return {
                "role": "tool",
                "tool_call_id": (
                    message.tool_call_id
                ),
                "content": (
                    message.content or ""
                ),
            }

        raise ValueError(
            f"Unsupported message role: "
            f"{message.role}"
        )

    # ==================================================================
    # Anthropic
    # ==================================================================

    async def _stream_anthropic(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[LLMStreamEvent]:
        system_prompt, messages = (
            self._anthropic_messages(
                request.messages
            )
        )

        tools = (
            [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in request.tools
            ]
            if request.tools
            else None
        )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "system": system_prompt,
            "temperature": request.temperature,
        }

        if tools:
            kwargs["tools"] = tools

        async with self.client.messages.stream(
            **kwargs
        ) as stream:
            current_tool_id: str | None = None
            current_tool_name: str | None = None
            current_tool_json = ""

            async for event in stream:
                event_type = getattr(
                    event,
                    "type",
                    None,
                )

                if event_type == "content_block_start":
                    block = getattr(
                        event,
                        "content_block",
                        None,
                    )

                    block_type = getattr(
                        block,
                        "type",
                        None,
                    )

                    if block_type == "tool_use":
                        current_tool_id = getattr(
                            block,
                            "id",
                            None,
                        )

                        current_tool_name = getattr(
                            block,
                            "name",
                            None,
                        )

                        current_tool_json = ""

                elif event_type == "content_block_delta":
                    delta = getattr(
                        event,
                        "delta",
                        None,
                    )

                    delta_type = getattr(
                        delta,
                        "type",
                        None,
                    )

                    if delta_type == "text_delta":
                        text = getattr(
                            delta,
                            "text",
                            None,
                        )

                        if text:
                            yield LLMStreamEvent(
                                kind="text",
                                text=text,
                            )

                    elif (
                        delta_type
                        == "input_json_delta"
                    ):
                        partial_json = getattr(
                            delta,
                            "partial_json",
                            "",
                        )

                        current_tool_json += (
                            partial_json
                        )

                elif event_type == "content_block_stop":
                    if (
                        current_tool_id
                        and current_tool_name
                    ):
                        yield LLMStreamEvent(
                            kind="tool_call",
                            tool_call=(
                                self._build_tool_call(
                                    tool_id=(
                                        current_tool_id
                                    ),
                                    name=(
                                        current_tool_name
                                    ),
                                    raw_arguments=(
                                        current_tool_json
                                    ),
                                )
                            ),
                        )

                        current_tool_id = None
                        current_tool_name = None
                        current_tool_json = ""

                elif event_type == "message_delta":
                    delta = getattr(
                        event,
                        "delta",
                        None,
                    )

                    stop_reason = getattr(
                        delta,
                        "stop_reason",
                        None,
                    )

                    if stop_reason:
                        finish_reason = (
                            self._normalize_anthropic_finish_reason(
                                stop_reason
                            )
                        )
                    else:
                        finish_reason = "unknown"

                    usage_data = getattr(
                        event,
                        "usage",
                        None,
                    )

                    if usage_data is not None:
                        usage = Usage(
                            prompt_tokens=(
                                0
                            ),
                            completion_tokens=(
                                getattr(
                                    usage_data,
                                    "output_tokens",
                                    0,
                                )
                            ),
                        )
                    else:
                        usage = Usage()

                    # Anthropic's final input-token usage is most
                    # reliably obtained from the final message.
                    final_message = (
                        await stream.get_final_message()
                    )

                    if final_message.usage:
                        usage = Usage(
                            prompt_tokens=(
                                getattr(
                                    final_message.usage,
                                    "input_tokens",
                                    0,
                                )
                            ),
                            completion_tokens=(
                                getattr(
                                    final_message.usage,
                                    "output_tokens",
                                    0,
                                )
                            ),
                        )

                    yield LLMStreamEvent(
                        kind="done",
                        usage=usage,
                        finish_reason=finish_reason,
                    )

                    return

            # Defensive fallback.
            final_message = (
                await stream.get_final_message()
            )

            usage = Usage(
                prompt_tokens=(
                    getattr(
                        final_message.usage,
                        "input_tokens",
                        0,
                    )
                    if final_message.usage
                    else 0
                ),
                completion_tokens=(
                    getattr(
                        final_message.usage,
                        "output_tokens",
                        0,
                    )
                    if final_message.usage
                    else 0
                ),
            )

            finish_reason = (
                self._normalize_anthropic_finish_reason(
                    getattr(
                        final_message,
                        "stop_reason",
                        None,
                    )
                    or "unknown"
                )
            )

            yield LLMStreamEvent(
                kind="done",
                usage=usage,
                finish_reason=finish_reason,
            )

    @staticmethod
    def _anthropic_messages(
        messages: list[Message],
    ) -> tuple[
        str | None,
        list[dict[str, Any]],
    ]:
        system_messages = [
            message
            for message in messages
            if message.role == "system"
        ]

        if len(system_messages) > 1:
            raise ValueError(
                "Anthropic adapter supports at most "
                "one system message."
            )

        system_prompt = (
            system_messages[0].content
            if system_messages
            else None
        )

        result: list[dict[str, Any]] = []

        for message in messages:
            if message.role == "system":
                continue

            if message.role == "user":
                text = message.content or ""

                if not text:
                    # An empty input (e.g. a late-report-only
                    # cycle) contributes nothing visible.
                    continue

                text_block = {
                    "type": "text",
                    "text": text,
                }

                # Anthropic requires strictly alternating roles.
                # Injected subagent reports can produce consecutive
                # user messages, so coalesce them into one message
                # holding multiple text blocks.
                previous = (
                    result[-1]
                    if result
                    else None
                )

                if (
                    previous is not None
                    and previous.get("role")
                    == "user"
                ):
                    previous["content"].append(
                        text_block
                    )

                    continue

                result.append({
                    "role": "user",
                    "content": [text_block],
                })

                continue

            if message.role == "assistant":
                blocks: list[dict[str, Any]] = []

                if not message.content and not message.tool_calls:
                    # Empty assistant placeholder (corrective-retry
                    # marker): carries no information.
                    continue

                if message.content:
                    blocks.append({
                        "type": "text",
                        "text": message.content,
                    })

                for tool_call in message.tool_calls:
                    blocks.append({
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": tool_call.arguments,
                    })

                result.append({
                    "role": "assistant",
                    "content": blocks,
                })

                continue

            if message.role == "tool":
                if message.tool_call_id is None:
                    raise ValueError(
                        "tool message requires "
                        "tool_call_id"
                    )

                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": (
                                message.tool_call_id
                            ),
                            "content": (
                                message.content or ""
                            ),
                        }
                    ],
                })

                continue

            raise ValueError(
                f"Unsupported message role: "
                f"{message.role}"
            )

        return system_prompt, result

    # ==================================================================
    # Normalization
    # ==================================================================

    @staticmethod
    def _build_tool_call(
        *,
        tool_id: str,
        name: str,
        raw_arguments: str,
    ) -> ToolCall:
        if not tool_id:
            raise ValueError(
                "Model returned a Tool call without id"
            )

        if not name:
            raise ValueError(
                "Model returned a Tool call without name"
            )

        if not raw_arguments:
            arguments: dict[str, Any] = {}

        else:
            try:
                parsed = json.loads(
                    raw_arguments
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Model returned invalid Tool "
                    f"arguments for '{name}': "
                    f"{raw_arguments!r}"
                ) from exc

            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Tool arguments for '{name}' "
                    "must be a JSON object."
                )

            arguments = parsed

        return ToolCall(
            id=tool_id,
            name=name,
            arguments=arguments,
        )

    @staticmethod
    def _normalize_openai_finish_reason(
        reason: str,
    ) -> FinishReason:
        if reason == "stop":
            return "stop"

        if reason == "tool_calls":
            return "tool_calls"

        if reason == "length":
            return "length"

        return "unknown"

    @staticmethod
    def _normalize_anthropic_finish_reason(
        reason: str,
    ) -> FinishReason:
        if reason == "end_turn":
            return "stop"

        if reason in {
            "tool_use",
        }:
            return "tool_calls"

        if reason == "max_tokens":
            return "length"

        return "unknown"
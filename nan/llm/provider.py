"""
Provider-neutral streaming LLM adapter (the model boundary).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic
from anthropic import DefaultAsyncHttpxClient as AsyncHttpxClientAnthropic
from anthropic.types import (
    InputJSONDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RedactedThinkingBlock,
    SignatureDelta,
    TextDelta,
    ThinkingDelta,
    ToolUseBlock,
)
from anthropic.types import (
    Message as AnthropicMessage,
)
from anthropic.types import (
    ThinkingBlock as AnthropicThinkingBlock,
)
from anthropic.types.raw_message_delta_event import Delta as MessageDeltaDelta
from loguru import logger
from openai import AsyncOpenAI
from openai import DefaultAsyncHttpxClient as AsyncHttpxClientOpenAI

from nan.config import settings

from .types import (
    FinishReason,
    LLMRequest,
    LLMStreamEvent,
    Message,
    ThinkingBlock,
    ToolCall,
    Usage,
)


def _proxy_kwargs(proxy: str) -> dict[str, Any]:
    """
    Proxy discipline: the endpoint never inherits an ambient proxy
    unless the configuration asks for it.
    """
    if proxy == "env":
        return {"trust_env": True}

    return {"trust_env": False, "proxy": proxy or None}


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
    """

    def __init__(self) -> None:
        """Build the provider described by the process-wide settings."""
        cfg = settings.llm

        provider = cfg.provider

        # empty config values = SDK defaults (env fallback / default endpoint)
        api_key = cfg.api_key or None
        model = cfg.model
        base_url = cfg.base_url or None
        timeout = cfg.timeout
        max_retries = cfg.max_retries
        proxy = cfg.proxy

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
            f"max_retries={max_retries}, "
            f"proxy={proxy or 'direct'}",
        )

        # Typed clients: one per protocol; each use site
        # dispatches on the one that is set.
        self._openai: AsyncOpenAI | None = None
        self._anthropic: AsyncAnthropic | None = None

        if provider == "openai":
            self._openai = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                http_client=AsyncHttpxClientOpenAI(**_proxy_kwargs(proxy)),
            )
        else:
            self._anthropic = AsyncAnthropic(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                http_client=AsyncHttpxClientAnthropic(**_proxy_kwargs(proxy)),
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
            raise RuntimeError("LLMProvider is already closed")

        if self._openai is not None:
            async for event in self._stream_openai(request):
                yield event
        else:
            async for event in self._stream_anthropic(request):
                yield event

    async def close(self) -> None:
        if self._closed:
            return

        logger.info(
            f"Closing LLMProvider: provider={self.provider}, model={self.model}",
        )

        if self._openai is not None:
            await self._openai.close()

        if self._anthropic is not None:
            await self._anthropic.close()

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
            for mapped in (self._openai_message(message) for message in request.messages)
            if mapped is not None
        ]

        logger.debug(
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

        if self._openai is None:
            raise RuntimeError("OpenAI client not initialized")

        stream = await self._openai.chat.completions.create(**kwargs)

        tool_buffers: dict[
            int,
            dict[str, Any],
        ] = {}

        usage = Usage()
        finish_reason: FinishReason = "unknown"

        # Reasoning arrives incrementally either as plain text
        # (reasoning_content / reasoning) or as structured
        # reasoning_details entries. Structured details win because
        # they carry the signature/encryption needed for replay.
        thinking_text: list[str] = []
        thinking_details: dict[int, ThinkingBlock] = {}
        has_details = False

        async for chunk in stream:
            if chunk.usage is not None:
                usage = Usage(
                    prompt_tokens=(chunk.usage.prompt_tokens),
                    completion_tokens=(chunk.usage.completion_tokens),
                )

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if settings.llm.thinking:
                details = getattr(delta, "reasoning_details", None)

                if isinstance(details, list):
                    for position, detail in enumerate(details):
                        has_details = True

                        text = self._merge_reasoning_detail(thinking_details, position, detail)

                        if text:
                            yield LLMStreamEvent(kind="thinking", thinking_text=text)

                plain = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )

                if isinstance(plain, str) and plain and not has_details:
                    thinking_text.append(plain)

                    yield LLMStreamEvent(kind="thinking", thinking_text=plain)

            if delta.content:
                yield LLMStreamEvent(
                    kind="text",
                    text=delta.content,
                )

            if delta.tool_calls:
                for tool_delta in delta.tool_calls:
                    index = tool_delta.index

                    buffer = tool_buffers.setdefault(
                        index,
                        {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        },
                    )

                    if tool_delta.id:
                        buffer["id"] = tool_delta.id

                    if tool_delta.function and tool_delta.function.name:
                        buffer["name"] = tool_delta.function.name

                    if tool_delta.function and tool_delta.function.arguments:
                        buffer["arguments"] += tool_delta.function.arguments

            if choice.finish_reason:
                finish_reason = self._normalize_openai_finish_reason(choice.finish_reason)

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

        if thinking_details:
            thinking = [thinking_details[index] for index in sorted(thinking_details)]
        elif thinking_text:
            thinking = [ThinkingBlock(text="".join(thinking_text))]
        else:
            thinking = []

        yield LLMStreamEvent(
            kind="done",
            usage=usage,
            finish_reason=finish_reason,
            thinking=thinking or None,
        )

    @staticmethod
    def _openai_message(
        message: Message,
    ) -> dict[str, Any] | None:
        """
        Map one message; None means it carries nothing and is dropped.
        """
        if message.role == "system":
            return {
                "role": "system",
                "content": message.content or "",
            }

        if message.role == "user":
            text = message.content or ""

            if not text:
                return None

            return {
                "role": "user",
                "content": text,
            }

        if message.role == "assistant":
            if not message.content and not message.tool_calls and not message.thinking:
                return None

            result: dict[str, Any] = {
                "role": "assistant",
                "content": message.content,
            }

            if message.thinking:
                # Strict cache: replay reasoning byte-for-byte. Structured
                # details (signature/encryption) must go back as
                # reasoning_details; plain reasoning uses the legacy field.
                if any(
                    block.type == "encrypted" or block.signature or block.format
                    for block in message.thinking
                ):
                    result["reasoning_details"] = [
                        LLMProvider._openai_reasoning_detail(block) for block in message.thinking
                    ]
                else:
                    result["reasoning_content"] = "".join(block.text for block in message.thinking)

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
                raise ValueError("tool message requires tool_call_id")

            return {
                "role": "tool",
                "tool_call_id": (message.tool_call_id),
                "content": (message.content or ""),
            }

        raise ValueError(f"Unsupported message role: {message.role}")

    @staticmethod
    def _openai_reasoning_detail(block: ThinkingBlock) -> dict[str, Any]:
        """
        Rebuild one reasoning_details entry from a stored block.
        """
        if block.type == "encrypted":
            detail: dict[str, Any] = {
                "type": "reasoning.encrypted",
                "data": block.data or "",
            }
        else:
            detail = {
                "type": "reasoning.text",
                "text": block.text,
            }

            if block.signature:
                detail["signature"] = block.signature

        if block.format:
            detail["format"] = block.format

        return detail

    @staticmethod
    def _merge_reasoning_detail(
        blocks: dict[int, ThinkingBlock],
        position: int,
        raw: Any,
    ) -> str | None:
        """
        Accumulate one streaming reasoning_details entry; returns the new
        visible text, if any.
        """
        if not isinstance(raw, dict):
            return None

        index = raw.get("index")
        key = index if isinstance(index, int) else position

        block = blocks.setdefault(key, ThinkingBlock())

        text: str | None = None

        if raw.get("type") == "reasoning.encrypted":
            block.type = "encrypted"

            data = raw.get("data")

            if isinstance(data, str) and data:
                block.data = data
        else:
            block.type = "text"

            chunk = raw.get("text")

            if isinstance(chunk, str) and chunk:
                block.text += chunk
                text = chunk

        signature = raw.get("signature")

        if isinstance(signature, str) and signature:
            block.signature = signature

        detail_format = raw.get("format")

        if isinstance(detail_format, str) and detail_format:
            block.format = detail_format

        return text

    # ==================================================================
    # Anthropic
    # ==================================================================

    async def _stream_anthropic(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[LLMStreamEvent]:
        system_prompt, messages = self._anthropic_messages(request.messages)

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
            "max_tokens": settings.llm.max_tokens,
        }

        if settings.llm.thinking and settings.llm.thinking_budget > 0:
            # Native extended thinking requires temperature=1 and a
            # budget smaller than max_tokens.
            budget = min(settings.llm.thinking_budget, settings.llm.max_tokens - 1)

            if budget > 0:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget,
                }
                kwargs["temperature"] = 1.0

        if tools:
            kwargs["tools"] = tools

        if self._anthropic is None:
            raise RuntimeError("Anthropic client not initialized")

        async with self._anthropic.messages.stream(**kwargs) as stream:
            current_tool_id: str | None = None
            current_tool_name: str | None = None
            current_tool_json = ""
            current_thinking: ThinkingBlock | None = None
            thinking_blocks: list[ThinkingBlock] = []

            async for event in stream:
                match event:
                    case RawContentBlockStartEvent(
                        content_block=ToolUseBlock(id=tool_id, name=tool_name)
                    ):
                        current_tool_id = tool_id
                        current_tool_name = tool_name
                        current_tool_json = ""

                    case RawContentBlockStartEvent(content_block=RedactedThinkingBlock(data=data)):
                        if settings.llm.thinking:
                            thinking_blocks.append(
                                ThinkingBlock(
                                    type="encrypted",
                                    data=data,
                                    format="anthropic",
                                )
                            )

                    case RawContentBlockStartEvent(
                        content_block=AnthropicThinkingBlock(
                            thinking=thinking,
                            signature=signature,
                        )
                    ):
                        if settings.llm.thinking:
                            current_thinking = ThinkingBlock(
                                text=thinking or "",
                                signature=signature or None,
                                format="anthropic",
                            )

                    case RawContentBlockDeltaEvent(delta=ThinkingDelta(thinking=thinking)):
                        if settings.llm.thinking and thinking:
                            if current_thinking is None:
                                current_thinking = ThinkingBlock(format="anthropic")

                            current_thinking.text += thinking

                            yield LLMStreamEvent(kind="thinking", thinking_text=thinking)

                    case RawContentBlockDeltaEvent(delta=SignatureDelta(signature=signature)):
                        if settings.llm.thinking and signature:
                            if current_thinking is None:
                                current_thinking = ThinkingBlock(format="anthropic")

                            current_thinking.signature = signature

                    case RawContentBlockDeltaEvent(delta=TextDelta(text=text)):
                        if text:
                            yield LLMStreamEvent(kind="text", text=text)

                    case RawContentBlockDeltaEvent(delta=InputJSONDelta(partial_json=partial_json)):
                        current_tool_json += partial_json

                    case RawContentBlockStopEvent():
                        if current_tool_id and current_tool_name:
                            yield LLMStreamEvent(
                                kind="tool_call",
                                tool_call=self._build_tool_call(
                                    tool_id=current_tool_id,
                                    name=current_tool_name,
                                    raw_arguments=current_tool_json,
                                ),
                            )

                            current_tool_id = None
                            current_tool_name = None
                            current_tool_json = ""

                        if current_thinking is not None:
                            thinking_blocks.append(current_thinking)
                            current_thinking = None

                    case RawMessageDeltaEvent(delta=MessageDeltaDelta(stop_reason=stop_reason)):
                        finish_reason = (
                            self._normalize_anthropic_finish_reason(stop_reason)
                            if stop_reason
                            else "unknown"
                        )

                        # Anthropic's final usage (input + output
                        # tokens) is most reliably obtained from
                        # the final message; message_delta carries
                        # only a partial view of it.
                        final_message = await stream.get_final_message()

                        yield LLMStreamEvent(
                            kind="done",
                            usage=self._usage_from_final_message(final_message),
                            finish_reason=finish_reason,
                            thinking=thinking_blocks or None,
                        )

                        return

            # Defensive fallback: stream ended without a
            # message_delta (e.g. an aborted connection).
            final_message = await stream.get_final_message()

            if current_thinking is not None:
                thinking_blocks.append(current_thinking)

            yield LLMStreamEvent(
                kind="done",
                usage=self._usage_from_final_message(final_message),
                finish_reason=self._normalize_anthropic_finish_reason(
                    final_message.stop_reason or "unknown"
                ),
                thinking=thinking_blocks or None,
            )

    @staticmethod
    def _anthropic_messages(
        messages: list[Message],
    ) -> tuple[
        str | None,
        list[dict[str, Any]],
    ]:
        system_messages = [message for message in messages if message.role == "system"]

        if len(system_messages) > 1:
            raise ValueError("Anthropic adapter supports at most one system message.")

        system_prompt = system_messages[0].content if system_messages else None

        result: list[dict[str, Any]] = []

        for message in messages:
            if message.role == "system":
                continue

            if message.role == "user":
                text = message.content or ""

                if not text:
                    continue

                result.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text,
                            }
                        ],
                    }
                )

                continue

            if message.role == "assistant":
                blocks: list[dict[str, Any]] = []

                for block in message.thinking:
                    if block.type == "encrypted" and block.data:
                        blocks.append(
                            {
                                "type": "redacted_thinking",
                                "data": block.data,
                            }
                        )
                    elif block.type == "text" and block.signature:
                        blocks.append(
                            {
                                "type": "thinking",
                                "thinking": block.text,
                                "signature": block.signature,
                            }
                        )

                if not message.content and not message.tool_calls and not blocks:
                    # Empty assistant placeholder (corrective-retry marker).
                    continue

                if message.content:
                    blocks.append(
                        {
                            "type": "text",
                            "text": message.content,
                        }
                    )

                for tool_call in message.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "input": tool_call.arguments,
                        }
                    )

                result.append(
                    {
                        "role": "assistant",
                        "content": blocks,
                    }
                )

                continue

            if message.role == "tool":
                if message.tool_call_id is None:
                    raise ValueError("tool message requires tool_call_id")

                result.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": (message.tool_call_id),
                                "content": (message.content or ""),
                            }
                        ],
                    }
                )

                continue

            raise ValueError(f"Unsupported message role: {message.role}")

        # Anthropic requires strictly alternating roles and
        # rejects consecutive same-role messages with 400.
        # Tool results travel in user messages, so a batch of
        # tool calls yields consecutive user messages; merge
        # any adjacent user messages into one (content blocks
        # concatenated, order preserved).
        merged: list[dict[str, Any]] = []

        for entry in result:
            previous = merged[-1] if merged else None

            if (
                previous is not None
                and previous.get("role") == "user"
                and entry.get("role") == "user"
            ):
                previous["content"].extend(entry["content"])

                continue

            merged.append(entry)

        return system_prompt, merged

    # ==================================================================
    # Normalization
    # ==================================================================

    @staticmethod
    def _usage_from_final_message(final_message: AnthropicMessage) -> Usage:
        """Anthropic usage lives on the final message (input +
        output tokens); pydantic guarantees the field is present."""

        return Usage(
            prompt_tokens=final_message.usage.input_tokens,
            completion_tokens=final_message.usage.output_tokens,
        )

    @staticmethod
    def _build_tool_call(
        *,
        tool_id: str,
        name: str,
        raw_arguments: str,
    ) -> ToolCall:
        if not tool_id:
            raise ValueError("Model returned a Tool call without id")

        if not name:
            raise ValueError("Model returned a Tool call without name")

        if not raw_arguments:
            arguments: dict[str, Any] = {}

        else:
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Model returned invalid Tool arguments for '{name}': {raw_arguments!r}"
                ) from exc

            if not isinstance(parsed, dict):
                raise ValueError(f"Tool arguments for '{name}' must be a JSON object.")

            arguments = parsed

        return ToolCall(
            id=tool_id,
            name=name,
            arguments=arguments,
        )

    _OPENAI_FINISH_REASONS: dict[str, FinishReason] = {
        "stop": "stop",
        "tool_calls": "tool_calls",
        "length": "length",
    }

    _ANTHROPIC_FINISH_REASONS: dict[str, FinishReason] = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
    }

    @staticmethod
    def _normalize_openai_finish_reason(
        reason: str,
    ) -> FinishReason:
        return LLMProvider._OPENAI_FINISH_REASONS.get(reason, "unknown")

    @staticmethod
    def _normalize_anthropic_finish_reason(
        reason: str,
    ) -> FinishReason:
        return LLMProvider._ANTHROPIC_FINISH_REASONS.get(reason, "unknown")

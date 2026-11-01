"""
The turn engine: the phases of one turn.

The Agent owns the turn (it opens the Turn when it observes and
keeps it resident in its Turn list); the engine runs each phase --
collect service contributions, one model call, run verbs
positionally -- and hands the updated Turn back. Freezing is the
last phase.

The engine owns no agent state: it reads only what rides the
parameters and writes nothing back. The Agent persists the turn
between phases.

Streaming always runs and is always emitted to the `sink`
singleton; without subscribers the events simply land in the
history ring. Agent identity rides the emit calls as plain keys.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from loguru import logger

from nan.events import sink
from nan.llm import (
    FinishReason,
    LLMRequest,
    LLMResponse,
    Message,
    ThinkingBlock,
    ToolCall,
    ToolDefinition,
    Usage,
)
from nan.turn import DeltaEvent, Injection, LLMCall, ToolResult, Turn, render_turn

if TYPE_CHECKING:
    from nan.llm import LLMProvider

# Hub-side injection: payload in, (contributions, service deltas) out.
InjectFn = Callable[..., Awaitable[tuple[tuple[Injection, ...], list[DeltaEvent]]]]

# Verb execution: one call in, its string result out.
ExecuteFn = Callable[[ToolCall], Awaitable[str]]

# Progress hook: the turn as it stands, made durable by the Agent.
RecordFn = Callable[[Turn], Awaitable[None]]


class TurnEngine:
    """
    One turn, phase by phase: the Agent calls inject, then think
    (possibly more than once while retrying), then act, then
    freeze.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def inject(
        self,
        turn: Turn,
        *,
        inject: InjectFn,
    ) -> Turn:
        contributions, service_deltas = await inject(turn)

        sink.emit(
            "injections_collected",
            content={
                "agent_id": turn.agent_id,
                "services": [
                    {
                        "service_id": i.service_id,
                        "ok": i.ok,
                        "duration_s": i.ended_at - i.started_at,
                        "text_len": len(i.text),
                    }
                    for i in contributions
                ],
            },
        )

        return replace(
            turn,
            injections=contributions,
            delta=tuple([*turn.delta, *service_deltas]),
            phase="injected",
        )

    async def think(
        self,
        turn: Turn,
        *,
        persona: str,
        history: Sequence[Message],
        tools: Sequence[ToolDefinition],
        retries: int,
    ) -> Turn:
        messages = [
            Message(role="system", content=persona),
            *history,
            *render_turn(turn),
        ]

        started_at = time.time()

        response = await self.generate(
            agent_id=turn.agent_id,
            depth=turn.depth,
            request=LLMRequest(messages=messages, tools=list(tools)),
        )

        return _with_response(
            turn,
            response,
            started_at=started_at,
            ended_at=time.time(),
            retries=retries,
        )

    async def act(self, turn: Turn, *, execute: ExecuteFn, record: RecordFn) -> Turn:
        """
        Run the calls in order. Every result is recorded before the
        next call starts, so a stop mid-batch loses only the call in
        flight -- never a reported one, and no call runs twice.
        """
        for call in turn.calls:
            logger.info("[{}] verb {} {}", turn.agent_id, call.name, call.arguments)

            started_at = time.time()

            output = await execute(call)

            turn = replace(
                turn,
                results=(
                    *turn.results,
                    ToolResult(
                        call_id=call.id,
                        name=call.name,
                        output=output,
                        started_at=started_at,
                        ended_at=time.time(),
                    ),
                ),
            )

            await record(turn)

        return replace(turn, phase="acted")

    # ------------------------------------------------------------------
    # The model call
    # ------------------------------------------------------------------

    async def generate(self, *, agent_id: str, depth: int, request: LLMRequest) -> LLMResponse:
        # Emission without subscribers simply lands in the
        # history ring.
        text_parts: list[str] = []
        thinking_text_parts: list[str] = []
        thinking_blocks: list[ThinkingBlock] = []
        tool_calls: list[ToolCall] = []
        usage = Usage()
        finish_reason: FinishReason = "unknown"

        # Streams start lazily: reasoning normally arrives before the
        # answer, so opening output first would render them out of order.
        thinking_id: str | None = None
        stream_id: str | None = None

        async for event in self.llm.generate(request):
            if event.kind == "thinking" and event.thinking_text:
                thinking_text_parts.append(event.thinking_text)

                if thinking_id is None:
                    thinking_id = sink.emit(
                        "thinking_started",
                        content={"agent_id": agent_id, "depth": depth},
                    )

                sink.emit(
                    "thinking_delta",
                    id=thinking_id,
                    content={"text": event.thinking_text, "agent_id": agent_id, "depth": depth},
                )

            elif event.kind == "text" and event.text:
                text_parts.append(event.text)

                if stream_id is None:
                    stream_id = sink.emit(
                        "output_started",
                        content={"agent_id": agent_id, "depth": depth},
                    )

                sink.emit(
                    "output_delta",
                    id=stream_id,
                    content={"text": event.text, "agent_id": agent_id, "depth": depth},
                )

            elif event.kind == "tool_call" and event.tool_call is not None:
                tool_calls.append(event.tool_call)

            elif event.kind == "done":
                if event.usage is not None:
                    usage = event.usage

                if event.finish_reason is not None:
                    finish_reason = event.finish_reason

                if event.thinking:
                    thinking_blocks = list(event.thinking)

        if thinking_id:
            sink.emit(
                "thinking_done",
                id=thinking_id,
                content={"agent_id": agent_id, "depth": depth},
            )

        if not thinking_blocks and thinking_text_parts:
            thinking_blocks = [ThinkingBlock(text="".join(thinking_text_parts))]

        # A tool-only answer streams nothing: no start/close pair to
        # keep. The UI renders the verbs; the mapper tolerates
        # orphans on replay either way.
        if stream_id is not None:
            sink.emit(
                "output_done" if not tool_calls else "output_cancelled",
                id=stream_id,
                content={"agent_id": agent_id, "depth": depth},
            )

        return LLMResponse(
            content=("".join(text_parts) or None),
            thinking=thinking_blocks,
            tool_calls=tool_calls,
            model=self.llm.model,
            usage=usage,
            finish_reason=finish_reason,
        )


def _with_response(
    turn: Turn,
    response: LLMResponse,
    *,
    started_at: float,
    ended_at: float,
    retries: int,
) -> Turn:
    # The turn ends at commit, not when the model answers.
    return replace(
        turn,
        phase="thought",
        reply=response.content or "",
        thinking=tuple(response.thinking),
        calls=tuple(response.tool_calls),
        llm=LLMCall(
            model=response.model,
            started_at=started_at,
            ended_at=ended_at,
            usage=response.usage,
            finish_reason=response.finish_reason,
            retries=retries,
        ),
    )

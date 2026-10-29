"""
The turn engine: one turn, in phases.

The Agent prepares the arrival shape (a fresh Turn skeleton:
identity, inbox, deltas, origin, persona marker) and hands it
over; the engine runs the pipeline -- collect service
contributions, one model call, run verbs positionally, freeze
the rendered messages -- and returns the complete Turn.

The engine owns no agent state: it reads only what rides the
parameters and writes nothing back. The Agent applies the
result (history, last_turn) after the call.

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
from nan.llm import FinishReason, LLMRequest, LLMResponse, Message, ToolCall, Usage
from nan.turn import DeltaEvent, Injection, LLMCall, ToolResult, Turn, freeze_turn, render_turn

from .verbs import tool_definitions

if TYPE_CHECKING:
    from nan.llm import LLMProvider

# Hub-side injection: payload in, (contributions, service deltas) out.
InjectFn = Callable[..., Awaitable[tuple[tuple[Injection, ...], list[DeltaEvent]]]]

# Verb execution: one call in, its string result out.
ExecuteFn = Callable[[ToolCall], Awaitable[str]]


def next_backoff(backoff: Sequence[float], index: int) -> tuple[float, int]:
    """
    The delay for retry `index`, and the index for the call
    after it. An empty ladder means: no delay, no advance.
    """
    if not backoff:
        return 0.0, index

    return backoff[min(index, len(backoff) - 1)], index + 1


class TurnEngine:
    """
    One turn per invocation: skeleton in, complete Turn out.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def run_turn(
        self,
        turn: Turn,
        *,
        persona: str,
        history: Sequence[Message],
        inject: InjectFn,
        execute: ExecuteFn,
        retries: int,
    ) -> Turn:
        turn = await self._inject(turn, inject=inject)

        turn = await self._think(turn, persona=persona, history=history, retries=retries)

        turn = await self._act(turn, execute=execute)

        return freeze_turn(turn)

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    async def _inject(
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
        )

    async def _think(
        self,
        turn: Turn,
        *,
        persona: str,
        history: Sequence[Message],
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
            request=LLMRequest(messages=messages, tools=tool_definitions(depth=turn.depth)),
        )

        return _with_response(
            turn,
            response,
            started_at=started_at,
            ended_at=time.time(),
            retries=retries,
        )

    async def _act(self, turn: Turn, *, execute: ExecuteFn) -> Turn:
        results: list[ToolResult] = []

        for call in turn.calls:
            logger.info("[{}] verb {} {}", turn.agent_id, call.name, call.arguments)

            started_at = time.time()

            output = await execute(call)

            results.append(
                ToolResult(
                    call_id=call.id,
                    name=call.name,
                    output=output,
                    started_at=started_at,
                    ended_at=time.time(),
                )
            )

        return _with_results(turn, results, ended_at=time.time())

    # ------------------------------------------------------------------
    # The model call
    # ------------------------------------------------------------------

    async def generate(self, *, agent_id: str, depth: int, request: LLMRequest) -> LLMResponse:
        # Emission without subscribers simply lands in the
        # history ring.
        text_parts: list[str] = []
        tool_calls: list = []
        usage = Usage()
        finish_reason: FinishReason = "unknown"

        stream_id = sink.emit("output_started", content={"agent_id": agent_id, "depth": depth})

        async for event in self.llm.generate(request):
            if event.kind == "text" and event.text:
                text_parts.append(event.text)

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

        if stream_id:
            sink.emit(
                "output_done" if not tool_calls else "output_cancelled",
                id=stream_id,
                content={"agent_id": agent_id, "depth": depth},
            )

        return LLMResponse(
            content=("".join(text_parts) or None),
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
    return replace(
        turn,
        reply=response.content or "",
        calls=tuple(response.tool_calls),
        llm=LLMCall(
            model=response.model,
            started_at=started_at,
            ended_at=ended_at,
            usage=response.usage,
            finish_reason=response.finish_reason,
            retries=retries,
        ),
        ended_at=ended_at,
    )


def _with_results(turn: Turn, results: list[ToolResult], *, ended_at: float) -> Turn:
    # The turn ends when its last verb ends.
    return replace(turn, results=tuple(results), ended_at=ended_at)

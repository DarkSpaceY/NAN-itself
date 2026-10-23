"""
The turn engine: the single model call of a turn, plus the
turn-level backoff ladder.

Streaming always runs and is always emitted to the `sink`
singleton; without subscribers the events simply land in the
history ring. Agent identity rides the emit calls as plain keys.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nan.events import sink
from nan.llm import FinishReason, LLMRequest, LLMResponse, Usage

if TYPE_CHECKING:
    from nan.llm import LLMProvider

# Turn-level backoff ladder: on failure the whole turn is
# retried after these delays; the index of the successful retry
# is recorded in LLMCall.retries.
DEFAULT_BACKOFF: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0)


def next_backoff(backoff: tuple[float, ...], index: int) -> tuple[float, int]:
    """
    The delay for retry `index`, and the index for the call
    after it. An empty ladder means: no delay, no advance.
    """
    if not backoff:
        return 0.0, index

    return backoff[min(index, len(backoff) - 1)], index + 1


class TurnEngine:
    """
    One model call per invocation. Kept apart from the Agent:
    the loop concerns itself with turn orchestration.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

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
            provider=self.llm.provider,
            finish_reason=finish_reason,
        )

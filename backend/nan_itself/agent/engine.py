"""
Step engine: how ANY agent (root or sub) thinks for one turn.

A step is exactly ONE model call:

    request    the model sees [system, *history, *messages]; the
               turn's observation message is already in `messages`
               (assembled by the Agent), followed by any tool
               results the Agent appended earlier in this turn
    result     plain text ends the turn; tool calls are appended
               to the turn as ONE assistant message and the turn
               ends there -- running the verbs and starting the
               next turn is the Agent's business

Every agent is identical; actions come exclusively from the verb
registry, executed by the Agent against itself. There is no
in-engine loop: after tool calls the next observation is rebuilt
(fresh modules, fresh inbox), which is what keeps the message
prefix cacheable.

There is no persistent history anywhere. Each Turn carries the
history snapshot as of its start (`turn.history`); the next turn
derives its snapshot as `last_turn.history + last_turn.messages`,
cleared to empty over the agent's history_char_limit. A turn's
full model input is exactly `turn.history + turn.messages` --
reconstruction needs nothing else.

The engine owns no agent state and assembles no observation:
identity, world snapshot, persona, the derived history snapshot
and the observation message all arrive on the Turn the caller
(Agent.run) built. The engine never runs a loop and never runs a
verb.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace

from loguru import logger

from .prompts import (
    build_system,
)
from .verbs import (
    FINISH_TOOL_NAME,
    VERBS,
)
from ..modules.model import (
    Turn,
)
from ..events import (
    sink,
)
from ..utils.llm import (
    LLMRequest,
    LLMResponse,
    Message,
    ToolDefinition,
    Usage,
)


class StepEngine:
    def __init__(
        self,
        *,
        llm,
    ) -> None:
        self.llm = llm

    async def step(
        self,
        turn: Turn,
    ) -> Turn:
        """
        Run one model call over the turn's messages.

        The caller (Agent.run) owns the turn boundary: it already
        refreshed skills, re-read the persona, took this agent's
        fresh world snapshot, derived the history snapshot,
        harvested its finished children's reports and assembled
        the observation message. Everything that describes this
        turn therefore rides on `turn`.

        The returned Turn is a new value: the incoming messages
        plus the assistant's reply as its last message -- one
        plain-text message, or one message carrying the tool
        calls. No verb is run here and nothing is delivered; a
        failure propagates to the caller, which records it on
        the Turn it delivers.

        The engine reads this agent's identity off `turn` and
        carries it explicitly into every event it emits.
        """
        messages = list(turn.messages)

        response = await self._generate(
            LLMRequest(
                messages=[
                    build_system(
                        turn.persona or ""
                    ),
                    *turn.history,
                    *messages,
                ],
                tools=self._tool_definitions(
                    depth=turn.depth
                ),
            ),
            turn,
        )

        # ------------------------------------------------------
        # Plain text ends the turn; tool calls ride out as ONE
        # assistant message whose results the caller runs and
        # appends.
        # ------------------------------------------------------

        if not response.tool_calls:
            if not (
                response.content or ""
            ).strip():
                logger.warning(
                    "[turn:{}] empty reply "
                    "(finish={})",
                    turn.agent_hash[:8],
                    response.finish_reason,
                )

            messages.append(
                _assistant_message(
                    response.content or ""
                )
            )

        else:
            messages.append(
                _assistant_message_with_calls(
                    response
                )
            )

        return replace(
            turn,
            messages=tuple(messages),
            usage=response.usage,
            finish_reason=response.finish_reason,
            model=self.llm.model,
            ended_at=time.time(),
        )

    # ------------------------------------------------------------------
    # Model call
    # ------------------------------------------------------------------

    async def _generate(
        self,
        request,
        turn,
    ) -> LLMResponse:
        """
        Stream the model on the global sink. Without an attached
        bus this is exactly generate_complete(); with one, text
        deltas are forwarded live and withdrawn if tool calls
        materialize.
        """
        if not sink.attached:
            return await self.llm.generate_complete(
                request
            )

        text_parts: list[str] = []
        tool_calls: list = []
        usage = Usage()
        finish_reason: str = "unknown"
        stream_id: str | None = None
        first_text: float | None = None

        stream_id = sink.emit(
            "output_started",
            content={
                "agent_hash": turn.agent_hash,
                "parent_hash": turn.parent_hash,
                "depth": turn.depth,
            },
        )

        async for event in self.llm.generate(
            request
        ):
            if event.kind == "text":
                if event.text:
                    if first_text is None:
                        first_text = time.time()

                    sink.emit(
                        "output_delta",
                        id=stream_id,
                        content={
                            "text": event.text,
                            "agent_hash": turn.agent_hash,
                            "parent_hash": turn.parent_hash,
                            "depth": turn.depth,
                        },
                    )

                    text_parts.append(
                        event.text
                    )

            elif event.kind == "tool_call":
                if event.tool_call is not None:
                    tool_calls.append(
                        event.tool_call
                    )

            elif event.kind == "done":
                if event.usage is not None:
                    usage = event.usage

                if (
                    event.finish_reason
                    is not None
                ):
                    finish_reason = (
                        event.finish_reason
                    )

        if stream_id is not None:
            if tool_calls:
                sink.emit(
                    "output_cancelled",
                    id=stream_id,
                    content={
                        "agent_hash": turn.agent_hash,
                        "parent_hash": turn.parent_hash,
                        "depth": turn.depth,
                    },
                )

            else:
                sink.emit(
                    "output_done",
                    id=stream_id,
                    content={
                        "duration": (
                            f"{time.time() - (first_text or time.time()):.1f}s"
                        ),
                        "agent_hash": turn.agent_hash,
                        "parent_hash": turn.parent_hash,
                        "depth": turn.depth,
                    },
                )

        return LLMResponse(
            content=(
                "".join(text_parts)
                or None
            ),
            tool_calls=tool_calls,
            model=self.llm.model,
            usage=usage,
            provider=self.llm.provider,
            finish_reason=finish_reason,
        )

    def _tool_definitions(
        self,
        depth: int = 0,
    ) -> list[ToolDefinition]:
        verbs = list(VERBS.values())

        # finish ends a subagent's loop; the root agent has no
        # such notion and must not see it.
        if depth == 0:
            verbs = [
                verb
                for verb in verbs
                if verb.name != FINISH_TOOL_NAME
            ]

        return [
            verb.definition()
            for verb in verbs
        ]


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def _assistant_message(
    content: str,
) -> Message:
    return Message(
        role="assistant",
        content=content,
    )


def _assistant_message_with_calls(
    response,
) -> Message:
    """
    Tool-call-only assistant messages are allowed to have no textual
    content (Message.content is `str | None`). Normalize a missing
    content to an empty string at the engine boundary so downstream
    snapshot handling never has to handle None.
    """
    return Message(
        role="assistant",
        content=response.content or "",
        tool_calls=response.tool_calls,
    )

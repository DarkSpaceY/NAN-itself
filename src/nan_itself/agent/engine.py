"""
Step engine: how ANY agent (main or sub) thinks for one turn.

A turn is exactly ONE cycle of three phases:

    observation   one user message carrying ambient Module
                  context (the inbox included); a subagent's
                  task is appended the same way
    model call    the model sees [system, *history, *turn]
    result        plain text ends the turn; tool calls are run,
                  their results appended, and the turn ends --
                  the caller starts the next turn

Every agent is identical; actions come exclusively from the verb
registry. There is no in-engine loop: after tool calls the next
observation is rebuilt (fresh modules, fresh inbox), which is
what keeps the message prefix cacheable.

`history` is owned by the caller (the main agent keeps it across
turns; spawned agents pass an empty list). The engine never
mutates it; this turn's messages are returned in AgentResult.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import replace
from typing import Callable

from loguru import logger

from .model import (
    AgentResult,
)
from .prompts import (
    build_observation,
    build_system,
)
from .reports import (
    format_child_report,
)
from .verbs import (
    INVOKE_SKILL_TOOL_NAME,
    INVOKE_TOOL_TOOL_NAME,
    SPAWN_TOOL_NAME,
    VERBS,
    ExecutionState,
)
from ..modules import (
    Turn,
)
from ..events import (
    StreamSink,
)
from ..utils.llm import (
    LLMRequest,
    LLMResponse,
    Message,
    ToolDefinition,
    Usage,
)


_BACKGROUND_TASKS: set[asyncio.Task] = set()


class StepEngine:
    def __init__(
        self,
        *,
        llm,
        modules,
        tools,
        skills,
        agent_runtime,
    ) -> None:
        self.llm = llm
        self.modules = modules
        self.tools = tools
        self.skills = skills
        self.agent_runtime = agent_runtime

    async def execute(
        self,
        *,
        context,
        persona: str,
        history: list[Message],
        report_sink: Callable[
            [list[str]], None
        ]
        | None = None,
        sink: StreamSink | None = None,
    ) -> AgentResult:
        started_wall = time.time()

        state = ExecutionState(
            persona=persona,
        )

        turn = Turn(
            agent_hash=context.agent_hash,
            parent_hash=context.parent_hash,
            depth=context.depth,
            task=context.task,
            world=context.world,
            started_at=started_wall,
        )

        pending_query_records: dict[str, str] = {}

        def on_module_start(
            module_id: str,
        ) -> None:
            if sink is None:
                return

            pending_query_records[module_id] = (
                sink.record_started(
                    kind="module",
                    name=module_id,
                )
            )

        def on_module_result(
            module_id: str,
            result: str | None,
            duration: float,
            failed: bool,
        ) -> None:
            if sink is None:
                return

            record_id = (
                pending_query_records.pop(
                    module_id,
                    None,
                )
            )

            if record_id is None:
                return

            if failed:
                sink.record_failed(
                    record_id,
                    summary="query failed",
                )
                return

            if not result:
                sink.record_void(
                    record_id
                )
                return

            for line in result.splitlines():
                sink.record_detail(
                    record_id,
                    line,
                )

            sink.record_done(
                record_id,
                note=f"{duration:.1f}s",
            )

        # ----------------------------------------------------------
        # Observation: the whole world as ONE user message.
        # A subagent's task rides along as its own part.
        # ----------------------------------------------------------

        ambient_context = (
            await self.modules.query_snapshot(
                turn,
                **(
                    {
                        "on_start": on_module_start,
                        "on_result": on_module_result,
                    }
                    if sink is not None
                    else {}
                ),
            )
        )

        system_message = build_system(
            persona
        )

        turn_messages: list[Message] = [
            build_observation(
                ambient_context=ambient_context,
                task=(
                    context.task
                    if context.depth > 0
                    else None
                ),
            )
        ]

        reply_text: str | None = None
        error_text: str | None = None

        try:
            response = await self._generate(
                LLMRequest(
                    messages=[
                        system_message,
                        *history,
                        *turn_messages,
                    ],
                    tools=self._tool_definitions(),
                ),
                sink=sink,
            )

            # --------------------------------------------------
            # Plain text: the turn is over.
            # --------------------------------------------------

            if not response.tool_calls:
                if not (
                    response.content or ""
                ).strip():
                    logger.warning(
                        "[turn:{}] empty reply "
                        "(finish={})",
                        context.agent_hash[:8],
                        response.finish_reason,
                    )

                turn_messages.append(
                    _assistant_message(
                        response.content or ""
                    )
                )

                reply_text = (
                    response.content or ""
                )

                return AgentResult(
                    content=reply_text,
                    messages=tuple(
                        turn_messages
                    ),
                    response=response,
                )

            # --------------------------------------------------
            # Results: run every call, append them.
            # The next turn (new observation) is the
            # caller's business.
            # --------------------------------------------------

            turn_messages.append(
                _assistant_message_with_calls(
                    response
                )
            )

            for call in response.tool_calls:
                logger.info(
                    "[turn:{}] tool {} {}",
                    context.agent_hash[:8],
                    call.name,
                    json.dumps(
                        call.arguments
                    ),
                )

                result_text = (
                    await self._run_call(
                        context=context,
                        state=state,
                        call=call,
                        sink=sink,
                    )
                )

                turn_messages.append(
                    _tool_message(
                        call.id,
                        result_text,
                    )
                )

            return AgentResult(
                content=None,
                messages=tuple(
                    turn_messages
                ),
                response=response,
            )

        except asyncio.CancelledError:
            error_text = "cancelled"
            raise

        except Exception as exc:
            error_text = (
                f"{type(exc).__name__}: {exc}"
            )
            raise

        finally:
            self.modules.deliver_turn(
                replace(
                    turn,
                    reply=reply_text,
                    error=error_text,
                    ended_at=time.time(),
                )
            )

            pending = [
                child
                for child in state.children
                if not child.reported
            ]

            if (
                pending
                and report_sink is not None
            ):
                task = asyncio.create_task(
                    self._archive_children(
                        pending,
                        report_sink,
                    ),
                    name="subagent-report-archive",
                )

                _BACKGROUND_TASKS.add(task)

                task.add_done_callback(
                    _BACKGROUND_TASKS.discard
                )

    # ------------------------------------------------------------------
    # Call routing
    # ------------------------------------------------------------------

    async def _run_call(
        self,
        *,
        context,
        state: ExecutionState,
        call,
        sink: StreamSink | None = None,
    ) -> str:
        record_id = None
        started = time.time()

        if sink is not None:
            if (
                call.name
                == SPAWN_TOOL_NAME
            ):
                kind = "agent"
            elif (
                call.name
                == INVOKE_SKILL_TOOL_NAME
            ):
                kind = "skill"
            elif (
                call.name
                == INVOKE_TOOL_TOOL_NAME
            ):
                kind = "tool"
            else:
                kind = "verb"

            record_id = sink.record_started(
                kind=kind,
                name=call.name,
            )

            for line in _pretty_args(
                call.arguments
            ):
                sink.record_detail(
                    record_id,
                    line,
                )

        try:
            verb = VERBS.get(
                call.name
            )

            if verb is None:
                result_text = (
                    f"Unknown action '{call.name}'. "
                    f"Available actions: "
                    f"{', '.join(VERBS)}."
                )

            else:
                result_text = (
                    await verb.execute(
                        call=call,
                        context=context,
                        state=state,
                        engine=self,
                    )
                )

        except Exception as exc:
            if (
                sink is not None
                and record_id
            ):
                sink.record_failed(
                    record_id,
                    summary=(
                        f"{type(exc).__name__}"
                    ),
                )

            raise

        if (
            sink is not None
            and record_id
        ):
            for line in _result_lines(
                result_text
            ):
                sink.record_detail(
                    record_id,
                    line,
                )

            sink.record_done(
                record_id,
                summary=_compact_result(
                    result_text
                ),
                note=(
                    f"{time.time() - started:.1f}s"
                ),
            )

        return result_text

    async def _generate(
        self,
        request,
        *,
        sink: StreamSink | None = None,
    ) -> LLMResponse:
        """
        Stream the model. Without a sink this is exactly
        generate_complete(); with one, text deltas are forwarded
        live and withdrawn if tool calls materialize.
        """
        if sink is None:
            return await self.llm.generate_complete(
                request
            )

        text_parts: list[str] = []
        tool_calls: list = []
        usage = Usage()
        finish_reason: str = "unknown"
        stream_id: str | None = None
        first_text: float | None = None

        stream_id = sink.output_started()

        async for event in self.llm.generate(
            request
        ):
            if event.kind == "text":
                if event.text:
                    if first_text is None:
                        first_text = time.time()

                    sink.output_delta(
                        stream_id,
                        event.text,
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
                sink.output_cancelled(
                    stream_id
                )

            else:
                sink.output_done(
                    stream_id,
                    duration=(
                        f"{time.time() - (first_text or time.time()):.1f}s"
                    ),
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
    ) -> list[ToolDefinition]:
        return [
            verb.definition()
            for verb in VERBS.values()
        ]

    async def _archive_children(
        self,
        pending,
        sink,
    ) -> None:
        reports: list[str] = []

        for child in pending:
            reports.append(
                await format_child_report(
                    child
                )
            )

        if reports:
            sink(reports)


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
    content. Message itself expects a concrete content value, so
    normalize None to an empty string at the agent boundary.
    """
    return Message(
        role="assistant",
        content=response.content or "",
        tool_calls=response.tool_calls,
    )


def _tool_message(
    tool_call_id: str,
    content: str,
) -> Message:
    return Message(
        role="tool",
        tool_call_id=tool_call_id,
        content=content,
    )


def _pretty_args(
    arguments: Any,
) -> list[str]:
    if not arguments:
        return []

    try:
        return json.dumps(
            arguments,
            ensure_ascii=False,
            indent=2,
        ).splitlines()

    except Exception:
        return [
            str(arguments)[:200]
        ]


def _result_lines(
    text: str,
) -> list[str]:
    return (
        (text or "").splitlines()
    )


def _compact_result(
    text: str,
) -> str:
    flat = (
        (text or "")
        .replace("\n", " ")
        .strip()
    )

    # MCP CallToolResult JSON: surface the human text,
    # not the envelope.
    if flat.startswith("{"):
        try:
            payload = json.loads(flat)

            if isinstance(
                payload,
                dict,
            ):
                parts = [
                    str(
                        block.get(
                            "text",
                            "",
                        )
                    )
                    for block in (
                        payload.get(
                            "content",
                            [],
                        )
                    )
                    if (
                        isinstance(
                            block,
                            dict,
                        )
                        and block.get(
                            "type"
                        )
                        == "text"
                    )
                ]

                if parts:
                    flat = (
                        " ".join(
                            parts
                        ).strip()
                    )

        except Exception:
            pass

    return flat

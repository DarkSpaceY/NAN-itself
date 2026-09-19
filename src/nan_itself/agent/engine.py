"""
Step engine: how ANY agent (main or sub) thinks for one turn.

A turn is exactly ONE cycle of three phases:

    observation   one user message carrying ambient Module
                  context (the inbox included); a subagent's
                  task and its finished children's late
                  reports are appended the same way
    model call    the model sees [system, *history, *turn]
    result        plain text ends the turn; tool calls are run,
                  their results appended, and the turn ends --
                  the caller starts the next turn

Every agent is identical; actions come exclusively from the verb
registry. There is no in-engine loop: after tool calls the next
observation is rebuilt (fresh modules, fresh inbox), which is
what keeps the message prefix cacheable.

There is no persistent history anywhere. Each Turn carries the
history snapshot as of its start (`turn.history`); the next turn
derives its snapshot as `last_turn.history + last_turn.messages`,
cleared to empty over history_char_limit. A turn's full model
input is exactly `turn.history + turn.messages` -- reconstruction
needs nothing else.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import replace
from collections.abc import Sequence
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
    FINISH_TOOL_NAME,
    INVOKE_SKILL_TOOL_NAME,
    INVOKE_TOOL_TOOL_NAME,
    LIST_SKILLS_TOOL_NAME,
    LIST_TOOLS_TOOL_NAME,
    SHOW_SKILL_TOOL_NAME,
    SHOW_TOOL_TOOL_NAME,
    SLEEP_TOOL_NAME,
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
        history_char_limit: int = 100_000,
    ) -> None:
        self.llm = llm
        self.modules = modules
        self.tools = tools
        self.skills = skills
        self.agent_runtime = agent_runtime

        # Shared history retention policy (main agent and
        # subagents alike): full retention, clear-all over
        # the character limit.
        self.history_char_limit = max(
            0,
            history_char_limit,
        )

    async def execute(
        self,
        *,
        context,
        persona: str,
        last_turn: Turn | None = None,
        state: ExecutionState | None = None,
        report_sink: Callable[
            [list[str]], None
        ]
        | None = None,
        pending_reports: Sequence[str] = (),
        sink: StreamSink | None = None,
    ) -> AgentResult:
        """
        Run one turn: observe once, call the model once.

        `last_turn` is the previous Turn of the same agent (None
        on its first turn). Its history + messages derive this
        turn's history snapshot; the clear-all policy applies at
        derivation time. The completed Turn rides out on
        AgentResult.turn -- callers keep no history themselves.

        `state` is the execution state verbs may read or change.
        Callers that omit it get a fresh per-turn state (the main
        agent); the subagent worker passes one instance per AGENT
        so its children list spans the whole task.

        `report_sink` receives the formatted reports of this
        turn's finished children: the root parks them into the
        inbox, a subagent's worker buffers them for its own next
        observation.

        `pending_reports` are already-finished child reports to
        fold into this turn's observation -- the delivery path
        that hands a subagent its own children's reports.
        """
        started_wall = time.time()

        # ----------------------------------------------------------
        # History snapshot: derived from the last turn; clear-all
        # over the limit. No persistent history exists anywhere.
        # ----------------------------------------------------------

        prior: tuple = ()

        if last_turn is not None:
            prior = (
                last_turn.history
                + last_turn.messages
            )

            if (
                self._history_chars(prior)
                > self.history_char_limit
            ):
                logger.info(
                    "History exceeded {} characters; "
                    "clearing conversation history",
                    self.history_char_limit,
                )

                prior = ()

        if state is None:
            state = ExecutionState(
                persona=persona,
                sink=sink,
            )

        turn = Turn(
            agent_hash=context.agent_hash,
            parent_hash=context.parent_hash,
            depth=context.depth,
            task=context.task,
            world=context.world,
            persona=persona,
            history=prior,
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
                reports=pending_reports,
            )
        ]

        response = None
        error_text: str | None = None

        def _completed() -> Turn:
            """
            The finished Turn, filled in one place. A successful
            return builds it for AgentResult.turn and delivery
            builds it again in finally -- only ever on success,
            where both are equal.
            """
            return replace(
                turn,
                messages=tuple(
                    turn_messages
                ),
                usage=(
                    response.usage
                    if response is not None
                    else None
                ),
                finish_reason=(
                    response.finish_reason
                    if response is not None
                    else None
                ),
                model=self.llm.model,
                error=error_text,
                ended_at=time.time(),
            )

        try:
            response = await self._generate(
                LLMRequest(
                    messages=[
                        system_message,
                        *prior,
                        *turn_messages,
                    ],
                    tools=self._tool_definitions(
                        depth=context.depth
                    ),
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

                reply_text = (
                    response.content or ""
                )

                turn_messages.append(
                    _assistant_message(
                        reply_text
                    )
                )

                return AgentResult(
                    content=reply_text,
                    turn=_completed(),
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

            # --------------------------------------------------
            # finish: the subagent task is over; the report
            # rides out as the result content.
            # --------------------------------------------------

            if state.finished:
                return AgentResult(
                    content=state.report,
                    turn=_completed(),
                    finished=True,
                )

            return AgentResult(
                content=None,
                turn=_completed(),
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
                _completed()
            )

            pending = [
                child
                for child in state.children
                if not child.reported
                and not child.archiving
            ]

            # A finished subagent's worker delivers its children's
            # reports inline (_settle_children in verbs.py);
            # background archiving here would only double-format
            # them into a buffer nobody drains anymore.
            if (
                pending
                and report_sink is not None
                and not state.finished
            ):
                for child in pending:
                    child.archiving = True

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
            record_id = sink.record_started(
                kind=_RECORD_KINDS.get(
                    call.name,
                    "verb",
                ),
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
        depth: int = 0,
    ) -> list[ToolDefinition]:
        verbs = list(VERBS.values())

        # finish ends a subagent's loop; the main agent has no
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

    @staticmethod
    def _history_chars(
        messages,
    ) -> int:
        return sum(
            len(message.content or "")
            for message in messages
        )

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

            # Delivery done: exclude these children from any
            # later archive pass.
            for child in pending:
                child.reported = True


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


# UI record kind per verb: tools, skills, spawning, sleeping and
# finishing each get their own kind; anything else stays a
# generic verb.
_RECORD_KINDS = {
    INVOKE_TOOL_TOOL_NAME: "tool",
    LIST_TOOLS_TOOL_NAME: "tool",
    SHOW_TOOL_TOOL_NAME: "tool",
    INVOKE_SKILL_TOOL_NAME: "skill",
    LIST_SKILLS_TOOL_NAME: "skill",
    SHOW_SKILL_TOOL_NAME: "skill",
    SPAWN_TOOL_NAME: "spawn",
    SLEEP_TOOL_NAME: "sleep",
    FINISH_TOOL_NAME: "finish",
}


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
    content to an empty string at the agent boundary so downstream
    snapshot handling never has to handle None.
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

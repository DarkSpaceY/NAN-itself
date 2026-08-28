"""
Step engine: how ANY agent (main or sub) thinks for one turn.

Depth-aware behaviour comes exclusively from RolePolicy; actions
come exclusively from the verb registry. This module has no
concept of a "main agent".

A turn ends when the model produces plain text, raises, or the
process is stopping — there is no step cap.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any, Callable

import mcp.types as mcp_types
from loguru import logger

from .model import (
    AgentResult,
    AgentTurn,
)
from .prompts import (
    build_messages,
    format_skill_section,
    render_running_subagents,
)
from .reports import (
    collect_finished_children,
    format_child_report,
)
from .role import (
    RolePolicy,
)
from .verbs import (
    ACTIVATE_SKILL_TOOL_NAME,
    DISPATCH_SUBAGENT_TOOL_NAME,
    VERBS,
    ExecutionState,
)
from ..modules import (
    TurnRecord,
)
from ..skills import (
    Skill,
)
from ..tools import (
    AgentToolView,
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
        providers,
        skills,
        agent_runtime,
    ) -> None:
        self.llm = llm
        self.modules = modules
        self.providers = providers
        self.skills = skills
        self.agent_runtime = agent_runtime

    async def execute(
        self,
        *,
        context,
        user_input: str,
        persona: str,
        seed_reports: list[str] | None = None,
        report_sink: Callable[
            [list[str]], None
        ]
        | None = None,
        sink: StreamSink | None = None,
    ) -> AgentResult:
        started_wall = time.time()

        state = ExecutionState(
            active_skill=(
                context.skill
                if isinstance(context.skill, Skill)
                else None
            ),
            persona=persona,
        )

        turn = AgentTurn(
            turn_id=uuid.uuid4().hex,
            agent_hash=context.agent_hash,
            depth=context.depth,
            user_input=user_input,
            data=context.world,
            task=context.task,
        )

        pending_query_records: dict[str, str] = {}

        def on_module_start(module_id: str) -> None:
            if sink is None:
                return

            pending_query_records[module_id] = sink.record_started(
                kind="module",
                name=module_id,
            )

        def on_module_result(
            module_id: str,
            result: str | None,
            duration: float,
            failed: bool,
        ) -> None:
            if sink is None:
                return

            record_id = pending_query_records.pop(module_id, None)

            if record_id is None:
                return

            if failed:
                sink.record_failed(
                    record_id,
                    summary="query failed",
                )
                return

            if not result:
                sink.record_void(record_id)
                return

            for line in result.splitlines():
                sink.record_detail(record_id, line)

            sink.record_done(
                record_id,
                note=f"{duration:.1f}s",
            )

        ambient_context = await self.modules.query_snapshot(
            turn,
            context.world,
            **(
                {"on_start": on_module_start, "on_result": on_module_result}
                if sink is not None
                else {}
            ),
        )

        provider_view = AgentToolView(self.providers)

        current_messages: list[Message] = [
            _user_message(report)
            for report in (seed_reports or [])
        ]

        current_messages.append(
            _user_message(user_input)
        )

        reply_text: str | None = None

        error_text: str | None = None

        try:
            while True:
                reports = await collect_finished_children(
                    state.children
                )

                for report in reports:
                    current_messages.append(
                        _user_message(report)
                    )

                skill_section = (
                    format_skill_section(
                        state.active_skill,
                        self.skills.catalog(),
                    )
                    if RolePolicy.injects_skills_section(
                        context.depth
                    )
                    else None
                )

                request_messages = build_messages(
                    persona=persona,
                    skill_section=skill_section,
                    ambient_context=ambient_context,
                    current=current_messages,
                    running_subagents=render_running_subagents(
                        [
                            child
                            for child in state.children
                            if not child.reported
                        ]
                    ),
                )

                tool_definitions = await self._tool_definitions(
                    provider_view=provider_view,
                    depth=context.depth,
                )

                if os.getenv("NAN_TRACE_MESSAGES") == "1":
                    logger.debug(
                        "[turn:{}] system:\n{}",
                        context.agent_hash[:8],
                        request_messages[0].content,
                    )

                response = await self._generate(
                    LLMRequest(
                        messages=request_messages,
                        tools=tool_definitions,
                    ),
                    sink=sink,
                )

                if not response.tool_calls:
                    if not (response.content or "").strip():
                        # Observed but not corrected: an empty
                        # reply simply ends the turn. Causes seen
                        # in the wild: silent tool-call parse
                        # drop (Ollama), max-tokens truncation,
                        # immediate EOS.
                        logger.warning(
                            "[turn:{}] empty reply "
                            "(finish={})",
                            context.agent_hash[:8],
                            response.finish_reason,
                        )

                    current_messages.append(
                        _assistant_message(
                            response.content or ""
                        )
                    )

                    reply_text = response.content or ""

                    return AgentResult(
                        content=reply_text,
                        messages=tuple(current_messages),
                        response=response,
                    )

                current_messages.append(
                    _assistant_message_with_calls(response)
                )

                for call in response.tool_calls:
                    logger.info(
                        "[turn:{}] tool {} {}",
                        context.agent_hash[:8],
                        call.name,
                        json.dumps(call.arguments),
                    )

                    result_text = await self._run_call(
                        context=context,
                        state=state,
                        provider_view=provider_view,
                        call=call,
                        sink=sink,
                    )

                    current_messages.append(
                        _tool_message(call.id, result_text)
                    )

        except asyncio.CancelledError:
            error_text = "cancelled"

            raise

        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"

            raise

        finally:
            self.modules.deliver_turn(
                TurnRecord(
                    agent_hash=context.agent_hash,
                    parent_hash=context.parent_hash,
                    depth=context.depth,
                    task=context.task,
                    user_input=user_input,
                    world=context.world,
                    reply=reply_text,
                    error=error_text,
                    started_at=started_wall,
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
                and RolePolicy.archives_orphan_reports(
                    context.depth
                )
                and report_sink is not None
            ):
                # The event loop only keeps a weak reference to
                # tasks; without this registry the archive job
                # can be garbage-collected mid-flight and the
                # report silently lost.
                task = asyncio.create_task(
                    self._archive_children(pending, report_sink),
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
        provider_view: AgentToolView,
        call,
        sink: StreamSink | None = None,
    ) -> str:
        record_id = None
        started = time.time()

        if sink is not None:
            if call.name == DISPATCH_SUBAGENT_TOOL_NAME:
                kind = "agent"
            elif call.name == ACTIVATE_SKILL_TOOL_NAME:
                kind = "skill"
            elif call.name in VERBS:
                kind = "verb"
            else:
                kind = "tool"

            record_id = sink.record_started(
                kind=kind,
                name=call.name,
            )

            for line in _pretty_args(call.arguments):
                sink.record_detail(record_id, line)

        try:
            verb = VERBS.get(call.name)

            if verb is not None:
                if not verb.visible(context.depth, RolePolicy):
                    result_text = RolePolicy.hidden_verb_reply(
                        call.name,
                    )
                else:
                    result_text = await verb.execute(
                        call=call,
                        context=context,
                        state=state,
                        engine=self,
                    )

                if (
                    call.name == ACTIVATE_SKILL_TOOL_NAME
                    and sink is not None
                    and isinstance(state.active_skill, Skill)
                ):
                    for line in _skill_structure(
                        state.active_skill,
                    ):
                        sink.record_detail(record_id, line)
            else:
                result = await provider_view.call_tool(
                    call.name,
                    call.arguments,
                )

                result_text = _serialize_tool_result(result)

        except Exception as exc:
            if sink is not None and record_id:
                sink.record_failed(
                    record_id,
                    summary=f"{type(exc).__name__}",
                )

            raise

        if sink is not None and record_id:
            for line in _result_lines(result_text):
                sink.record_detail(record_id, line)

            sink.record_done(
                record_id,
                summary=_compact_result(result_text),
                note=f"{time.time() - started:.1f}s",
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
            return await self.llm.generate_complete(request)

        text_parts: list[str] = []
        tool_calls: list = []
        usage = Usage()
        finish_reason: str = "unknown"
        stream_id: str | None = None
        first_text: float | None = None

        # The live caret appears the moment the request is sent,
        # not when the first token lands: silence is visible.
        if sink is not None:
            stream_id = sink.output_started()

        async for event in self.llm.generate(request):
            if event.kind == "text":
                if event.text:
                    if first_text is None:
                        first_text = time.time()

                    sink.output_delta(stream_id, event.text)
                    text_parts.append(event.text)

            elif event.kind == "tool_call":
                if event.tool_call is not None:
                    tool_calls.append(event.tool_call)

            elif event.kind == "done":
                if event.usage is not None:
                    usage = event.usage

                if event.finish_reason is not None:
                    finish_reason = event.finish_reason

        if stream_id is not None:
            if tool_calls:
                # Text streamed before tool calls: not the turn's
                # final output, so withdraw it from the stream.
                sink.output_cancelled(stream_id)

            else:
                sink.output_done(
                    stream_id,
                    duration=(
                        f"{time.time() - (first_text or time.time()):.1f}s"
                    ),
                )

        return LLMResponse(
            content="".join(text_parts) or None,
            tool_calls=tool_calls,
            model=self.llm.model,
            usage=usage,
            provider=self.llm.provider,
            finish_reason=finish_reason,
        )

    async def _tool_definitions(
        self,
        *,
        provider_view: AgentToolView,
        depth: int = 0,
    ) -> list:
        definitions = [
            ToolDefinition(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
            )
            for tool in await provider_view.list_tools()
        ]

        for verb in VERBS.values():
            if verb.visible(depth, RolePolicy):
                definitions.append(verb.definition())

        return definitions

    async def _archive_children(self, pending, sink) -> None:
        reports: list[str] = []

        for child in pending:
            reports.append(
                await format_child_report(child)
            )

        if reports:
            sink(reports)


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def _user_message(content: str) -> Message:
    return Message(role="user", content=content)


def _assistant_message(content: str) -> Message:
    return Message(role="assistant", content=content)


def _assistant_message_with_calls(response) -> Message:
    return Message(
        role="assistant",
        content=response.content,
        tool_calls=response.tool_calls,
    )


def _tool_message(tool_call_id: str, content: str) -> Message:
    return Message(
        role="tool",
        tool_call_id=tool_call_id,
        content=content,
    )


def _compact_args(arguments: Any) -> str:
    if not arguments:
        return ""

    try:
        return json.dumps(arguments, ensure_ascii=False)
    except Exception:
        return str(arguments)[:96]


def _pretty_args(arguments: Any) -> list[str]:
    if not arguments:
        return []

    try:
        return json.dumps(
            arguments,
            ensure_ascii=False,
            indent=2,
        ).splitlines()
    except Exception:
        return [str(arguments)[:200]]


def _result_lines(text: str) -> list[str]:
    lines = (text or "").splitlines()
    return lines


def _skill_structure(skill: Any) -> list[str]:
    meta = skill.metadata

    lines = [
        f"name: {meta.name}",
        f"description: {meta.description}",
        f"origin: {meta.origin}",
    ]

    resources = getattr(skill, "resources", None) or []

    if resources:
        lines.append(f"resources: {len(resources)}")

        for path in resources:
            lines.append(f"  · {path}")

    body = (skill.instructions or "").strip()

    if body:
        lines.append("── body ──")
        lines.extend(body.splitlines())

    return lines


def _compact_result(text: str) -> str:
    flat = (text or "").replace("\n", " ").strip()

    # MCP CallToolResult JSON: surface the human text, not the envelope.
    if flat.startswith("{"):
        try:
            payload = json.loads(flat)

            if isinstance(payload, dict):
                parts = [
                    str(block.get("text", ""))
                    for block in payload.get("content", [])
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                ]

                if parts:
                    flat = " ".join(parts).strip()
        except Exception:
            pass

    return flat


def _serialize_tool_result(result: Any) -> str:
    if isinstance(
        result,
        mcp_types.CallToolResult,
    ):
        try:
            dumped = result.model_dump(mode="json")

            return json.dumps(
                dumped,
                ensure_ascii=False,
            )
        except Exception:
            return str(result)

    if isinstance(result, str):
        return result

    try:
        return json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        return str(result)

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

from src.nan_itself.agent.model import (
    AgentResult,
    AgentTurn,
)
from src.nan_itself.agent.prompts import (
    build_messages,
    format_skill_section,
    render_running_subagents,
)
from src.nan_itself.agent.reports import (
    collect_finished_children,
    format_child_report,
)
from src.nan_itself.agent.role import (
    RolePolicy,
)
from src.nan_itself.agent.verbs import (
    VERBS,
    ExecutionState,
)
from src.nan_itself.modules import (
    TurnRecord,
)
from src.nan_itself.skills import (
    Skill,
)
from src.nan_itself.tools import (
    AgentToolView,
)
from src.nan_itself.utils.llm import (
    LLMRequest,
    Message,
    ToolDefinition,
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

        ambient_context = await self.modules.query_snapshot(
            turn,
            context.world,
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

                response = await self.llm.generate_complete(
                    LLMRequest(
                        messages=request_messages,
                        temperature=0.7,
                        max_tokens=4096,
                        tools=tool_definitions,
                    )
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
    ) -> str:
        verb = VERBS.get(call.name)

        if verb is not None:
            if not verb.visible(context.depth, RolePolicy):
                return RolePolicy.hidden_verb_reply(
                    call.name
                )

            return await verb.execute(
                call=call,
                context=context,
                state=state,
                engine=self,
            )

        result = await provider_view.call_tool(
            call.name,
            call.arguments,
        )

        return _serialize_tool_result(result)

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

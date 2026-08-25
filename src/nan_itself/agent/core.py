from __future__ import annotations

import asyncio
import json
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import mcp.types as mcp_types
from loguru import logger

from src.nan_itself.agent.runtime import (
    AgentContext,
    AgentRuntime,
    SubagentHandle,
    SubagentLimitError,
)
from src.nan_itself.skills.facade import (
    Skill,
    SkillRuntime,
)
from src.nan_itself.tools.facade import (
    AgentToolView,
    ProviderRuntime,
)
from src.nan_itself.utils.llm import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
)


DEFAULT_MAX_STEPS = 64
DEFAULT_HISTORY_TOKEN_BUDGET = 2048

SLEEP_TOOL_NAME = "sleep"
DISPATCH_SUBAGENT_TOOL_NAME = "dispatch_subagent"
AWAIT_SUBAGENTS_TOOL_NAME = "await_subagents"
ACTIVATE_SKILL_TOOL_NAME = "activate_skill"

_REPORT_PREFIX = "[Subagent Report]"
_REPORT_TASK_PREVIEW_LIMIT = 200


# ============================================================================
# Token estimation
# ============================================================================


def _estimate_tokens(
    text: str,
) -> int:
    """
    Cheap heuristic token estimate.

    CJK characters count roughly one token each; other text is
    approximated at four characters per token. Good enough for
    budgeting a deliberately small history window.
    """
    if not text:
        return 0

    cjk = 0

    for character in text:
        code = ord(character)

        if (
            0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0x3000 <= code <= 0x303F
            or 0xFF00 <= code <= 0xFFEF
        ):
            cjk += 1

    other = len(text) - cjk

    return cjk + (other + 3) // 4


def _message_token_cost(
    message: Message,
) -> int:
    cost = 4 + _estimate_tokens(
        message.content or ""
    )

    for call in message.tool_calls:
        cost += 4
        cost += _estimate_tokens(call.name)
        cost += _estimate_tokens(
            json.dumps(
                call.arguments,
                ensure_ascii=False,
            )
        )

    return cost


@dataclass
class _ChildSubagent:
    """
    Execution-local record of one dispatched child.

    Handles are never exposed to the model. Reports are delivered
    automatically at step boundaries, via the await_subagents
    barrier, or into the next turn when they arrive late.
    """

    id: str
    task: str
    handle: SubagentHandle
    reported: bool = False


@dataclass(frozen=True)
class AgentTurn:
    """
    One Core / Subagent turn.

    `data` is the shared world snapshot for the entire dispatch tree.
    """

    turn_id: str
    agent_hash: str
    depth: int

    user_input: str

    data: Mapping[str, Mapping[str, Any]]

    task: str | None = None


@dataclass(frozen=True)
class ConversationTurn:
    """
    Compact persistent conversation unit.

    One turn may contain multiple assistant/tool messages.
    This is what gets retained in the short-term history window.
    """

    messages: tuple[Message, ...]


@dataclass(frozen=True)
class AgentResult:
    """
    Result of one completed Agent execution.
    """

    content: str
    messages: tuple[Message, ...]
    response: LLMResponse | None


class CoreAgent:
    """
    Core Agent runtime.

    Main Agent:
        permanently uses core_skill and never sees the
        concept of Skills.

    Subagent:
        inherits its parent's Skill at start and may switch
        its own Skill via activate_skill.

    All descendants of one Core turn share the same world snapshot.

    Subagent reports are pushed back automatically:
        - finished children are injected at the next step boundary;
        - await_subagents blocks until every outstanding child is
          done and returns all reports at once;
        - reports arriving after the turn ended are delivered at
          the start of the next main turn.

    History follows the "less history, more observation" philosophy:
    durable state lives in Module observations, so the conversation
    window is bounded by a small token budget instead of turn counts.
    """

    def __init__(
        self,
        *,
        llm: LLMProvider,
        modules: Any,
        providers: ProviderRuntime,
        skills: SkillRuntime,
        core_skill: Skill,
        max_subagent_depth: int = 3,
        history_token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        if history_token_budget < 1:
            raise ValueError(
                "history_token_budget must be >= 1"
            )

        if max_steps < 1:
            raise ValueError(
                "max_steps must be >= 1"
            )

        self.llm = llm
        self.modules = modules
        self.providers = providers
        self.skills = skills
        self.core_skill = core_skill

        self.agent_runtime = AgentRuntime(
            max_subagent_depth=max_subagent_depth,
        )

        self.history_token_budget = history_token_budget
        self.max_steps = max_steps

        self._main_history: list[
            ConversationTurn
        ] = []

        # Reports from children that finished after their parent's
        # turn already ended. Delivered at the start of the next run().
        self._late_reports: deque[str] = deque()

    # ==================================================================
    # Main Agent
    # ==================================================================

    async def run(
        self,
        user_input: str,
    ) -> AgentResult:
        """
        Run one Main Agent turn.

        A new world snapshot is captured exactly once.
        Late subagent reports are seeded into this turn.
        """
        world = self.modules.snapshot()

        root = self.agent_runtime.create_root(
            world=world,
            skill=self.core_skill,
            task=user_input,
        )

        seed_reports = list(
            self._late_reports
        )

        self._late_reports.clear()

        result = await self._run_agent(
            context=root,
            user_input=user_input,
            history=self._main_history,
            seed_reports=seed_reports,
        )

        # Keep only what fits the token budget; the newest turn is
        # always retained even when it alone exceeds the budget.
        self._main_history.append(
            ConversationTurn(
                messages=result.messages,
            )
        )

        while (
            len(self._main_history) > 1
            and self._history_cost(
                self._main_history
            )
            > self.history_token_budget
        ):
            self._main_history.pop(0)

        return result

    # ==================================================================
    # Generic Agent execution
    # ==================================================================

    async def _run_agent(
        self,
        *,
        context: AgentContext,
        user_input: str,
        history: list[ConversationTurn],
        seed_reports: list[str] | None = None,
    ) -> AgentResult:
        # The Skill is inherited through the AgentContext and may
        # be switched mid-execution by activate_skill (Subagents
        # only). The Main Agent stays pinned to core_skill.
        active_skill: Skill = (
            context.skill
            if isinstance(context.skill, Skill)
            else self.core_skill
        )

        skill_catalog = (
            self._format_skill_catalog()
            if context.depth > 0
            else []
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

        provider_view = self.providers.create_agent_view()

        children: list[_ChildSubagent] = []

        corrected_empty = False

        current_messages: list[Message] = [
            Message(
                role="user",
                content=report,
            )
            for report in (seed_reports or [])
        ]

        current_messages.append(
            Message(
                role="user",
                content=user_input,
            )
        )

        try:
            for _step in range(self.max_steps):
                # Deliver reports from children that finished since
                # the previous step.
                for report in await self._collect_finished_reports(
                    children,
                ):
                    current_messages.append(
                        Message(
                            role="user",
                            content=report,
                        )
                    )

                request_messages = (
                    self._build_messages(
                        skill=active_skill,
                        ambient_context=ambient_context,
                        history=history,
                        current=current_messages,
                        skill_catalog=skill_catalog,
                    )
                )

                tool_definitions = await self._tool_definitions(
                    provider_view=provider_view,
                    depth=context.depth,
                )

                response = await self.llm.generate_complete(
                    LLMRequest(
                        messages=request_messages,
                        temperature=0.7,
                        max_tokens=4096,
                        tools=tool_definitions,
                    )
                )

                empty_reply = (
                    not response.tool_calls
                    and not (response.content or "").strip()
                )

                if empty_reply:
                    # Some backends (e.g. Ollama) silently discard
                    # unparseable tool-call output: empty content
                    # while tokens were counted. Feed back a
                    # corrective note once so the model re-aims at
                    # the tools it actually has.
                    logger.warning(
                        "[turn:{}] empty reply "
                        "(finish={}); possible silent "
                        "tool-call parse drop or idle cycle",
                        context.agent_hash[:8],
                        response.finish_reason,
                    )

                    if not corrected_empty:
                        corrected_empty = True

                        tool_names = ", ".join(
                            tool.name
                            for tool in tool_definitions
                        )

                        current_messages.append(
                            Message(
                                role="assistant",
                                content="",
                            )
                        )

                        current_messages.append(
                            Message(
                                role="user",
                                content=(
                                    "[system] 你的上一条输出为空且无法解析。"
                                    f"当前可用工具：{tool_names}。"
                                    "若要使用未激活的工具组，先调用 route 激活；"
                                    "然后调用合适的工具，或直接用文本回答。"
                                    "不要调用不存在的工具。"
                                ),
                            )
                        )

                        continue

                if not response.tool_calls:
                    assistant_message = Message(
                        role="assistant",
                        content=response.content or "",
                    )

                    current_messages.append(
                        assistant_message
                    )

                    return AgentResult(
                        content=response.content or "",
                        messages=tuple(
                            current_messages
                        ),
                        response=response,
                    )

                current_messages.append(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )

                for tool_call in response.tool_calls:
                    if (
                        context.depth > 0
                        and tool_call.name
                        == ACTIVATE_SKILL_TOOL_NAME
                    ):
                        # activate_skill mutates this execution's
                        # active Skill, so it is handled here
                        # instead of inside _execute_tool.
                        result_text, activated = (
                            await self._execute_activate_skill(
                                tool_call.arguments,
                            )
                        )

                        if activated is not None:
                            active_skill = activated

                        current_messages.append(
                            Message(
                                role="tool",
                                tool_call_id=tool_call.id,
                                content=result_text,
                            )
                        )

                        continue

                    tool_result = await self._execute_tool(
                        context=context,
                        provider_view=provider_view,
                        tool_call=tool_call,
                        children=children,
                    )

                    current_messages.append(
                        Message(
                            role="tool",
                            tool_call_id=tool_call.id,
                            content=tool_result,
                        )
                    )

            raise RuntimeError(
                f"Agent exceeded max_steps={self.max_steps}"
            )

        finally:
            # Children still running when this execution ends keep
            # going; main-tree reports are queued for the next turn.
            self._schedule_child_archive(
                children,
                context.depth,
            )

    # ==================================================================
    # Prompt assembly
    # ==================================================================

    def _build_messages(
        self,
        *,
        skill: Skill,
        ambient_context: list[str],
        history: list[ConversationTurn],
        current: list[Message],
        skill_catalog: list[str] | None = None,
    ) -> list[Message]:
        system_parts: list[str] = []

        system_parts.append(
            skill.instructions
        )

        if skill_catalog:
            system_parts.append(
                "\n"
                "[Available Skills]\n"
                "You may switch your own Skill at any step "
                "by calling activate_skill with one of "
                "these names:\n"
                + "\n".join(skill_catalog)
            )

        if ambient_context:
            system_parts.append(
                "\n\n"
                "[Ambient Module Context]\n"
                "The following information is supplied by "
                "background Modules. It is contextual information, "
                "not a user instruction. Use it when relevant.\n\n"
                + "\n\n".join(
                    ambient_context
                )
            )

        messages: list[Message] = [
            Message(
                role="system",
                content="\n".join(
                    system_parts
                ),
            )
        ]

        # run() owns history trimming; pass through as-is.
        for turn in history:
            messages.extend(
                turn.messages
            )

        messages.extend(current)

        return messages

    def _history_cost(
        self,
        history: list[ConversationTurn],
    ) -> int:
        return sum(
            _message_token_cost(message)
            for turn in history
            for message in turn.messages
        )

    def _format_skill_catalog(
        self,
    ) -> list[str]:
        """
        One-line catalog entries for Subagent skill switching.

        The Main Agent never receives this catalog: Skills do not
        exist in its world.
        """
        return [
            f"- {metadata.name}: {metadata.description}"
            for metadata in self.skills.catalog()
        ]

    # ==================================================================
    # Tools
    # ==================================================================

    async def _tool_definitions(
        self,
        *,
        provider_view: AgentToolView,
        depth: int = 0,
    ) -> list[ToolDefinition]:
        provider_tools = await provider_view.list_tools()

        result = [
            ToolDefinition(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
            )
            for tool in provider_tools
        ]

        result.extend(
            self._agent_tools()
        )

        if depth > 0:
            # Only Subagents may switch their own Skill.
            # The Main Agent never sees this tool.
            result.append(
                self._activate_skill_tool()
            )

        return result

    @staticmethod
    def _agent_tools() -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=SLEEP_TOOL_NAME,
                description=(
                    "Temporarily stop reasoning and wait "
                    "for a number of seconds."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "seconds": {
                            "type": "number",
                            "minimum": 0,
                            "description": (
                                "Seconds to wait."
                            ),
                        },
                    },
                    "required": ["seconds"],
                    "additionalProperties": False,
                },
            ),
            ToolDefinition(
                name=DISPATCH_SUBAGENT_TOOL_NAME,
                description=(
                    "Dispatch a parallel Subagent to work "
                    "on a task. Dispatch several in the same "
                    "response when tasks are independent. "
                    "Each Subagent starts immediately and its "
                    "final report is delivered to this agent's "
                    "context automatically once it finishes."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": (
                                "The task to delegate."
                            ),
                        },
                    },
                    "required": [
                        "task",
                    ],
                    "additionalProperties": False,
                },
            ),
            ToolDefinition(
                name=AWAIT_SUBAGENTS_TOOL_NAME,
                description=(
                    "Block until every subagent dispatched "
                    "so far has finished, then receive all "
                    "of their reports at once. Returns "
                    "immediately when nothing is outstanding."
                ),
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
        ]

    @staticmethod
    def _activate_skill_tool() -> ToolDefinition:
        """
        Subagent-only tool for switching its own Skill.
        """
        return ToolDefinition(
            name=ACTIVATE_SKILL_TOOL_NAME,
            description=(
                "Switch this Subagent's own Skill. The new "
                "Skill takes effect from the next step."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "The name of the Skill to "
                            "activate."
                        ),
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        )

    # ==================================================================
    # Tool execution
    # ==================================================================

    async def _execute_tool(
        self,
        *,
        context: AgentContext,
        provider_view: AgentToolView,
        tool_call: ToolCall,
        children: list[_ChildSubagent],
    ) -> str:
        logger.info(
            "[turn:{}] tool {} {}",
            context.agent_hash[:8],
            tool_call.name,
            json.dumps(
                tool_call.arguments,
                ensure_ascii=False,
            )[:160],
        )

        if tool_call.name == SLEEP_TOOL_NAME:
            return await self._execute_sleep(
                tool_call.arguments,
            )

        if tool_call.name == ACTIVATE_SKILL_TOOL_NAME:
            # Reached only when the Main Agent (depth 0)
            # hallucinates this Subagent-only tool.
            return (
                "activate_skill is only available "
                "to Subagents."
            )

        if (
            tool_call.name
            == DISPATCH_SUBAGENT_TOOL_NAME
        ):
            return await self._execute_dispatch(
                context=context,
                arguments=tool_call.arguments,
                children=children,
            )

        if (
            tool_call.name
            == AWAIT_SUBAGENTS_TOOL_NAME
        ):
            return await self._execute_await_subagents(
                children,
            )

        result = await provider_view.call_tool(
            tool_call.name,
            tool_call.arguments,
        )

        return self._serialize_tool_result(
            result
        )

    async def _execute_sleep(
        self,
        arguments: dict[str, Any],
    ) -> str:
        seconds = arguments.get("seconds")

        if seconds is None:
            return (
                "sleep requires 'seconds'."
            )

        if not isinstance(
            seconds,
            (int, float),
        ):
            return (
                "'seconds' must be a number."
            )

        if seconds < 0:
            return (
                "'seconds' must be >= 0."
            )

        await self.agent_runtime.sleep(
            float(seconds)
        )

        return (
            f"Waited {float(seconds):g} seconds."
        )

    async def _execute_dispatch(
        self,
        *,
        context: AgentContext,
        arguments: dict[str, Any],
        children: list[_ChildSubagent],
    ) -> str:
        task = arguments.get("task")

        if not isinstance(task, str) or not task.strip():
            return (
                "dispatch_subagent requires "
                "a non-empty 'task'."
            )

        async def worker(
            child_context: AgentContext,
        ) -> AgentResult:
            # The child inherits its parent's Skill through
            # AgentContext and may switch it via activate_skill.
            return await self._run_agent(
                context=child_context,
                user_input=task,
                history=[],
            )

        try:
            handle = self.agent_runtime.dispatch(
                context,
                task=task,
                worker=worker,
            )

        except SubagentLimitError as exc:
            return str(exc)

        child = _ChildSubagent(
            id=handle.agent_hash[:8],
            task=task,
            handle=handle,
        )

        children.append(child)

        return (
            "Subagent dispatched.\n"
            f"id: {child.id}\n"
            f"depth: {handle.depth}\n"
            "It runs in parallel; its report will be "
            "delivered automatically."
        )

    async def _execute_await_subagents(
        self,
        children: list[_ChildSubagent],
    ) -> str:
        outstanding = [
            child
            for child in children
            if not child.reported
        ]

        if not outstanding:
            return (
                "No outstanding subagents."
            )

        # Concurrent barrier over the snapshot taken at call time.
        # Individual failures are formatted as failed reports.
        await asyncio.gather(
            *(
                child.handle.wait()
                for child in outstanding
            ),
            return_exceptions=True,
        )

        reports: list[str] = []

        for child in outstanding:
            child.reported = True

            reports.append(
                await self._format_child_report(
                    child
                )
            )

        return (
            f"{len(outstanding)} subagent(s) "
            "finished.\n\n"
            + "\n\n".join(reports)
        )

    async def _execute_activate_skill(
        self,
        arguments: dict[str, Any],
    ) -> tuple[str, Skill | None]:
        """
        Switch this Subagent's own Skill.

        Returns the tool-result text plus the newly activated
        Skill; the second item is None when activation failed
        and the current Skill should stay in effect.
        """
        name = arguments.get("name")

        if not isinstance(name, str) or not name.strip():
            return (
                "activate_skill requires 'name'. "
                f"Available Skills: "
                f"{', '.join(self.skills.names())}",
                None,
            )

        try:
            skill = self.skills.activate(name)

        except KeyError:
            return (
                f"Unknown Skill '{name}'. "
                f"Available Skills: "
                f"{', '.join(self.skills.names())}",
                None,
            )

        return (
            f"Skill '{skill.name}' activated. It takes "
            "effect from your next step.",
            skill,
        )

    # ==================================================================
    # Subagent report delivery
    # ==================================================================

    async def _collect_finished_reports(
        self,
        children: list[_ChildSubagent],
    ) -> list[str]:
        reports: list[str] = []

        for child in children:
            if child.reported:
                continue

            if not child.handle.done:
                continue

            child.reported = True

            reports.append(
                await self._format_child_report(
                    child
                )
            )

        return reports

    async def _format_child_report(
        self,
        child: _ChildSubagent,
    ) -> str:
        try:
            result = await child.handle.wait()

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            body = (
                "status: failed\n"
                f"error: {exc}"
            )

        else:
            body = (
                "status: completed\n"
                + (result.content or "")
            )

        task_preview = child.task

        if len(task_preview) > _REPORT_TASK_PREVIEW_LIMIT:
            task_preview = (
                task_preview[
                    :_REPORT_TASK_PREVIEW_LIMIT
                ]
                + "..."
            )

        return (
            f"{_REPORT_PREFIX}\n"
            f"id: {child.id}\n"
            f"task: {task_preview}\n"
            + body
        )

    def _schedule_child_archive(
        self,
        children: list[_ChildSubagent],
        depth: int,
    ) -> None:
        pending = [
            child
            for child in children
            if not child.reported
        ]

        if not pending:
            return

        asyncio.create_task(
            self._archive_children(
                pending,
                depth,
            ),
            name="subagent-report-archive",
        )

    async def _archive_children(
        self,
        pending: list[_ChildSubagent],
        depth: int,
    ) -> None:
        """
        Park reports of children outliving their execution.

        Only the main tree keeps them (delivered next turn);
        deeper orphaned results are dropped.
        """
        for child in pending:
            if depth != 0:
                continue

            try:
                report = (
                    await self._format_child_report(
                        child
                    )
                )

            except asyncio.CancelledError:
                raise

            except Exception:
                continue

            child.reported = True

            self._late_reports.append(
                report
            )

    def _drain_late_reports(self) -> list[str]:
        reports = list(self._late_reports)

        self._late_reports.clear()

        return reports

    # ==================================================================
    # Serialization helpers
    # ==================================================================

    @staticmethod
    def _serialize_tool_result(
        result: Any,
    ) -> str:
        if isinstance(
            result,
            mcp_types.CallToolResult,
        ):
            try:
                dumped = result.model_dump(
                    mode="json"
                )

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

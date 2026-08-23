from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

import mcp.types as mcp_types

from src.nan_itself.agent.runtime import (
    AgentContext,
    AgentRuntime,
    SubagentHandle,
    SubagentLimitError,
)
from src.nan_itself.modules.facade import ModuleState
from src.nan_itself.skills.facade import (
    Skill,
    SkillRuntime,
)
from src.nan_itself.tools.facade import (
    AgentMCPView,
    MCPRuntime,
)
from src.nan_itself.utils.llm import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
)


MAX_HISTORY_TURNS = 4
DEFAULT_MAX_STEPS = 64

SLEEP_TOOL_NAME = "sleep"
DISPATCH_SUBAGENT_TOOL_NAME = "dispatch_subagent"


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
        permanently uses core_skill.

    Subagent:
        may select another Skill, including core_skill.

    All descendants of one Core turn share the same world snapshot.
    """

    def __init__(
        self,
        *,
        llm: LLMProvider,
        modules: Any,
        mcp: MCPRuntime,
        skills: SkillRuntime,
        core_skill: Skill,
        max_subagent_depth: int = 3,
        history_turns: int = MAX_HISTORY_TURNS,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        if history_turns < 1:
            raise ValueError(
                "history_turns must be >= 1"
            )

        if max_steps < 1:
            raise ValueError(
                "max_steps must be >= 1"
            )

        self.llm = llm
        self.modules = modules
        self.mcp = mcp
        self.skills = skills
        self.core_skill = core_skill

        self.agent_runtime = AgentRuntime(
            max_subagent_depth=max_subagent_depth,
        )

        self.history_turns = history_turns
        self.max_steps = max_steps

        self._main_history: list[
            ConversationTurn
        ] = []

        self._subagents: dict[
            str,
            SubagentHandle,
        ] = {}

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
        """
        world = self.modules.snapshot()

        root = self.agent_runtime.create_root(
            world=world,
            skill=self.core_skill,
            task=user_input,
        )

        result = await self._run_agent(
            context=root,
            user_input=user_input,
            skill=self.core_skill,
            history=self._main_history,
        )

        # Keep only the recent conversation window.
        self._main_history.append(
            ConversationTurn(
                messages=result.messages,
            )
        )

        if len(self._main_history) > self.history_turns:
            self._main_history = (
                self._main_history[
                    -self.history_turns:
                ]
            )

        return result

    # ==================================================================
    # Generic Agent execution
    # ==================================================================

    async def _run_agent(
        self,
        *,
        context: AgentContext,
        user_input: str,
        skill: Skill,
        history: list[ConversationTurn],
    ) -> AgentResult:
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

        mcp_view = self.mcp.create_agent_view()

        current_messages: list[Message] = [
            Message(
                role="user",
                content=user_input,
            ),
        ]

        child_handles: dict[
            str,
            SubagentHandle,
        ] = {}

        last_response: LLMResponse | None = None

        for _step in range(self.max_steps):
            request_messages = (
                self._build_messages(
                    skill=skill,
                    ambient_context=ambient_context,
                    history=history,
                    current=current_messages,
                )
            )

            tool_definitions = await self._tool_definitions(
                mcp_view=mcp_view,
                skill=skill,
            )

            response = await self.llm.generate_complete(
                LLMRequest(
                    messages=request_messages,
                    temperature=0.7,
                    max_tokens=4096,
                    tools=tool_definitions,
                )
            )

            last_response = response

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
                tool_result = await self._execute_tool(
                    context=context,
                    skill=skill,
                    mcp_view=mcp_view,
                    tool_call=tool_call,
                    child_handles=child_handles,
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
    ) -> list[Message]:
        system_parts: list[str] = []

        system_parts.append(
            skill.instructions
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

        # Recent turns only.
        for turn in history[
            -self.history_turns:
        ]:
            messages.extend(
                turn.messages
            )

        messages.extend(current)

        return messages

    # ==================================================================
    # Tools
    # ==================================================================

    async def _tool_definitions(
        self,
        *,
        mcp_view: AgentMCPView,
        skill: Skill,
    ) -> list[ToolDefinition]:
        mcp_tools = await mcp_view.list_tools()

        result = [
            ToolDefinition(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
            )
            for tool in mcp_tools
        ]

        result.extend(
            self._agent_tools(
                allow_subagent=True,
            )
        )

        return result

    @staticmethod
    def _agent_tools(
        *,
        allow_subagent: bool,
    ) -> list[ToolDefinition]:
        tools = [
            ToolDefinition(
                name=SLEEP_TOOL_NAME,
                description=(
                    "Temporarily stop reasoning and wait. "
                    "Use seconds for a time delay, or "
                    "until_subagent to wait for a dispatched "
                    "Subagent to finish."
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
                        "until_subagent": {
                            "type": "string",
                            "description": (
                                "Subagent handle ID to wait for."
                            ),
                        },
                    },
                    "additionalProperties": False,
                },
            ),
        ]

        if allow_subagent:
            tools.append(
                ToolDefinition(
                    name=DISPATCH_SUBAGENT_TOOL_NAME,
                    description=(
                        "Dispatch a parallel Subagent to work "
                        "on a task. The Subagent starts immediately "
                        "and this call returns a handle ID instead "
                        "of waiting for completion."
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
                            "skill": {
                                "type": "string",
                                "description": (
                                    "Optional Skill to use. "
                                    "The special value 'core' "
                                    "selects Core Skill."
                                ),
                            },
                        },
                        "required": [
                            "task",
                        ],
                        "additionalProperties": False,
                    },
                )
            )

        return tools

    # ==================================================================
    # Tool execution
    # ==================================================================

    async def _execute_tool(
        self,
        *,
        context: AgentContext,
        skill: Skill,
        mcp_view: AgentMCPView,
        tool_call: ToolCall,
        child_handles: dict[
            str,
            SubagentHandle,
        ],
    ) -> str:
        if tool_call.name == SLEEP_TOOL_NAME:
            return await self._execute_sleep(
                tool_call.arguments,
                child_handles,
            )

        if (
            tool_call.name
            == DISPATCH_SUBAGENT_TOOL_NAME
        ):
            return await self._execute_dispatch(
                context=context,
                arguments=tool_call.arguments,
                child_handles=child_handles,
            )

        result = await mcp_view.call_tool(
            tool_call.name,
            tool_call.arguments,
        )

        return self._serialize_tool_result(
            result
        )

    async def _execute_sleep(
        self,
        arguments: dict[str, Any],
        child_handles: dict[
            str,
            SubagentHandle,
        ],
    ) -> str:
        until_subagent = arguments.get(
            "until_subagent"
        )

        if until_subagent:
            handle = child_handles.get(
                until_subagent
            )

            if handle is None:
                return (
                    f"Unknown Subagent handle: "
                    f"{until_subagent}"
                )

            try:
                result = await handle.wait()

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                return (
                    "Subagent failed: "
                    f"{exc}"
                )

            return (
                "Subagent completed.\n"
                + self._serialize_agent_result(
                    result
                )
            )

        seconds = arguments.get(
            "seconds"
        )

        if seconds is None:
            return (
                "sleep requires either "
                "'seconds' or 'until_subagent'."
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
        child_handles: dict[
            str,
            SubagentHandle,
        ],
    ) -> str:
        task = arguments.get("task")

        if not isinstance(task, str) or not task.strip():
            return (
                "dispatch_subagent requires "
                "a non-empty 'task'."
            )

        skill_name = arguments.get(
            "skill"
        )

        try:
            selected_skill = (
                self._resolve_skill(
                    skill_name
                )
            )
        except Exception as exc:
            return str(exc)

        async def worker(
            child_context: AgentContext,
        ) -> AgentResult:
            return await self._run_agent(
                context=child_context,
                user_input=task,
                skill=selected_skill,
                history=[],
            )

        try:
            handle = self.agent_runtime.dispatch(
                context,
                task=task,
                skill=selected_skill,
                worker=worker,
            )

        except SubagentLimitError as exc:
            return str(exc)

        handle_id = handle.agent_hash

        child_handles[
            handle_id
        ] = handle

        self._subagents[
            handle_id
        ] = handle

        return (
            "Subagent dispatched.\n"
            f"handle: {handle_id}\n"
            f"depth: {handle.depth}\n"
            "It is running in parallel."
        )

    def _resolve_skill(
        self,
        name: str | None,
    ) -> Skill:
        if name is None or name == "core":
            return self.core_skill

        try:
            return self.skills.activate(
                name
            )
        except KeyError as exc:
            available = ", ".join(
                self.skills.names()
            )

            raise ValueError(
                f"Unknown Skill '{name}'. "
                f"Available Skills: {available}"
            ) from exc

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

    @staticmethod
    def _serialize_agent_result(
        result: AgentResult,
    ) -> str:
        return result.content
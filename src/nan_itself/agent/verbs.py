"""
Agent verbs: the cognitive actions an agent can take.

One registry, one protocol:

    name            literal tool name
    definition()    JSON schema handed to the model
    execute(...)    performs the action, may mutate
                    the execution state

Every agent sees every verb. Tools and skills are NOT exposed to
the model directly: they are reached only through these verbs,
and their content therefore arrives as tool results.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import mcp.types as mcp_types

from .model import (
    ChildSubagent,
)
from .runtime import (
    SubagentLimitError,
)
from ..events import StreamSink
from ..skills import (
    UnknownSkillError,
)
from ..utils.llm import (
    ToolDefinition,
)


SLEEP_TOOL_NAME = "sleep"

SPAWN_TOOL_NAME = "spawn"

LIST_TOOLS_TOOL_NAME = "list_tools"

SHOW_TOOL_TOOL_NAME = "show_tool"

INVOKE_TOOL_TOOL_NAME = "invoke_tool"

LIST_SKILLS_TOOL_NAME = "list_skills"

SHOW_SKILL_TOOL_NAME = "show_skill"

INVOKE_SKILL_TOOL_NAME = "invoke_skill"

FINISH_TOOL_NAME = "finish"


class ExecutionState:
    """
    Per-execution mutable state a verb may read or change.

    One instance per StepEngine.execute call; never shared
    across agents.
    """

    def __init__(
        self,
        *,
        persona: str,
        sink: StreamSink | None = None,
    ) -> None:
        self.persona = persona

        self.sink = sink

        self.children: list[ChildSubagent] = []

        # Set by FinishVerb; the subagent loop checks it.
        self.finished = False

        self.report: str | None = None


def serialize_tool_result(
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

    if isinstance(
        result,
        str,
    ):
        return result

    try:
        return json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

    except Exception:
        return str(result)


class SleepVerb:
    name: ClassVar[str] = SLEEP_TOOL_NAME

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Wait for a while before your next step."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "number",
                        "description": (
                            "How long to wait."
                        ),
                    },
                },
                "required": ["seconds"],
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        *,
        call,
        context,
        state,
        engine,
    ) -> str:
        seconds = call.arguments.get("seconds")

        if seconds is None:
            return (
                "sleep requires 'seconds'."
            )

        if not isinstance(
            seconds,
            (int, float),
        ):
            return "'seconds' must be a number."

        if seconds < 0:
            return "'seconds' must be >= 0."

        waited = float(seconds)

        await engine.agent_runtime.sleep(
            waited
        )

        return (
            f"Waited {waited:g} seconds."
        )


class SpawnVerb:
    name: ClassVar[str] = SPAWN_TOOL_NAME

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Spawn a parallel Subagent to work "
                "on a task. Spawn several in the same "
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
                "required": ["task"],
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        *,
        call,
        context,
        state,
        engine,
    ) -> str:
        task = call.arguments.get("task")

        if not isinstance(task, str) or not task.strip():
            return (
                "spawn requires "
                "a non-empty 'task'."
            )

        async def worker(
            child_context,
        ):
            # Subagent loop: identical to the main agent cycle
            # (obs -> model call -> result), but only the finish
            # tool ends it. History follows the same retention
            # policy as the main agent -- the engine maintains
            # it in place, clearing over the char limit.
            history: list = []

            while True:
                result = await engine.execute(
                    context=child_context,
                    persona=state.persona,
                    history=history,
                )

                if result.finished:
                    return result

        try:
            handle = engine.agent_runtime.dispatch(
                context,
                task=task,
                worker=worker,
            )

        except SubagentLimitError as exc:
            return str(exc)

        child = ChildSubagent(
            id=handle.agent_hash[:8],
            task=task,
            handle=handle,
        )

        state.children.append(
            child
        )

        if state.sink is not None:
            record_id = state.sink.record_started(
                kind="agent",
                name=task,
            )

            state.sink.record_detail(
                record_id,
                f"id: {child.id}",
            )

            state.sink.record_detail(
                record_id,
                f"depth: {handle.depth}",
            )

            state.sink.record_done(
                record_id,
                summary="spawned",
            )

        return (
            "Subagent spawned.\n"
            f"id: {child.id}\n"
            f"depth: {handle.depth}\n"
            "It runs in parallel; its report will be "
            "delivered automatically."
        )


class ListToolsVerb:
    name: ClassVar[str] = LIST_TOOLS_TOOL_NAME

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "List the names of every available tool. "
                "Tools are addressed as 'provider/tool'. "
                "Use show_tool to inspect one and "
                "invoke_tool to call one."
            ),
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        *,
        call,
        context,
        state,
        engine,
    ) -> str:
        try:
            pairs = engine.tools.list_all_tools()

        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

        if not pairs:
            return "No tools are available."

        return "\n".join(
            f"{provider_name}/{tool.name}"
            for provider_name, tool in pairs
        )


class ShowToolVerb:
    name: ClassVar[str] = SHOW_TOOL_TOOL_NAME

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Show the JSON definition of one tool "
                "('provider/tool'): its description and "
                "input schema."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Tool name as 'provider/tool'."
                        ),
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        *,
        call,
        context,
        state,
        engine,
    ) -> str:
        name = call.arguments.get("name")

        if not isinstance(name, str) or not name.strip():
            return (
                "show_tool requires 'name' "
                "(format 'provider/tool')."
            )

        try:
            resolved = (
                await engine.tools.resolve_tool(
                    name
                )
            )

        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

        if resolved is None:
            return (
                f"Unknown tool '{name}'. "
                "Use list_tools to see available tools."
            )

        provider, tool = resolved

        definition = {
            "name": (
                f"{provider.spec.name}/{tool.name}"
            ),
            "description": (
                tool.description or ""
            ),
            "inputSchema": tool.inputSchema,
        }

        return json.dumps(
            definition,
            ensure_ascii=False,
            indent=2,
        )


class InvokeToolVerb:
    name: ClassVar[str] = INVOKE_TOOL_TOOL_NAME

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Invoke one tool ('provider/tool') with "
                "arguments matching its input schema. "
                "Use show_tool first when the schema is "
                "unknown."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Tool name as 'provider/tool'."
                        ),
                    },
                    "arguments": {
                        "type": "object",
                        "description": (
                            "Arguments for the tool."
                        ),
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        *,
        call,
        context,
        state,
        engine,
    ) -> str:
        name = call.arguments.get("name")

        if not isinstance(name, str) or not name.strip():
            return (
                "invoke_tool requires 'name' "
                "(format 'provider/tool')."
            )

        arguments = (
            call.arguments.get("arguments")
            or {}
        )

        if not isinstance(
            arguments,
            dict,
        ):
            return "'arguments' must be an object."

        try:
            resolved = (
                await engine.tools.resolve_tool(
                    name
                )
            )

            if resolved is None:
                return (
                    f"Unknown tool '{name}'. "
                    "Use list_tools to see available tools."
                )

            provider, tool = resolved

            result = (
                await engine.tools.call_tool(
                    provider.spec.name,
                    tool.name,
                    arguments,
                )
            )

        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

        return serialize_tool_result(
            result
        )


class ListSkillsVerb:
    name: ClassVar[str] = LIST_SKILLS_TOOL_NAME

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "List every available Skill with its "
                "description. Use show_skill to inspect "
                "one and invoke_skill to read its "
                "resources or run its scripts."
            ),
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        *,
        call,
        context,
        state,
        engine,
    ) -> str:
        catalog = engine.skills.catalog()

        if not catalog:
            return "No skills are available."

        return "\n".join(
            f"{metadata.name}: {metadata.description}"
            for metadata in catalog
        )


class ShowSkillVerb:
    name: ClassVar[str] = SHOW_SKILL_TOOL_NAME

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Show one Skill's metadata, description "
                "and the directory of resources it ships "
                "(scripts / references / assets)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "The Skill name."
                        ),
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        *,
        call,
        context,
        state,
        engine,
    ) -> str:
        name = call.arguments.get("name")

        if not isinstance(name, str) or not name.strip():
            return "show_skill requires 'name'."

        metadata = engine.skills.get_metadata(
            name
        )

        if metadata is None:
            return (
                f"Unknown Skill '{name}'. "
                "Use list_skills to see available skills."
            )

        lines = [
            f"name: {metadata.name}",
            f"description: {metadata.description}",
            f"source: {metadata.source}",
        ]

        try:
            resources = (
                engine.skills.resource_paths(
                    name
                )
            )

        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

        for group in (
            "scripts",
            "references",
            "assets",
        ):
            paths = resources.get(
                group,
                (),
            )

            if paths:
                lines.append(
                    f"{group}:"
                )

                lines.extend(
                    f"  - {path}"
                    for path in paths
                )

        return "\n".join(
            lines
        )


class InvokeSkillVerb:
    name: ClassVar[str] = INVOKE_SKILL_TOOL_NAME

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Invoke one Skill resource. Paths under "
                "scripts/ are executed with the given "
                "arguments; any other resource is "
                "returned as text."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "The Skill name."
                        ),
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Resource path relative to "
                            "the Skill root, e.g. "
                            "'scripts/run.py' or "
                            "'references/guide.md'."
                        ),
                    },
                    "args": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": (
                            "Arguments for a script."
                        ),
                    },
                },
                "required": ["name", "path"],
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        *,
        call,
        context,
        state,
        engine,
    ) -> str:
        name = call.arguments.get("name")

        path = call.arguments.get("path")

        args = (
            call.arguments.get("args")
            or []
        )

        if not isinstance(name, str) or not name.strip():
            return "invoke_skill requires 'name'."

        if not isinstance(path, str) or not path.strip():
            return "invoke_skill requires 'path'."

        if not isinstance(
            args,
            list,
        ) or any(
            not isinstance(
                item,
                str,
            )
            for item in args
        ):
            return "'args' must be a list of strings."

        try:
            result = (
                await engine.skills.invoke(
                    name,
                    path,
                    args,
                )
            )

        except UnknownSkillError:
            return (
                f"Unknown Skill '{name}'. "
                "Use list_skills to see available skills."
            )

        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

        return result


class FinishVerb:
    """
    Subagent-only tool: submit the final report and end the
    task. The main agent never gets this definition, so a
    depth-0 call is rejected defensively.
    """

    name: ClassVar[str] = FINISH_TOOL_NAME

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Submit your final report and end this "
                "subagent task. Only this tool ends the task: "
                "a plain-text reply keeps the task running. "
                "Call it exactly once, when the task is fully "
                "complete."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "report": {
                        "type": "string",
                        "description": (
                            "The final report of the task: "
                            "outcome, findings and anything "
                            "the spawner needs."
                        ),
                    },
                },
                "required": ["report"],
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        *,
        call,
        context,
        state,
        engine,
    ) -> str:
        if context.depth == 0:
            return (
                "finish is only available to subagents."
            )

        report = call.arguments.get("report")

        if not isinstance(report, str) or not report.strip():
            return (
                "finish requires "
                "a non-empty 'report'."
            )

        state.finished = True

        state.report = report

        return "Report submitted; task finished."


VERBS: dict[str, Any] = {
    SleepVerb.name: SleepVerb(),
    SpawnVerb.name: SpawnVerb(),
    ListToolsVerb.name: ListToolsVerb(),
    ShowToolVerb.name: ShowToolVerb(),
    InvokeToolVerb.name: InvokeToolVerb(),
    ListSkillsVerb.name: ListSkillsVerb(),
    ShowSkillVerb.name: ShowSkillVerb(),
    InvokeSkillVerb.name: InvokeSkillVerb(),
    FinishVerb.name: FinishVerb(),
}

"""
Agent verbs: the cognitive actions an agent can take.

One registry, one protocol:

    name            literal tool name
    definition()    JSON schema handed to the model
    execute(...)    performs the action against the Agent

Every agent sees every verb except finish, which is filtered out at
depth 0: only subagents may end their own loop. Tools and skills are
NOT exposed to the model directly: they are reached only through these
verbs, and their content therefore arrives as tool results.

The interface-face verb triples share one mental model -- list
enumerates, show inspects, invoke acts -- for tools, skills and
module channels alike.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, ClassVar

import mcp.types as mcp_types

from .model import (
    SubagentLimitError,
)
from ..events import sink
from ..skills import (
    UnknownSkillError,
)
from ..utils.llm import (
    ToolDefinition,
)

if TYPE_CHECKING:
    from .core import Agent


SLEEP_TOOL_NAME = "sleep"

SPAWN_TOOL_NAME = "spawn"

LIST_TOOLS_TOOL_NAME = "list_tools"

SHOW_TOOL_TOOL_NAME = "show_tool"

INVOKE_TOOL_TOOL_NAME = "invoke_tool"

LIST_SKILLS_TOOL_NAME = "list_skills"

SHOW_SKILL_TOOL_NAME = "show_skill"

INVOKE_SKILL_TOOL_NAME = "invoke_skill"

LIST_CHANNELS_TOOL_NAME = "list_channels"

SHOW_CHANNELS_TOOL_NAME = "show_channels"

INVOKE_CHANNELS_TOOL_NAME = "invoke_channels"

FINISH_TOOL_NAME = "finish"


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
        agent: "Agent",
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

        await asyncio.sleep(
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
        agent: "Agent",
    ) -> str:
        task = call.arguments.get("task")

        if not isinstance(task, str) or not task.strip():
            return (
                "spawn requires "
                "a non-empty 'task'."
            )

        try:
            child = agent.spawn(
                task,
            )

        except SubagentLimitError as exc:
            return str(exc)

        record_id = sink.emit(
            "record_started",
            content={
                "kind": "agent",
                "name": task,
                "summary": "",
                "agent_hash": agent.agent_hash,
                "parent_hash": agent.parent_hash,
                "depth": agent.depth,
            },
        )

        if record_id:
            sink.emit(
                "record_detail",
                id=record_id,
                content={
                    "line": f"id: {child.agent_hash[:8]}",
                    "agent_hash": agent.agent_hash,
                    "parent_hash": agent.parent_hash,
                    "depth": agent.depth,
                },
            )

            sink.emit(
                "record_detail",
                id=record_id,
                content={
                    "line": f"depth: {child.depth}",
                    "agent_hash": agent.agent_hash,
                    "parent_hash": agent.parent_hash,
                    "depth": agent.depth,
                },
            )

            sink.emit(
                "record_done",
                id=record_id,
                content={
                    "summary": "spawned",
                    "note": "",
                    "agent_hash": agent.agent_hash,
                    "parent_hash": agent.parent_hash,
                    "depth": agent.depth,
                },
            )

        return (
            "Subagent spawned.\n"
            f"id: {child.agent_hash[:8]}\n"
            f"depth: {child.depth}\n"
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
        agent: "Agent",
    ) -> str:
        try:
            pairs = agent.tools.list_all_tools()

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
        agent: "Agent",
    ) -> str:
        name = call.arguments.get("name")

        if not isinstance(name, str) or not name.strip():
            return (
                "show_tool requires 'name' "
                "(format 'provider/tool')."
            )

        try:
            resolved = (
                await agent.tools.resolve_tool(
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
        agent: "Agent",
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
                await agent.tools.resolve_tool(
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
                await agent.tools.call_tool(
                    provider.spec.name,
                    tool.name,
                    arguments,
                )
            )

        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

        # MCP results must be dumped structurally; plain str passes
        # through unquoted; anything else falls back to JSON with
        # str() as the escape hatch for non-serializable values.
        if isinstance(result, mcp_types.CallToolResult):
            return result.model_dump_json()

        if isinstance(result, str):
            return result

        return json.dumps(result, ensure_ascii=False, default=str)


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
        agent: "Agent",
    ) -> str:
        catalog = agent.skills.catalog()

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
        agent: "Agent",
    ) -> str:
        name = call.arguments.get("name")

        if not isinstance(name, str) or not name.strip():
            return "show_skill requires 'name'."

        metadata = agent.skills.get_metadata(
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
                agent.skills.resource_paths(
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
        agent: "Agent",
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
                await agent.skills.invoke(
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


class ListChannelsVerb:
    name: ClassVar[str] = LIST_CHANNELS_TOOL_NAME

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "List every module channel the model may "
                "feed, as 'module/channel'. Channels are "
                "downlink endpoints: use show_channels to "
                "inspect one and invoke_channels to feed a "
                "payload."
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
        agent: "Agent",
    ) -> str:
        try:
            return (
                agent.modules.list_module_channels()
            )

        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"


class ShowChannelsVerb:
    name: ClassVar[str] = SHOW_CHANNELS_TOOL_NAME

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Show channel details: description and "
                "JSON schema."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "module": {
                        "type": "string",
                        "description": (
                            "Module id, e.g. 'desktop'."
                        ),
                    },
                    "channel": {
                        "type": "string",
                        "description": (
                            "Channel name; omit to show "
                            "every channel of the module."
                        ),
                    },
                },
                "required": ["module"],
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        *,
        call,
        agent: "Agent",
    ) -> str:
        module_id = call.arguments.get("module")

        if not isinstance(module_id, str) or (
            not module_id.strip()
        ):
            return "show_channels requires 'module'."

        channel = call.arguments.get("channel")

        if channel is not None and (
            not isinstance(channel, str)
            or not channel.strip()
        ):
            return (
                "'channel' must be a string."
            )

        try:
            return agent.modules.show_module_channels(
                module_id,
                channel,
            )

        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"


class InvokeChannelsVerb:
    name: ClassVar[str] = INVOKE_CHANNELS_TOOL_NAME

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Feed one payload into a module channel: "
                "schema validation, then hand-down to the "
                "module. Returns 'written' or 'rejected'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "module": {
                        "type": "string",
                        "description": (
                            "Module id, e.g. 'desktop'."
                        ),
                    },
                    "channel": {
                        "type": "string",
                        "description": (
                            "Channel name, e.g. 'goal'."
                        ),
                    },
                    "payload": {
                        "type": "object",
                        "description": (
                            "Payload, shaped by the "
                            "channel schema (see "
                            "show_channels)."
                        ),
                    },
                },
                "required": ["module", "channel", "payload"],
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        *,
        call,
        agent: "Agent",
    ) -> str:
        module_id = call.arguments.get("module")

        if not isinstance(module_id, str) or (
            not module_id.strip()
        ):
            return "invoke_channels requires 'module'."

        channel = call.arguments.get("channel")

        if not isinstance(channel, str) or (
            not channel.strip()
        ):
            return "invoke_channels requires 'channel'."

        if "payload" not in call.arguments:
            return "invoke_channels requires 'payload'."

        payload = call.arguments["payload"]

        try:
            return agent.modules.write_module_channel(
                module_id,
                channel,
                payload,
            )

        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"


class FinishVerb:
    """
    Subagent-only tool: submit the final report and end the
    task. The root agent never gets this definition, so a
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
        agent: "Agent",
    ) -> str:
        if agent.depth == 0:
            return (
                "finish is only available to subagents."
            )

        report = call.arguments.get("report")

        if not isinstance(report, str) or not report.strip():
            return (
                "finish requires "
                "a non-empty 'report'."
            )

        agent.finish(
            report,
        )

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
    ListChannelsVerb.name: ListChannelsVerb(),
    ShowChannelsVerb.name: ShowChannelsVerb(),
    InvokeChannelsVerb.name: InvokeChannelsVerb(),
    FinishVerb.name: FinishVerb(),
}

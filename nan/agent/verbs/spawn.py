from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nan.agent.verbs import Verb, VerbOutcome
from nan.llm import ToolDefinition

if TYPE_CHECKING:
    from nan.agent.core import Agent


class SpawnVerb(Verb):
    name = "spawn"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Spawn a subagent with a task. Returns its short "
                "id: keep it, you need it for send/kill. The child "
                "runs in parallel; its finish report arrives in "
                "your inbox automatically."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task to delegate.",
                    },
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        )

    async def execute(self, *, call: Any, agent: Agent) -> VerbOutcome:
        task = call.arguments.get("task")

        if not isinstance(task, str) or not task.strip():
            return VerbOutcome("spawn requires a non-empty 'task'.")

        child, error = agent.spawn_child(task)

        if error:
            return VerbOutcome(error)

        return VerbOutcome(
            f"Subagent spawned.\nid: {child.agent_id}\ndepth: {child.depth}",
            {"child_id": child.agent_id, "depth": child.depth},
        )

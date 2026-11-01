from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nan.agent.verbs import Verb, VerbOutcome
from nan.llm import ToolDefinition

if TYPE_CHECKING:
    from nan.agent.core import Agent


class KillVerb(Verb):
    name = "kill"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Kill a child you spawned (by child_id). Its own "
                "subtree dies with it (cascade). You will receive a "
                "death notice in your inbox."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "child_id": {
                        "type": "string",
                        "description": "The child's short id.",
                    },
                },
                "required": ["child_id"],
                "additionalProperties": False,
            },
        )

    async def execute(self, *, call: Any, agent: Agent) -> VerbOutcome:
        child_id = call.arguments.get("child_id")

        child = agent.children.get(child_id) if isinstance(child_id, str) else None

        if child is None:
            held = ", ".join(agent.children) or "(none)"
            return VerbOutcome(f"Unknown child '{child_id}'. Children: {held}")

        child.request_death(f"killed by holder {agent.agent_id}")

        return VerbOutcome(
            f"Kill requested for {child.agent_id}.",
            {"target_id": child.agent_id},
        )

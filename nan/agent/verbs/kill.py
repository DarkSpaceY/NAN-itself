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
                "Kill a child you spawned (by handle_id). Its own "
                "subtree dies with it (cascade). You will receive a "
                "death notice in your inbox."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "handle_id": {
                        "type": "string",
                        "description": "The child's short id.",
                    },
                },
                "required": ["handle_id"],
                "additionalProperties": False,
            },
        )

    async def execute(self, *, call: Any, agent: Agent) -> VerbOutcome:
        handle_id = call.arguments.get("handle_id")

        handle = agent.child_handle(handle_id)

        if handle is None:
            held = ", ".join(h.agent_id for h in agent.child_handles()) or "(none)"
            return VerbOutcome(f"Unknown handle '{handle_id}'. Held children: {held}")

        handle.kill()

        return VerbOutcome(
            f"Kill requested for {handle.agent_id}.",
            {"target_id": handle.agent_id},
        )

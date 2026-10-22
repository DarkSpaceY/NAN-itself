from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nan.agent.verbs import Verb
from nan.llm import ToolDefinition

if TYPE_CHECKING:
    from nan.agent.core import Agent


class SendVerb(Verb):
    name = "send"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Send text along a handle. direction 'up' delivers to "
                "your parent's inbox (your parent handle needs no id). "
                "direction 'down' delivers into a child you spawned: "
                "pass its handle_id (the short id spawn gave you). "
                "Text only: handles themselves never travel."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "'up' to parent, 'down' to a held child.",
                    },
                    "handle_id": {
                        "type": "string",
                        "description": "Child short id; required for direction 'down'.",
                    },
                    "text": {"type": "string", "description": "The message."},
                },
                "required": ["direction", "text"],
                "additionalProperties": False,
            },
        )

    async def execute(self, *, call: Any, agent: Agent) -> str:
        direction = call.arguments.get("direction")

        text = call.arguments.get("text")

        if direction not in ("up", "down"):
            return "'direction' must be 'up' or 'down'."

        if not isinstance(text, str) or not text.strip():
            return "send requires a non-empty 'text'."

        if direction == "up":
            if agent.parent is None:
                return "No parent: the root cannot send up."

            await agent.parent.receive_up(text, agent.agent_id)

            return "Sent up."

        handle_id = call.arguments.get("handle_id")

        handle = agent.child_handle(handle_id)

        if handle is None:
            held = ", ".join(h.agent_id for h in agent.child_handles()) or "(none)"
            return f"Unknown handle '{handle_id}'. Held children: {held}"

        await handle.send(text)

        return f"Sent down to {handle.agent_id}."

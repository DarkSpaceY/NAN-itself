from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nan.agent.verbs import Verb, VerbOutcome
from nan.llm.types import VerbDefinition
from nan.turn import InboxMessage

if TYPE_CHECKING:
    from nan.agent.core import Agent


class SendVerb(Verb):
    name = "send"

    def definition(self) -> VerbDefinition:
        return VerbDefinition(
            name=self.name,
            description=(
                "Send text to another agent. direction 'up' delivers to "
                "your parent's inbox (the root has no parent). "
                "direction 'down' delivers into a child you spawned: "
                "pass its child_id. Text only: ids never travel."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "'up' to parent, 'down' to a child.",
                    },
                    "child_id": {
                        "type": "string",
                        "description": "Child short id; required for direction 'down'.",
                    },
                    "text": {"type": "string", "description": "The message."},
                },
                "required": ["direction", "text"],
                "additionalProperties": False,
            },
        )

    async def execute(self, *, call: Any, agent: Agent) -> VerbOutcome:
        direction = call.arguments.get("direction")

        text = call.arguments.get("text")

        if direction not in ("up", "down"):
            return VerbOutcome("'direction' must be 'up' or 'down'.")

        if not isinstance(text, str) or not text.strip():
            return VerbOutcome("send requires a non-empty 'text'.")

        if direction == "up":
            if agent.parent is None:
                return VerbOutcome("No parent: the root cannot send up.")

            agent.parent.deliver(InboxMessage.from_child(text, source=agent.agent_id))

            return VerbOutcome("Sent up.", {"direction": "up"})

        child_id = call.arguments.get("child_id")

        child = agent.children.get(child_id) if isinstance(child_id, str) else None

        if child is None:
            held = ", ".join(agent.children) or "(none)"
            return VerbOutcome(f"Unknown child '{child_id}'. Children: {held}")

        child.deliver(InboxMessage.from_parent(text, source=agent.agent_id))

        return VerbOutcome(
            f"Sent down to {child.agent_id}.",
            {"direction": "down", "to": child.agent_id},
        )

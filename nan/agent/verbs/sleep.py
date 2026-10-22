from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from nan.agent.verbs import Verb
from nan.llm import ToolDefinition

if TYPE_CHECKING:
    from nan.agent.core import Agent


class SleepVerb(Verb):
    name = "sleep"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Wait for a while before your next step.",
            input_schema={
                "type": "object",
                "properties": {
                    "seconds": {"type": "number", "description": "How long to wait."},
                },
                "required": ["seconds"],
                "additionalProperties": False,
            },
        )

    async def execute(self, *, call: Any, agent: Agent) -> str:
        seconds = call.arguments.get("seconds")

        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
            return "'seconds' must be a number."

        if seconds < 0:
            return "'seconds' must be >= 0."

        await asyncio.sleep(float(seconds))

        return f"Waited {float(seconds):g} seconds."

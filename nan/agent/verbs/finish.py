from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nan.agent.verbs import Verb, VerbOutcome
from nan.llm import ToolDefinition

if TYPE_CHECKING:
    from nan.agent.core import Agent


class FinishVerb(Verb):
    name = "finish"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Submit your final report and end your task. Only "
                "this ends the task: a plain reply keeps you running. "
                "The report is delivered to your parent automatically. "
                "Call it exactly once, when the task is complete."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "report": {
                        "type": "string",
                        "description": "Outcome, findings, and anything the spawner needs.",
                    },
                },
                "required": ["report"],
                "additionalProperties": False,
            },
        )

    async def execute(self, *, call: Any, agent: Agent) -> VerbOutcome:
        if agent.depth == 0:
            return VerbOutcome("finish is only available to subagents.")

        report = call.arguments.get("report")

        if not isinstance(report, str) or not report.strip():
            return VerbOutcome("finish requires a non-empty 'report'.")

        agent.finish(report)

        # The report itself travels in agent_exited.
        return VerbOutcome("Report submitted upstream; task finished.")

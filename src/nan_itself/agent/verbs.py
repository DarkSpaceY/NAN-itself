"""
Agent verbs: the cognitive actions an agent can take.

One registry, one protocol:

    name                      literal tool name
    definition()              JSON schema handed to the model
    visible(context, policy)  role gate (RolePolicy decides)
    execute(...)              performs the action, may mutate
                              the execution state

The Main Agent simply never sees verbs whose visibility fails —
Skills do not exist in its world.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from .model import (
    ChildSubagent,
)
from .runtime import (
    SubagentLimitError,
)
from ..skills import (
    UnknownSkillError,
)
from ..utils.llm import (
    ToolDefinition,
)

if TYPE_CHECKING:
    from .role import (
        RolePolicy,
    )


SLEEP_TOOL_NAME = "sleep"

DISPATCH_SUBAGENT_TOOL_NAME = "dispatch_subagent"

ACTIVATE_SKILL_TOOL_NAME = "activate_skill"


class ExecutionState:
    """
    Per-execution mutable state a verb may read or change.

    One instance per StepEngine.execute call; never shared
    across agents.
    """

    def __init__(
        self,
        *,
        active_skill: Any | None,
        persona: str,
    ) -> None:
        self.active_skill = active_skill
        self.persona = persona

        self.children: list[ChildSubagent] = []


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

    def visible(
        self,
        depth: int,
        policy: "RolePolicy",
    ) -> bool:
        return True

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

        waited, interrupted = (
            await engine.agent_runtime.sleep(
                float(seconds)
            )
        )

        if interrupted:
            return (
                f"Sleep interrupted after {waited:.1f}s: "
                "new input arrived. End your turn now so it "
                "can be processed."
            )

        return (
            f"Waited {float(seconds):g} seconds."
        )


class DispatchVerb:
    name: ClassVar[str] = (
        DISPATCH_SUBAGENT_TOOL_NAME
    )

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Dispatch a parallel Subagent to work "
                "on a task. Dispatch several in the same "
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

    def visible(
        self,
        depth: int,
        policy: "RolePolicy",
    ) -> bool:
        return True

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
                "dispatch_subagent requires "
                "a non-empty 'task'."
            )

        async def worker(
            child_context,
        ):
            # The child inherits the CURRENT execution Skill.
            #
            # AgentContext itself is immutable, while the active
            # Skill is mutable execution state. Therefore the current
            # state.active_skill must be passed explicitly.
            return await engine.execute(
                context=child_context,
                user_input=task,
                persona=state.persona,
            )

        try:
            handle = engine.agent_runtime.dispatch(
                context,
                task=task,
                worker=worker,
                skill=state.active_skill,
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

        return (
            "Subagent dispatched.\n"
            f"id: {child.id}\n"
            f"depth: {handle.depth}\n"
            "It runs in parallel; its report will be "
            "delivered automatically."
        )


class ActivateSkillVerb:
    name: ClassVar[str] = (
        ACTIVATE_SKILL_TOOL_NAME
    )

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Switch this Subagent's own Skill. The new "
                "Skill takes effect from the next step."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "The name of the Skill to "
                            "activate."
                        ),
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        )

    def visible(
        self,
        depth: int,
        policy: "RolePolicy",
    ) -> bool:
        return policy.may_switch_skill(
            depth
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
                "activate_skill requires 'name'. "
                f"Available Skills: "
                f"{', '.join(engine.skills.names())}"
            )

        try:
            skill = engine.skills.activate(
                name
            )

        except UnknownSkillError:
            return (
                f"Unknown Skill '{name}'. "
                f"Available Skills: "
                f"{', '.join(engine.skills.names())}"
            )

        state.active_skill = skill

        return (
            f"Skill '{skill.name}' activated. It takes "
            "effect from your next step."
        )


VERBS: dict[str, Any] = {
    SleepVerb.name: SleepVerb(),
    DispatchVerb.name: DispatchVerb(),
    ActivateSkillVerb.name: ActivateSkillVerb(),
}
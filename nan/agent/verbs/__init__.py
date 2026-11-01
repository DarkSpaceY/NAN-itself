"""
Agent verbs: the six cognitive actions.

    spawn    create a child agent
    sleep    block for a while
    shell    run a command in the workspace
    send     send text message up to the parent, or down into a child
    kill     kill a child (cascade is automatic)
    finish   report upstream and die

The verb set IS the agent's capability: a root runs ROOT_VERBS
(no finish -- the root never exits), a child runs CHILD_VERBS.

Tools and skills are plain files the agent reaches through
shell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from nan.agent.core import Agent
    from nan.llm import ToolDefinition


@dataclass(frozen=True)
class VerbOutcome:
    """
    What a verb returns: the text the model receives, plus
    structured domain facts for observers (child id, exit code,
    ...).
    """

    text: str
    detail: dict[str, Any] = field(default_factory=dict)


class Verb:
    """
    One cognitive action. `name` is the tool name the model
    sees; `definition()` is the protocol it follows;
    `execute()` does the work.
    """

    name: ClassVar[str]

    def definition(self) -> ToolDefinition:
        raise NotImplementedError

    async def execute(self, *, call: Any, agent: Agent) -> VerbOutcome:
        raise NotImplementedError


def _build_registry() -> dict[str, Verb]:
    from .finish import FinishVerb
    from .kill import KillVerb
    from .send import SendVerb
    from .shell import ShellVerb
    from .sleep import SleepVerb
    from .spawn import SpawnVerb

    return {
        SpawnVerb.name: SpawnVerb(),
        SleepVerb.name: SleepVerb(),
        ShellVerb.name: ShellVerb(),
        SendVerb.name: SendVerb(),
        KillVerb.name: KillVerb(),
        FinishVerb.name: FinishVerb(),
    }


def truncate_output(output: str, returncode: int | None, limit: int) -> str:
    """Head+tail truncation for shell output, exit code first."""
    output = output.rstrip("\n")

    if len(output) > limit:
        head = output[: limit // 2]

        tail = output[-(limit // 2) :]

        output = f"{head}\n...[truncated {len(output) - limit} chars]...\n{tail}"

    return f"exit code: {returncode}\n{output}"


VERBS: dict[str, Verb] = _build_registry()

# A root never exits, so it cannot finish; a child can.
CHILD_VERBS: dict[str, Verb] = dict(VERBS)

ROOT_VERBS: dict[str, Verb] = {name: verb for name, verb in VERBS.items() if name != "finish"}


def tool_definitions(verbs: dict[str, Verb]) -> list[ToolDefinition]:
    """The verb protocol handed to the model: exactly `verbs`."""
    return [verb.definition() for verb in verbs.values()]

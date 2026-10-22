"""
Agent verbs: the six cognitive actions.

    spawn    create a child agent, hold its handle
    sleep    block for a while
    shell    run a command in the workspace
    send     send text message up to the parent, or down into a held child
    kill     kill a held child (cascade is automatic)
    finish   subagent-only: report upstream and die

Tools and skills are plain files the agent reaches through
shell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from nan.agent.core import Agent
    from nan.llm import ToolDefinition

# Shell output truncation: keep head+tail around this size
# (the default; the effective value is agent.config.shell_output_limit).
SHELL_OUTPUT_LIMIT = 20_000


@dataclass(frozen=True)
class VerbOutcome:
    """
    What a verb returns: the text the model receives, plus
    structured domain facts for observers (child id, exit code,
    ...) so no one has to parse the prose.
    """

    text: str
    detail: dict[str, Any] = field(default_factory=dict)


class Verb:
    """
    One cognitive action. `name` is the tool name the model
    sees; `definition()` is the authoritative protocol (the
    system prompt never repeats it); `execute()` does the work.
    """

    name: ClassVar[str]

    def definition(self) -> ToolDefinition:
        raise NotImplementedError

    async def execute(self, *, call: Any, agent: Agent) -> VerbOutcome:
        raise NotImplementedError


def _build_registry() -> dict[str, Verb]:
    # Imported inside the function: the base class above must
    # exist before the verb modules subclass it (they import it
    # from this partially-initialized package).
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


def truncate_output(output: str, returncode: int | None, limit: int = SHELL_OUTPUT_LIMIT) -> str:
    """Head+tail truncation for shell output, exit code first."""
    output = output.rstrip("\n")

    if len(output) > limit:
        head = output[: limit // 2]

        tail = output[-(limit // 2) :]

        output = f"{head}\n...[truncated {len(output) - limit} chars]...\n{tail}"

    return f"exit code: {returncode}\n{output}"


VERBS: dict[str, Verb] = _build_registry()


def tool_definitions(*, depth: int) -> list[ToolDefinition]:
    """
    The verb list handed to the model. finish exists only below
    the root: the root has no one to report to.
    """
    verbs = [v for v in VERBS.values() if not (v.name == "finish" and depth == 0)]

    return [v.definition() for v in verbs]

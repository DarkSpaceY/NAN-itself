"""
Data contracts of the agent package.

Pure value types shared across the engine, verbs and the main
turn orchestrator. Imports nothing from sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runtime import (
        SubagentHandle,
    )


@dataclass(frozen=True)
class AgentResult:
    """
    Result of one completed Agent execution.

    `content` is the model's plain-text reply, or the finish
    report when the model ended a subagent task via finish;
    None when the turn ended with tool calls (the next turn
    continues from them).

    `finished` marks an explicit finish-tool completion: only a
    subagent loop terminates on it (plain text alone never does,
    so the main agent is unaffected).

    `turn` is the completed Turn record of this execution: the
    sole carrier of identity, snapshots (persona / history),
    messages and response metadata. Callers chain turns by
    passing it back as `last_turn` -- there is no persistent
    history outside the Turn chain.
    """

    content: str | None

    turn: Any | None = None

    finished: bool = False


@dataclass
class ChildSubagent:
    """
    Execution-local record of one dispatched child.

    Handles are never exposed to the model. Reports are delivered
    automatically at step boundaries or into the next turn when
    they arrive late.
    """

    id: str
    task: str
    handle: "SubagentHandle"
    reported: bool = False

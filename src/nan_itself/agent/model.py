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

    `content` is the model's plain-text reply; None when the turn
    ended with tool calls (the next turn continues from them).
    """

    content: str | None

    messages: tuple[Any, ...]

    response: Any


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

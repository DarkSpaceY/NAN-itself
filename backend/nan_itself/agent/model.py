"""
Data contracts of the agent package.

Pure value types shared across the agent's turn orchestrator and
the verbs. Imports nothing from sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass


class SubagentLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class Report:
    """
    One finished subagent's report, as harvested by its parent.

    Structured end to end: the child fills it on exit (before
    `done` is flipped), the parent harvests it onto its next
    turn (as `turn.reports`) and the engine renders it into the
    observation message (wrapped in <subagent_report>). The UI
    mirror reads the fields directly; no prompt text is ever
    parsed back.
    """

    agent_id: str

    task: str | None

    status: str

    body: str

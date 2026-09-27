"""
Data contracts of the agent package.

Pure value types shared across the agent's turn orchestrator and
the verbs. Imports nothing from sibling modules.
"""

from __future__ import annotations


class SubagentLimitError(RuntimeError):
    pass

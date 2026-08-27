"""
Role policy: the single source of truth for main/sub differences.

Before this module existed, the same `depth > 0` question was
answered in five different places. Every role-dependent decision
must go through here.
"""

from __future__ import annotations


class RolePolicy:
    """
    Static predicates over an agent's dispatch depth.

    Main agent  -> depth == 0
    Subagent    -> depth > 0
    """

    @staticmethod
    def may_switch_skill(depth: int) -> bool:
        """Only subagents may activate_skill."""
        return depth > 0

    @staticmethod
    def injects_skills_section(depth: int) -> bool:
        """Only subagents see the <skills> section."""
        return depth > 0

    @staticmethod
    def archives_orphan_reports(depth: int) -> bool:
        """
        Orphaned children of the MAIN agent are parked for the
        next turn; orphans deeper in a dead subtree are dropped
        (nobody left with context to digest them).
        """
        return depth == 0

    @staticmethod
    def hidden_verb_reply(verb: str) -> str:
        """Reply when a role hallucinates a verb it cannot use."""
        return f"{verb} is only available to Subagents."

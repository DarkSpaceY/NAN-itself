"""
The goal tree: state and rendering, standard library only.

Goals are outcomes, not steps: a node says what will be true when it
is done, never how to get there. Children decompose a goal into
smaller outcomes; the agent stays free to choose the method.

The tree is stored as JSON on disk. The CLI is its only writer and
the service reads it fresh on every turn, so a down service never
blocks planning.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ACTIVE = "active"
DONE = "done"
DROPPED = "dropped"

STATUSES = (ACTIVE, DONE, DROPPED)


@dataclass
class Goal:
    id: str
    text: str
    parent: str | None = None
    status: str = ACTIVE


def load(path: Path) -> list[Goal]:
    """Read the tree; a missing or broken file is an empty plan."""
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))

    except (OSError, json.JSONDecodeError):
        return []

    raw = data.get("goals") if isinstance(data, dict) else None

    if not isinstance(raw, list):
        return []

    goals = []

    for item in raw:
        if not isinstance(item, dict):
            continue

        goal_id = item.get("id")
        text = item.get("text")

        if not isinstance(goal_id, str) or not isinstance(text, str):
            continue

        parent = item.get("parent")

        status = item.get("status")

        goals.append(
            Goal(
                id=goal_id,
                text=text,
                parent=parent if isinstance(parent, str) else None,
                status=status if status in STATUSES else ACTIVE,
            )
        )

    return goals


def save(path: Path, goals: list[Goal]) -> None:
    """Write the tree atomically: a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    body = json.dumps({"goals": [asdict(goal) for goal in goals]}, ensure_ascii=False)

    tmp = path.with_name(path.name + ".tmp")

    tmp.write_text(body, encoding="utf-8")

    os.replace(tmp, path)


def add(goals: list[Goal], text: str, parent: str | None = None) -> Goal:
    text = text.strip()

    if not text:
        raise ValueError("goal text is empty")

    if parent is not None and not any(goal.id == parent for goal in goals):
        raise KeyError(parent)

    goal = Goal(id=_next_id(goals), text=text, parent=parent)

    goals.append(goal)

    return goal


def close(goals: list[Goal], goal_id: str, status: str) -> tuple[Goal, int]:
    """Close a goal and every still-active descendant.

    Returns the goal and how many descendants were closed with it.
    Without the cascade a child under a closed parent would stay
    active but invisible in the tree.
    """
    if status not in (DONE, DROPPED):
        raise ValueError(f"cannot close with status {status}")

    goal = next((goal for goal in goals if goal.id == goal_id), None)

    if goal is None:
        raise KeyError(goal_id)

    closed = 0

    if goal.status == ACTIVE:
        goal.status = status

    for descendant in _descendants(goals, goal_id):
        if descendant.status == ACTIVE:
            descendant.status = status

            closed += 1

    return goal, closed


def render_full(goals: list[Goal]) -> str:
    """The whole tree, indented by depth, in insertion order."""
    lines: list[str] = []

    for goal in _roots(goals):
        _render_branch(goals, goal, 0, lines)

    return "\n".join(lines)


def render_tree(goals: list[Goal]) -> str:
    """The same tree for the terminal: ids and statuses, no bullets."""
    lines: list[str] = []

    for goal in _roots(goals):
        _render_branch(goals, goal, 0, lines, bullet=False)

    return "\n".join(lines)


def snapshot(goals: list[Goal]) -> dict[str, str]:
    return {goal.id: goal.status for goal in goals}


def render_delta(seen: dict[str, str], goals: list[Goal]) -> str:
    """Only what changed since the agent last saw the tree."""
    lines: list[str] = []

    for goal in goals:
        previous = seen.get(goal.id)

        if previous is None:
            parent = f" (under {goal.parent})" if goal.parent else ""

            lines.append(f"- {goal.id} new{parent}: {goal.text}")

        elif previous != goal.status:
            lines.append(f"- {goal.id} {goal.status}")

    current = {goal.id for goal in goals}

    for goal_id in sorted(seen, key=_id_number):
        if goal_id not in current:
            lines.append(f"- {goal_id} removed")

    return "\n".join(lines)


def _next_id(goals: list[Goal]) -> str:
    numbers = [_id_number(goal.id) for goal in goals]

    return f"g{max(numbers, default=0) + 1}"


def _id_number(goal_id: str) -> int:
    try:
        return int(goal_id[1:])

    except ValueError:
        return 0


def _roots(goals: list[Goal]) -> list[Goal]:
    ids = {goal.id for goal in goals}

    return [goal for goal in goals if goal.parent is None or goal.parent not in ids]


def _children(goals: list[Goal], parent: str) -> list[Goal]:
    return [goal for goal in goals if goal.parent == parent]


def _descendants(goals: list[Goal], goal_id: str) -> list[Goal]:
    found: list[Goal] = []

    queue = [goal_id]

    while queue:
        for child in _children(goals, queue.pop(0)):
            found.append(child)

            queue.append(child.id)

    return found


def _render_branch(
    goals: list[Goal],
    goal: Goal,
    depth: int,
    lines: list[str],
    *,
    bullet: bool = True,
) -> None:
    indent = "  " * depth

    prefix = "- " if bullet else ""

    lines.append(f"{indent}{prefix}{goal.id} {goal.status}: {goal.text}")

    for child in _children(goals, goal.id):
        _render_branch(goals, child, depth + 1, lines, bullet=bullet)

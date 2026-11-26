"""
`plan`: read and edit the agent's goal tree.

The tree lives in services/plan/data/plan.json next to this file; the
service reads it on every turn and injects what changed. Goals are
outcomes, not steps -- write what will be true, not what to do.

Exit codes: 0 success, 2 usage error, 3 unknown goal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plan as tree


class UsageError(Exception):
    """Bad arguments; reported as exit 2 instead of argparse's exit 2 loop."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise UsageError(message)


def _path() -> Path:
    override = os.environ.get("NAN_PLAN_FILE")

    if override:
        return Path(override)

    return Path(__file__).resolve().parent / "data" / "plan.json"


def _fail(code: str, message: str, status: int) -> int:
    print(json.dumps({"error": {"code": code, "message": message}}), file=sys.stderr)

    return status


def _show(goals: list[tree.Goal]) -> None:
    if not goals:
        print("no goals")

        return

    print(tree.render_tree(goals))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    parser = _Parser(prog="plan", description="Show and edit the goal tree.")

    sub = parser.add_subparsers(dest="command")

    add_parser = sub.add_parser("add", help="Add a goal.")

    add_parser.add_argument("text", help="What will be true when this is done.")

    add_parser.add_argument("--under", metavar="ID", help="Parent goal.")

    for name, help_text in (
        ("done", "Mark a goal and its open children done."),
        ("drop", "Abandon a goal and its open children."),
    ):
        close_parser = sub.add_parser(name, help=help_text)

        close_parser.add_argument("id")

    try:
        args = parser.parse_args(argv)

    except UsageError as exc:
        parser.print_usage(sys.stderr)

        return _fail("usage", str(exc), 2)

    path = _path()

    goals = tree.load(path)

    if args.command is None:
        _show(goals)

        return 0

    if args.command == "add":
        try:
            goal = tree.add(goals, args.text, args.under)

        except ValueError as exc:
            return _fail("invalid_goal", str(exc), 2)

        except KeyError as exc:
            return _fail("unknown_goal", f"no goal with id {exc.args[0]}", 3)

        tree.save(path, goals)

        print(f"add {goal.id}")

        return 0

    status = tree.DONE if args.command == "done" else tree.DROPPED

    try:
        goal, closed = tree.close(goals, args.id, status)

    except KeyError:
        return _fail("unknown_goal", f"no goal with id {args.id}", 3)

    tree.save(path, goals)

    suffix = f" ({closed} children closed)" if closed else ""

    print(f"{args.command} {goal.id}{suffix}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

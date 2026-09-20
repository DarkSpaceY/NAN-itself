#!/usr/bin/env python3
"""Static checker for NAN-itself Module files.

Usage:
    check_module.py <module.py> [<module.py> ...]

Checks the loadable-file contract enforced by the module loader:
  - the literal `# @module` header within the first 20 lines
  - exactly one concrete Module subclass (base named Module or ActionSurface)
  - a non-empty `id` assignment on the class
  - `async def start(self)` and `async def query(self, turn)` present
  - `requires` (if present) is a tuple/list of unique identifier strings

Pure AST + text checks; the file is never imported or executed.
Exits 0 when every file passes, 1 otherwise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

HEADER = "# @module"
SCAN_LINES = 20
BASES = {"Module", "ActionSurface"}


class Report:

    def __init__(self, path: Path) -> None:
        self.path = path
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def dump(self) -> bool:
        for warning in self.warnings:
            print(f"warn: {self.path}: {warning}")

        for error in self.errors:
            print(f"FAIL: {self.path}: {error}")

        return not self.errors


def check(path: Path) -> Report:
    report = Report(path)

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        report.error(f"cannot read: {exc}")
        return report

    head = "\n".join(text.splitlines()[:SCAN_LINES])

    if HEADER not in head:
        report.error(
            f"missing '{HEADER}' header within the first {SCAN_LINES} lines"
        )

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        report.error(f"syntax error: {exc}")
        return report

    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]

    subclasses = [
        node
        for node in classes
        if any(
            (
                isinstance(base, ast.Name) and base.id in BASES
            )
            or (
                isinstance(base, ast.Attribute)
                and base.attr in BASES
            )
            for base in node.bases
        )
    ]

    if len(subclasses) != 1:
        report.error(
            f"expected exactly one Module subclass, found {len(subclasses)}"
        )

        return report

    cls = subclasses[0]

    assignments = {
        target.id: node.value
        for node in cls.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    ann_assignments = {
        node.target.id: node.value
        for node in cls.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    }

    assignments |= ann_assignments

    module_id = assignments.get("id")

    if module_id is None:
        report.error("class does not assign `id`")
    elif not (
        isinstance(module_id, ast.Constant)
        and isinstance(module_id.value, str)
        and module_id.value.strip()
    ):
        report.error("`id` must be a non-empty string literal")
    elif not module_id.value.replace("-", "_").isidentifier():
        report.warn(f"`id` is unconventional: {module_id.value!r}")

    requires = assignments.get("requires")

    if requires is not None:
        names = []

        if isinstance(requires, (ast.Tuple, ast.List)):
            for element in requires.elts:
                if isinstance(element, ast.Constant) and isinstance(
                    element.value, str
                ):
                    names.append(element.value)
                else:
                    report.error("`requires` entries must be string literals")

        duplicates = {
            name for name in names if names.count(name) > 1
        }

        if duplicates:
            report.error(f"duplicate requires entries: {sorted(duplicates)}")

    methods = {
        node.name: node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for required in ("start", "query"):
        method = methods.get(required)

        if method is None:
            report.error(f"missing `{required}()`")
        elif not isinstance(method, ast.AsyncFunctionDef):
            report.error(f"`{required}()` must be async")

    on_turn = methods.get("on_turn")

    if on_turn is not None and not isinstance(
        on_turn, ast.AsyncFunctionDef
    ):
        report.error("`on_turn()` must be async")

    query = methods.get("query")

    if isinstance(query, ast.AsyncFunctionDef):
        args = [arg.arg for arg in query.args.args]

        if len(args) != 2 or args[0] != "self":
            report.error("`query(self, turn)` takes exactly (self, turn)")

    return report


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    ok = True

    for argument in argv:
        path = Path(argument)

        if not path.is_file():
            print(f"FAIL: {path}: not a file")
            ok = False
            continue

        ok &= check(path).dump()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

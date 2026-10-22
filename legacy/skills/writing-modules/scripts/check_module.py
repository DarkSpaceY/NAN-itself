#!/usr/bin/env python3
"""Static checker for NAN-itself Module folders.

Usage:
    check_module.py <module_folder> [<module_folder> ...]

Checks the folder contract enforced by the module loader:
  - a main.py entry file exists
  - main.py defines exactly one concrete Module subclass
    (classes in sibling files are not entry candidates)
  - `requires` (if present) is a tuple of unique identifier strings
  - `async def start(self)` and `async def ask(self, turn)` present

Pure AST checks; no file is imported or executed.
Exits 0 when every folder passes, 1 otherwise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ENTRY = "main.py"
BASES = {"Module"}


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


def check(folder: Path) -> Report:
    report = Report(folder)

    if not folder.is_dir():
        report.error("not a folder")
        return report

    entry = folder / ENTRY

    if not entry.is_file():
        report.error(f"missing {ENTRY}")
        return report

    try:
        text = entry.read_text(encoding="utf-8")
    except OSError as exc:
        report.error(f"cannot read: {exc}")
        return report

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        report.error(f"syntax error: {exc}")
        return report

    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]

    subclasses = [
        node
        for node in classes
        if any(
            (isinstance(base, ast.Name) and base.id in BASES)
            or (isinstance(base, ast.Attribute) and base.attr in BASES)
            for base in node.bases
        )
    ]

    if len(subclasses) != 1:
        report.error(f"expected exactly one Module subclass, found {len(subclasses)}")

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

    if "id" in assignments:
        report.warn("class assigns `id`; identity is the folder name and `id` is ignored")

    requires = assignments.get("requires")

    if requires is not None:
        if not isinstance(requires, ast.Tuple):
            report.error(
                "`requires` must be a tuple of identifier strings "
                "(tuple[str, ...]), not "
                f"{type(requires).__name__}"
            )

        else:
            names = []

            for element in requires.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    names.append(element.value)
                else:
                    report.error("`requires` entries must be string literals")

            duplicates = {name for name in names if names.count(name) > 1}

            if duplicates:
                report.error(f"duplicate requires entries: {sorted(duplicates)}")

    methods = {
        node.name: node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for required in ("start", "ask"):
        method = methods.get(required)

        if method is None:
            report.error(f"missing `{required}()`")
        elif not isinstance(method, ast.AsyncFunctionDef):
            report.error(f"`{required}()` must be async")

    tell = methods.get("tell")

    if tell is not None and not isinstance(tell, ast.AsyncFunctionDef):
        report.error("`tell()` must be async")

    ask = methods.get("ask")

    if isinstance(ask, ast.AsyncFunctionDef):
        args = [arg.arg for arg in ask.args.args]

        if len(args) != 2 or args[0] != "self":
            report.error("`ask(self, turn)` takes exactly (self, turn)")

    return report


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    ok = True

    for argument in argv:
        path = Path(argument)

        ok &= check(path).dump()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

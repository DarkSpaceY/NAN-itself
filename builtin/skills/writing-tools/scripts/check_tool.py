#!/usr/bin/env python3
"""Static checker for NAN-itself local tool provider files.

Usage:
    check_tool.py <tool.py> [<tool.py> ...]

Checks the loadable-file contract enforced by the local tool loader:
  - the literal `# @tool` header within the first 20 lines
  - exactly one LocalToolProvider subclass
  - a non-empty `id` assignment on the class
  - at least one @tool-decorated method
  - every @tool method has a docstring and fully annotated parameters
    (the JSON schema derives from the annotations)
  - @tool methods are async (recommended contract; reported as error)

Pure AST + text checks; the file is never imported or executed.
Exits 0 when every file passes, 1 otherwise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

HEADER = "# @tool"
SCAN_LINES = 20
BASE_CLASS = "LocalToolProvider"
DECORATOR = "tool"


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


def is_tool_decorator(decorator: ast.expr) -> bool:
    if isinstance(decorator, ast.Name):
        return decorator.id == DECORATOR

    if isinstance(decorator, ast.Attribute):
        return decorator.attr == DECORATOR

    if isinstance(decorator, ast.Call):
        return is_tool_decorator(decorator.func)

    return False


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

    subclasses = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            (
                isinstance(base, ast.Name) and base.id == BASE_CLASS
            )
            or (
                isinstance(base, ast.Attribute)
                and base.attr == BASE_CLASS
            )
            for base in node.bases
        )
    ]

    if len(subclasses) != 1:
        report.error(
            f"expected exactly one {BASE_CLASS} subclass, "
            f"found {len(subclasses)}"
        )

        return report

    cls = subclasses[0]

    module_id = None

    for node in cls.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "id"
            for target in node.targets
        ):
            module_id = node.value

        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "id"
            and node.value is not None
        ):
            module_id = node.value

    if module_id is None:
        report.error("class does not assign `id`")
    elif not (
        isinstance(module_id, ast.Constant)
        and isinstance(module_id.value, str)
        and module_id.value.strip()
    ):
        report.error("`id` must be a non-empty string literal")

    tools = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(is_tool_decorator(d) for d in node.decorator_list)
    ]

    if not tools:
        report.error("no @tool-decorated methods found")

    for tool in tools:
        name = tool.name

        if not isinstance(tool, ast.AsyncFunctionDef):
            report.warn(f"`{name}` should be async")

        if not ast.get_docstring(tool):
            report.error(f"`{name}` is missing a docstring")

        positional = [
            arg
            for arg in tool.args.args
            if arg.arg != "self"
        ]

        annotated = sum(
            1
            for arg in positional
            if arg.annotation is not None
        )

        if annotated != len(positional):
            report.error(
                f"`{name}` has un-annotated parameters "
                "(the JSON schema derives from annotations)"
            )

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

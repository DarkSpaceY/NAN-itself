#!/usr/bin/env python3
"""Static checker for NAN-itself local tool folders.

Usage:
    check_tool.py <tool_folder> [<tool_folder> ...]

Checks the folder contract enforced by the local tool loader:
  - a main.py entry file exists
  - main.py defines exactly one ToolSet subclass
    (classes in sibling files are not entry candidates)
  - at least one @tool-decorated method
  - every @tool method has a docstring and fully annotated parameters
    (the JSON schema derives from the annotations)
  - @tool methods take no *args / **kwargs (var args are rejected by
    the loader)
  - @tool methods are async (recommended contract; reported as a warning)

Pure AST checks; no file is imported or executed.
Exits 0 when every folder passes, 1 otherwise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ENTRY = "main.py"
BASE_CLASS = "ToolSet"
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

    subclasses = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            (isinstance(base, ast.Name) and base.id == BASE_CLASS)
            or (isinstance(base, ast.Attribute) and base.attr == BASE_CLASS)
            for base in node.bases
        )
    ]

    if len(subclasses) != 1:
        report.error(f"expected exactly one {BASE_CLASS} subclass, found {len(subclasses)}")

        return report

    cls = subclasses[0]

    has_id = any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "id" for target in node.targets)
        or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "id"
            and node.value is not None
        )
        for node in cls.body
    )

    if has_id:
        report.warn("class assigns `id`; identity is the folder name and `id` is ignored")

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

        if tool.args.vararg is not None:
            report.error(
                f"`{name}` takes *{tool.args.vararg.arg}; "
                "var args are not supported by the tool loader"
            )

        if tool.args.kwarg is not None:
            report.error(
                f"`{name}` takes **{tool.args.kwarg.arg}; "
                "var args are not supported by the tool loader"
            )

        positional = [arg for arg in tool.args.args if arg.arg != "self"]

        annotated = sum(1 for arg in positional if arg.annotation is not None)

        if annotated != len(positional):
            report.error(
                f"`{name}` has un-annotated parameters (the JSON schema derives from annotations)"
            )

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

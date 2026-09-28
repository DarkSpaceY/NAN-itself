"""
Builtin skill checker tests (static AST checkers, no network).

Covers the checkers shipped with the builtin skills:
    - check_module.py: a non-tuple `requires` is rejected, while the
      same module with a tuple passes
    - check_tool.py: *args / **kwargs on a @tool method are errors
    - guards: the shipped example files still pass their checkers
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]

CHECK_MODULE = (
    REPO
    / "builtin"
    / "skills"
    / "writing-modules"
    / "scripts"
    / "check_module.py"
)

CHECK_TOOL = (
    REPO
    / "builtin"
    / "skills"
    / "writing-tools"
    / "scripts"
    / "check_tool.py"
)

EXAMPLE_MODULE = (
    REPO
    / "builtin"
    / "skills"
    / "writing-modules"
    / "references"
    / "example-module.py"
)

EXAMPLE_TOOL = (
    REPO
    / "builtin"
    / "skills"
    / "writing-tools"
    / "references"
    / "example-tool.py"
)


def _run(
    checker: Path,
    target: Path,
) -> subprocess.CompletedProcess[str]:
    """
    Run one checker script against one file, like the skill does.
    """
    return subprocess.run(
        [
            sys.executable,
            str(checker),
            str(target),
        ],
        capture_output=True,
        text=True,
    )


def _module_source(requires: str) -> str:
    """
    A minimal valid module; `requires` is spliced in verbatim so a
    tuple or a list can be exercised side by side.
    """
    return (
        "# @module\n"
        '"""Checker fixture module."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "from typing import ClassVar\n"
        "\n"
        "\n"
        "class FixtureModule(Module):\n"
        "\n"
        '    id = "fixture"\n'
        "\n"
        "    requires: ClassVar[tuple[str, ...]] = "
        f"{requires}\n"
        "\n"
        "    async def start(self) -> None:\n"
        "        return None\n"
        "\n"
        "    async def ask(self, turn: Turn) -> str | None:\n"
        "        return None\n"
    )


def _tool_source(parameters: str) -> str:
    """
    A minimal valid tool provider; `parameters` is spliced after
    `self, ` so var args can be exercised.
    """
    return (
        "# @tool\n"
        '"""Checker fixture tool provider."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "from typing import Any\n"
        "\n"
        "\n"
        "class FixtureProvider(ToolSet):\n"
        "\n"
        '    id = "fixture"\n'
        "\n"
        "    @tool\n"
        f"    async def run(self, {parameters}) -> Any:\n"
        '        """Run the fixture.\n'
        "\n"
        "        Args:\n"
        "            arguments: ignored.\n"
        '        """\n'
        "        return None\n"
    )


def test_check_module_rejects_list_requires(tmp_path: Path) -> None:
    path = tmp_path / "fixture_module.py"

    path.write_text(
        _module_source('["a", "b"]'),
        encoding="utf-8",
    )

    result = _run(CHECK_MODULE, path)

    assert result.returncode == 1

    assert "requires" in result.stdout

    assert "tuple" in result.stdout


def test_check_module_accepts_tuple_requires(tmp_path: Path) -> None:
    path = tmp_path / "fixture_module.py"

    path.write_text(
        _module_source('("a", "b")'),
        encoding="utf-8",
    )

    result = _run(CHECK_MODULE, path)

    assert result.returncode == 0


def test_check_tool_rejects_var_positional(tmp_path: Path) -> None:
    path = tmp_path / "fixture_tool.py"

    path.write_text(
        _tool_source("*args: Any"),
        encoding="utf-8",
    )

    result = _run(CHECK_TOOL, path)

    assert result.returncode == 1

    assert "*args" in result.stdout


def test_check_tool_rejects_var_keyword(tmp_path: Path) -> None:
    path = tmp_path / "fixture_tool.py"

    path.write_text(
        _tool_source("**kwargs: Any"),
        encoding="utf-8",
    )

    result = _run(CHECK_TOOL, path)

    assert result.returncode == 1

    assert "**kwargs" in result.stdout


def test_example_module_still_passes() -> None:
    result = _run(CHECK_MODULE, EXAMPLE_MODULE)

    assert result.returncode == 0


def test_example_tool_still_passes() -> None:
    result = _run(CHECK_TOOL, EXAMPLE_TOOL)

    assert result.returncode == 0

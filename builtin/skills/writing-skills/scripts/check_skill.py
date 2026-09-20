#!/usr/bin/env python3
"""Static checker for NAN-itself SKILL.md files.

Usage:
    check_skill.py <skill-dir-or-SKILL.md> [<...> ...]

Checks the validation contract enforced by the skill registry:
  - SKILL.md exists and starts with `---` frontmatter terminated by `---`
  - frontmatter parses as a YAML mapping
  - `name`: string, kebab-case ^[a-z0-9]+(-[a-z0-9]+)*$, max 64 chars
  - `description`: non-empty string, max 1024 chars
  - resource folders (scripts/references/assets), if present, contain
    no hidden files and scripts/ has no unknown-interpreter leftovers
    reported as warnings

Pure text + YAML checks; scripts are never executed.
Exits 0 when every skill passes, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_NAME = 64
MAX_DESCRIPTION = 1024
GROUPS = ("scripts", "references", "assets")

KNOWN_SCRIPT_SUFFIXES = {".py", ".sh", ".bash", ".js"}


def fail(message: str) -> None:
    print(f"FAIL: {message}")


def warn(message: str) -> None:
    print(f"warn: {message}")


def check(root: Path) -> bool:
    skill_file = (
        root if root.is_file() else root / "SKILL.md"
    )

    ok = True

    if not skill_file.is_file():
        fail(f"{skill_file}: SKILL.md not found")
        return False

    text = skill_file.read_text(encoding="utf-8")

    lines = text.splitlines()

    if not lines or lines[0].strip() != "---":
        fail(f"{skill_file}: frontmatter must start at line 1 with '---'")
        return False

    end_index = None

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break

    if end_index is None:
        fail(f"{skill_file}: frontmatter is not terminated by '---'")
        return False

    try:
        frontmatter = yaml.safe_load("\n".join(lines[1:end_index])) or {}
    except yaml.YAMLError as exc:
        fail(f"{skill_file}: invalid YAML frontmatter: {exc}")
        return False

    if not isinstance(frontmatter, dict):
        fail(f"{skill_file}: frontmatter must be a mapping")
        return False

    name = frontmatter.get("name")

    if not isinstance(name, str) or not name:
        fail(f"{skill_file}: 'name' must be a non-empty string")
        ok = False
    else:
        if len(name) > MAX_NAME:
            fail(
                f"{skill_file}: 'name' exceeds {MAX_NAME} characters"
            )
            ok = False

        if NAME_RE.fullmatch(name) is None:
            fail(
                f"{skill_file}: 'name' is not kebab-case "
                f"(^[a-z0-9]+(-[a-z0-9]+)*$): {name!r}"
            )
            ok = False
        elif root.is_dir() and root.name != name:
            warn(
                f"{root}: directory name {root.name!r} != name {name!r}"
            )

    description = frontmatter.get("description")

    if not isinstance(description, str) or not description.strip():
        fail(f"{skill_file}: 'description' must be a non-empty string")
        ok = False
    elif len(description) > MAX_DESCRIPTION:
        fail(
            f"{skill_file}: 'description' exceeds "
            f"{MAX_DESCRIPTION} characters"
        )
        ok = False

    body = "\n".join(lines[end_index + 1:]).strip()

    if not body:
        warn(f"{skill_file}: body is empty")

    if root.is_dir():
        for group in GROUPS:
            directory = root / group

            if not directory.is_dir():
                continue

            for path in sorted(directory.rglob("*")):
                if not path.is_file():
                    continue

                relative = path.relative_to(directory)

                if any(
                    part.startswith(".")
                    for part in relative.parts
                ):
                    warn(
                        f"{path}: hidden files are excluded from resources"
                    )
                    continue

                if group == "scripts" and path.suffix.lower() not in (
                    KNOWN_SCRIPT_SUFFIXES
                ):
                    resolved = path.resolve()

                    if not (
                        path.suffix == ""
                        and resolved.stat().st_mode & 0o111
                    ):
                        warn(
                            f"{path}: extension is not in the interpreter "
                            "table and the file is not executable "
                            "(shebang + exec bit required)"
                        )

    return ok


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    ok = True

    for argument in argv:
        path = Path(argument)

        if not path.exists():
            fail(f"{path}: not found")
            ok = False
            continue

        ok &= check(path)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

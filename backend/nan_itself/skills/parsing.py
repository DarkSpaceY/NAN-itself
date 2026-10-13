"""
SKILL.md parsing and field validation.

Owns everything about the file format itself:

    - splitting YAML frontmatter from the Markdown body
    - validating name / description constraints
    - enumerating bundled resource folders
    - change fingerprints

Pure functions over paths and text; no registry state.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import yaml

from .model import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
    NAME_RE,
    SKILL_FILENAME,
    SkillMetadata,
    SkillValidationError,
)


def split_skill_file(
    text: str,
) -> tuple[dict[str, Any], str]:
    """
    Split YAML frontmatter and Markdown body.

    Frontmatter must start at the very beginning with --- and
    terminate with the next --- line.
    """
    lines = text.splitlines()

    if not lines or lines[0].strip() != "---":
        raise SkillValidationError("SKILL.md must start with YAML frontmatter")

    end_index: int | None = None

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break

    if end_index is None:
        raise SkillValidationError("SKILL.md frontmatter is not terminated")

    frontmatter_text = "\n".join(lines[1:end_index])

    body = "\n".join(lines[end_index + 1 :])

    parsed = yaml.safe_load(frontmatter_text) or {}

    if not isinstance(parsed, dict):
        raise SkillValidationError("SKILL.md frontmatter must be a mapping")

    return parsed, body


def read_metadata(
    root: Path,
) -> SkillMetadata:
    skill_file = root / SKILL_FILENAME

    if not skill_file.is_file():
        raise SkillValidationError(f"Missing {SKILL_FILENAME}: {root}")

    text = skill_file.read_text(
        encoding="utf-8",
    )

    frontmatter, _ = split_skill_file(text)

    name = frontmatter.get("name")
    description = frontmatter.get("description")

    validate_name(name, skill_file)

    validate_description(description, skill_file)

    # 上面两个校验器不通过（含 None）即抛 SkillValidationError。
    return SkillMetadata(
        name=cast(str, name),
        description=cast(str, description),
        source=skill_file,
        frontmatter=MappingProxyType(dict(frontmatter)),
    )


def read_instructions(
    path: Path,
) -> str:
    text = path.read_text(
        encoding="utf-8",
    )

    _, body = split_skill_file(text)

    return body.strip()


def validate_name(
    name: Any,
    path: Path,
) -> None:
    if not isinstance(name, str):
        raise SkillValidationError(f"Skill name must be a string: {path}")

    if not name:
        raise SkillValidationError(f"Skill name cannot be empty: {path}")

    if len(name) > MAX_NAME_LENGTH:
        raise SkillValidationError(f"Skill name exceeds {MAX_NAME_LENGTH} characters: {path}")

    if NAME_RE.fullmatch(name) is None:
        raise SkillValidationError(f"Invalid Skill name '{name}': {path}")


def validate_description(
    description: Any,
    path: Path,
) -> None:
    if not isinstance(description, str):
        raise SkillValidationError(f"Skill description must be a string: {path}")

    if not description.strip():
        raise SkillValidationError(f"Skill description cannot be empty: {path}")

    if len(description) > MAX_DESCRIPTION_LENGTH:
        raise SkillValidationError(
            f"Skill description exceeds {MAX_DESCRIPTION_LENGTH} characters: {path}"
        )


def skill_files(
    root: Path,
) -> tuple[Path, ...]:
    """
    Enumerate every file belonging to a Skill.

    The protocol fixes only SKILL.md itself; scripts/,
    references/ and assets/ are conventions, not the
    exhaustive surface. A Skill directory may contain
    arbitrary additional files and folders and all of them
    belong to the Skill. Hidden entries and __pycache__ are
    excluded; ordering is stable.
    """
    from nan_itself.utils.hotload import detected_files

    return detected_files(root, exclude_dirs=())


def directory_fingerprint(
    root: Path,
) -> tuple[int, int]:
    """
    Aggregate change fingerprint over every Skill file.

    A single-file fingerprint would miss edits to bundled
    resources, so the whole directory is folded into one
    value.
    """
    from nan_itself.utils.hotload import directory_fingerprint as _fingerprint

    return _fingerprint(root, exclude_dirs=())

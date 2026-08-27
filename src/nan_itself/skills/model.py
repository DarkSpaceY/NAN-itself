"""
Skill data model and format contract.

A skill is a directory containing SKILL.md with YAML frontmatter
plus optional scripts/, references/, assets/ resource folders.
This module owns the vocabulary (types, exceptions, limits) that
every other skills module speaks; it imports nothing from them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


NAME_RE = re.compile(
    r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
)

MAX_NAME_LENGTH = 64

MAX_DESCRIPTION_LENGTH = 1024

SKILL_FILENAME = "SKILL.md"


@dataclass(frozen=True)
class SkillMetadata:
    """
    Lightweight Skill metadata.

    This is what the runtime can expose in a catalog without loading
    the full SKILL.md body.
    """

    name: str
    description: str

    source: Path
    origin: str  # "builtin" | "workspace"

    frontmatter: Mapping[str, Any]


@dataclass(frozen=True)
class Skill:
    """
    Fully loaded Skill.

    The instructions are the Markdown body of SKILL.md.

    Bundled resources are intentionally represented as paths rather than
    eagerly loaded bytes/text. This preserves progressive disclosure.
    """

    metadata: SkillMetadata
    instructions: str

    scripts: tuple[Path, ...]
    references: tuple[Path, ...]
    assets: tuple[Path, ...]

    generation: int = 0

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def description(self) -> str:
        return self.metadata.description

    @property
    def source(self) -> Path:
        return self.metadata.source

    @property
    def origin(self) -> str:
        return self.metadata.origin

    @property
    def frontmatter(self) -> Mapping[str, Any]:
        return self.metadata.frontmatter


class SkillValidationError(ValueError):
    pass


class UnknownSkillError(SkillValidationError):
    """
    Raised when activating a Skill name that is not registered.

    Part of the same ValueError family as validation failures so
    callers can handle all skill errors uniformly.
    """

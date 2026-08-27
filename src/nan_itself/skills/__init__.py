"""
Skills package public surface.

Layering (dependencies point downward only):

    runtime ──> registry ──> { model, parsing }

Import from this package, never from sibling modules.
"""

from src.nan_itself.skills.model import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
    SKILL_FILENAME,
    Skill,
    SkillMetadata,
    SkillValidationError,
    UnknownSkillError,
)
from src.nan_itself.skills.runtime import (
    SkillRuntime,
)

__all__ = [
    "MAX_DESCRIPTION_LENGTH",
    "MAX_NAME_LENGTH",
    "SKILL_FILENAME",
    "Skill",
    "SkillMetadata",
    "SkillValidationError",
    "UnknownSkillError",
    "SkillRuntime",
]

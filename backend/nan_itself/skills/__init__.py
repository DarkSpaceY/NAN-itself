"""
Skills package public surface.

Layering (dependencies point downward only):

    runtime ──> registry ──> { model, parsing }

Import from this package, never from sibling modules.
"""

from .model import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
    SKILL_FILENAME,
    SkillMetadata,
    SkillValidationError,
    UnknownSkillError,
)
from .runtime import (
    SkillRuntime,
)

__all__ = [
    "MAX_DESCRIPTION_LENGTH",
    "MAX_NAME_LENGTH",
    "SKILL_FILENAME",
    "SkillMetadata",
    "SkillValidationError",
    "UnknownSkillError",
    "SkillRuntime",
]

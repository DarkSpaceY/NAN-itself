"""
Skill discovery and registration bookkeeping.

Owns every mutable piece of skill state:

    - the name -> record table
    - per-directory change fingerprints and sticky load errors
    - generation counters across reloads

Discovery policies enforced here:

    - workspace skills cannot override builtin ones
    - one name may only be claimed by one source directory
    - an unchanged broken skill directory is never retried

This module does not parse file contents beyond delegating to
the parsing module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.nan_itself.skills.model import (
    SKILL_FILENAME,
    SkillMetadata,
    SkillValidationError,
)
from src.nan_itself.skills.parsing import (
    fingerprint,
    read_metadata,
)


@dataclass
class SkillRecord:
    """
    Internal registry entry for one discovered skill.
    """

    metadata: SkillMetadata
    root: Path
    generation: int = 0
    fingerprint: tuple[int, int] | None = None


class SkillRegistry:
    def __init__(self) -> None:
        # Skill name -> record.
        self.records: dict[str, SkillRecord] = {}

        # Workspace skill dir -> latest fingerprint.
        self.fingerprints: dict[
            Path,
            tuple[int, int],
        ] = {}

        # Workspace skill dir -> last load error for fingerprint.
        self.errors: dict[
            Path,
            BaseException,
        ] = {}

        # Skill name -> generation counter (survives reloads).
        self.generations: dict[str, int] = {}

    # ==================================================================
    # Discovery entry points
    # ==================================================================

    def discover(
        self,
        *,
        builtin_roots: Iterable[Path],
        workspace_root: Path,
    ) -> None:
        self.discover_builtin(builtin_roots)

        self.discover_workspace(workspace_root)

    def discover_builtin(
        self,
        roots: Iterable[Path],
    ) -> None:
        for root in roots:
            if not root.exists():
                continue

            if not root.is_dir():
                raise ValueError(
                    f"Builtin Skill root is not a directory: {root}"
                )

            # Allow either:
            #
            # builtin_root/foo/SKILL.md
            #
            # or:
            #
            # builtin_root/SKILL.md
            #
            if (root / SKILL_FILENAME).is_file():
                self.register(root, origin="builtin")
                continue

            for child in sorted(root.iterdir()):
                if child.is_dir():
                    self.register(child, origin="builtin")

    def discover_workspace(
        self,
        root: Path,
    ) -> None:
        root.mkdir(
            parents=True,
            exist_ok=True,
        )

        current = {
            path.resolve()
            for path in root.iterdir()
            if path.is_dir()
            and not path.name.startswith("_")
            and (path / SKILL_FILENAME).is_file()
        }

        known = set(self.fingerprints)

        # Removed Skills.
        for skill_dir in known - current:
            record = self.record_by_workspace_dir(skill_dir)

            if record is not None:
                self.records.pop(
                    record.metadata.name,
                    None,
                )

            self.fingerprints.pop(skill_dir, None)

            self.errors.pop(skill_dir, None)

        # New / changed Skills.
        for skill_dir in sorted(current):
            skill_file = skill_dir / SKILL_FILENAME

            fp = fingerprint(skill_file)

            previous = self.fingerprints.get(skill_dir)

            previous_error = self.errors.get(skill_dir)

            if (
                previous == fp
                and previous_error is not None
            ):
                continue

            if (
                previous == fp
                and skill_dir in self.fingerprints
            ):
                continue

            self.fingerprints[skill_dir] = fp

            self.errors.pop(skill_dir, None)

            try:
                self.register(skill_dir, origin="workspace")

            except Exception as exc:
                self.errors[skill_dir] = exc

    # ==================================================================
    # Registration
    # ==================================================================

    def register(
        self,
        root: Path,
        *,
        origin: str,
    ) -> None:
        metadata = read_metadata(
            root,
            origin=origin,
        )

        existing = self.records.get(metadata.name)

        if existing is not None:
            if existing.metadata.origin == "builtin":
                if origin == "workspace":
                    raise SkillValidationError(
                        f"Workspace Skill '{metadata.name}' "
                        f"cannot override builtin Skill"
                    )

            if existing.metadata.source.resolve() != (
                metadata.source.resolve()
            ):
                raise SkillValidationError(
                    f"Duplicate Skill name '{metadata.name}': "
                    f"{existing.metadata.source} and "
                    f"{metadata.source}"
                )

        generation = (
            self.generations.get(
                metadata.name,
                -1,
            )
            + 1
        )

        self.generations[
            metadata.name
        ] = generation

        self.records[
            metadata.name
        ] = SkillRecord(
            metadata=metadata,
            root=root,
            generation=generation,
            fingerprint=(
                fingerprint(
                    root / SKILL_FILENAME
                )
                if origin == "workspace"
                else None
            ),
        )

    # ==================================================================
    # Lookups
    # ==================================================================

    def get_record(
        self,
        name: str,
    ) -> SkillRecord | None:
        return self.records.get(name)

    def all_records(self) -> tuple[SkillRecord, ...]:
        return tuple(
            sorted(
                self.records.values(),
                key=lambda item: item.metadata.name,
            )
        )

    def record_by_workspace_dir(
        self,
        directory: Path,
    ) -> SkillRecord | None:
        directory = directory.resolve()

        for record in self.records.values():
            if record.metadata.origin != "workspace":
                continue

            if record.root.resolve() == directory:
                return record

        return None

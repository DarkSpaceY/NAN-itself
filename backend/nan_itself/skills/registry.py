"""
Skill discovery and registration bookkeeping.

Owns every mutable piece of skill state:

    - the name -> record table
    - per-directory change fingerprints and sticky load errors
    - generation counters across reloads

Discovery policies enforced here:

    - every skill root (builtin and workspace) is scanned with
      exactly the same hot-reload logic; deleting a skill
      directory unregisters it
    - one name may only be claimed by one source directory
    - one directory owns exactly one Skill
    - an unchanged broken skill directory is never retried

This module does not parse file contents beyond delegating to
the parsing module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .model import (
    SKILL_FILENAME,
    SkillMetadata,
    SkillValidationError,
)
from .parsing import (
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

        # Skill dir -> latest fingerprint.
        self.fingerprints: dict[
            Path,
            tuple[int, int],
        ] = {}

        # Skill dir -> last load error for fingerprint.
        self.errors: dict[
            Path,
            BaseException,
        ] = {}

        # Skill name -> generation counter (survives reloads).
        self.generations: dict[str, int] = {}

    # ==================================================================
    # Discovery
    # ==================================================================

    def discover_roots(
        self,
        roots: Iterable[Path],
    ) -> None:
        """
        Re-scan every root. Removed directories are unregistered;
        changed directories are re-registered transactionally.
        """
        for root in roots:
            self.discover_root(root)

    def discover_root(
        self,
        root: Path,
    ) -> None:
        root = root.resolve()

        root.mkdir(
            parents=True,
            exist_ok=True,
        )

        current = {
            path.resolve()
            for path in root.iterdir()
            if (
                path.is_dir()
                and not path.name.startswith("_")
                and (
                    path / SKILL_FILENAME
                ).is_file()
            )
        }

        # A root that is itself a skill directory is also accepted.
        if (
            root / SKILL_FILENAME
        ).is_file():
            current.add(root)

        known = {
            path
            for path in self.fingerprints
            if path.is_relative_to(root)
        }

        # --------------------------------------------------------------
        # Removed Skills.
        # --------------------------------------------------------------

        for skill_dir in (
            known - current
        ):
            record = (
                self.record_by_dir(
                    skill_dir
                )
            )

            if record is not None:
                self.records.pop(
                    record.metadata.name,
                    None,
                )

            self.fingerprints.pop(
                skill_dir,
                None,
            )

            self.errors.pop(
                skill_dir,
                None,
            )

        # --------------------------------------------------------------
        # New / changed Skills.
        # --------------------------------------------------------------

        for skill_dir in sorted(
            current
        ):
            skill_file = (
                skill_dir / SKILL_FILENAME
            )

            fp = fingerprint(
                skill_file
            )

            previous = (
                self.fingerprints.get(
                    skill_dir
                )
            )

            previous_error = (
                self.errors.get(
                    skill_dir
                )
            )

            # Don't repeatedly retry an unchanged broken Skill.
            if (
                previous == fp
                and previous_error is not None
            ):
                continue

            # Nothing changed.
            if (
                previous == fp
                and skill_dir in self.fingerprints
            ):
                continue

            self.fingerprints[
                skill_dir
            ] = fp

            self.errors.pop(
                skill_dir,
                None,
            )

            try:
                self.register(
                    skill_dir
                )

            except Exception as exc:
                self.errors[
                    skill_dir
                ] = exc

    # ==================================================================
    # Registration
    # ==================================================================

    def register(
        self,
        root: Path,
    ) -> None:
        root = root.resolve()

        # --------------------------------------------------------------
        # Parse candidate FIRST.
        #
        # Nothing in registry state is mutated until metadata parsing
        # succeeds.
        # --------------------------------------------------------------

        metadata = read_metadata(
            root
        )

        # --------------------------------------------------------------
        # Find the Skill currently owned by this directory.
        # --------------------------------------------------------------

        previous = (
            self.record_by_dir(
                root
            )
        )

        existing = self.records.get(
            metadata.name
        )

        # --------------------------------------------------------------
        # Validate the candidate name BEFORE removing the old record.
        #
        # This is critical for transactional reload.
        #
        # Example:
        #
        #   alpha/ -> alpha
        #   beta/  -> beta
        #
        # If alpha/ changes to name=beta, beta already belongs to
        # another source. The old alpha record MUST remain intact.
        # --------------------------------------------------------------

        if (
            existing is not None
            and existing is not previous
            and existing.metadata.source.resolve()
            != metadata.source.resolve()
        ):
            raise SkillValidationError(
                f"Duplicate Skill name '{metadata.name}': "
                f"{existing.metadata.source} and "
                f"{metadata.source}"
            )

        # --------------------------------------------------------------
        # All validation has passed.
        #
        # NOW it is safe to replace the old record owned by this
        # directory if its Skill name changed.
        # --------------------------------------------------------------

        if (
            previous is not None
            and previous.metadata.name
            != metadata.name
        ):
            self.records.pop(
                previous.metadata.name,
                None,
            )

        # --------------------------------------------------------------
        # Generation.
        #
        # The counter is maintained by Skill name and intentionally
        # survives removal/reappearance.
        # --------------------------------------------------------------

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
            fingerprint=fingerprint(
                root / SKILL_FILENAME
            ),
        )

    # ==================================================================
    # Lookups
    # ==================================================================

    def get_record(
        self,
        name: str,
    ) -> SkillRecord | None:
        return self.records.get(
            name
        )

    def all_records(
        self,
    ) -> tuple[SkillRecord, ...]:
        return tuple(
            sorted(
                self.records.values(),
                key=lambda item: (
                    item.metadata.name
                ),
            )
        )

    def record_by_dir(
        self,
        directory: Path,
    ) -> SkillRecord | None:
        directory = (
            directory.resolve()
        )

        for record in (
            self.records.values()
        ):
            if (
                record.root.resolve()
                == directory
            ):
                return record

        return None

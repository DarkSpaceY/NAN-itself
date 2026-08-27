"""
Skill runtime facade.

The agent-facing surface of the skills package:

    - discover(): one-shot discovery at boot
    - refresh(): re-scan workspace skills on demand (called by
      the agent loop at every turn start, so skills dropped into
      the workspace become visible without a restart)
    - catalog()/names()/get_metadata(): lightweight lookups that
      never load skill bodies
    - activate(): full load with progressive disclosure — the
      body is read, resources stay lazy paths

Deliberately synchronous: discovery only touches metadata files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.nan_itself.skills.model import (
    SKILL_FILENAME,
    Skill,
    SkillMetadata,
    UnknownSkillError,
)
from src.nan_itself.skills.parsing import (
    read_instructions,
    resource_files,
)
from src.nan_itself.skills.registry import (
    SkillRegistry,
)


class SkillRuntime:
    """
    Agent Skills runtime.

    Responsibilities:
        - discover builtin Skills
        - discover workspace Skills
        - validate SKILL.md
        - maintain lightweight metadata catalog
        - activate/load complete Skills
        - detect workspace changes

    It deliberately does NOT:
        - manage Agents
        - manage Tools
        - execute Skill scripts
        - decide which Skill an Agent should use
    """

    def __init__(
        self,
        workspace_skills: str | Path | None = None,
        *,
        builtin_skills: Iterable[Path] = (),
    ) -> None:
        project_root = Path(__file__).resolve().parents[3]

        self.workspace_skills = (
            Path(workspace_skills).resolve()
            if workspace_skills is not None
            else (
                project_root
                / "workspace"
                / "skills"
            ).resolve()
        )

        self.builtin_skills = tuple(
            Path(path).resolve()
            for path in builtin_skills
        )

        self._registry = SkillRegistry()

    # ==================================================================
    # Discovery
    # ==================================================================

    def discover(self) -> None:
        """
        Discover all Skills synchronously.

        This is intentionally synchronous because discovery only reads
        metadata files. The caller may run it in a thread if needed.
        """
        self._registry.discover(
            builtin_roots=self.builtin_skills,
            workspace_root=self.workspace_skills,
        )

    def refresh(self) -> None:
        """
        Re-scan workspace Skills.

        Builtins are not repeatedly reloaded.
        """
        self._registry.discover_workspace(
            self.workspace_skills
        )

    # ==================================================================
    # Catalog
    # ==================================================================

    def catalog(self) -> tuple[SkillMetadata, ...]:
        """
        Return lightweight metadata only.

        Full SKILL.md bodies are not loaded here.
        """
        return tuple(
            record.metadata
            for record in self._registry.all_records()
        )

    def get_metadata(self, name: str) -> SkillMetadata | None:
        record = self._registry.get_record(name)

        if record is None:
            return None

        return record.metadata

    def names(self) -> tuple[str, ...]:
        return tuple(
            metadata.name
            for metadata in self.catalog()
        )

    # ==================================================================
    # Activation
    # ==================================================================

    def activate(
        self,
        name: str,
    ) -> Skill:
        """
        Load the complete Skill.

        Activation loads SKILL.md instructions, but leaves references,
        assets, and scripts lazy as filesystem paths.
        """
        record = self._registry.get_record(name)

        if record is None:
            raise UnknownSkillError(
                f"Unknown Skill: {name}"
            )

        skill_file = record.root / SKILL_FILENAME

        instructions = read_instructions(skill_file)

        scripts = resource_files(
            record.root / "scripts"
        )

        references = resource_files(
            record.root / "references"
        )

        assets = resource_files(
            record.root / "assets"
        )

        return Skill(
            metadata=record.metadata,
            instructions=instructions,
            scripts=scripts,
            references=references,
            assets=assets,
            generation=record.generation,
        )

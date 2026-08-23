from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import yaml


_NAME_RE = re.compile(
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


@dataclass
class _SkillRecord:
    metadata: SkillMetadata
    root: Path
    generation: int = 0
    fingerprint: tuple[int, int] | None = None


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
        builtin_skills: Iterable[str | Path] = (),
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]

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

        self._skills: dict[str, _SkillRecord] = {}

        self._workspace_fingerprints: dict[
            Path,
            tuple[int, int],
        ] = {}

        self._workspace_errors: dict[
            Path,
            BaseException,
        ] = {}

        self._generations: dict[str, int] = {}

    # ==================================================================
    # Discovery
    # ==================================================================

    def discover(self) -> None:
        """
        Discover all Skills synchronously.

        This is intentionally synchronous because discovery only reads
        metadata files. The caller may run it in a thread if needed.
        """
        self._discover_builtin()
        self._discover_workspace()

    def _discover_builtin(self) -> None:
        for root in self.builtin_skills:
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
                self._register_skill_directory(
                    root,
                    origin="builtin",
                )
                continue

            for child in sorted(root.iterdir()):
                if child.is_dir():
                    self._register_skill_directory(
                        child,
                        origin="builtin",
                    )

    def _discover_workspace(self) -> None:
        root = self.workspace_skills
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

        known = set(
            self._workspace_fingerprints
        )

        # Removed Skills.
        for skill_dir in known - current:
            record = self._record_by_workspace_dir(
                skill_dir
            )

            if record is not None:
                self._skills.pop(
                    record.metadata.name,
                    None,
                )

            self._workspace_fingerprints.pop(
                skill_dir,
                None,
            )

            self._workspace_errors.pop(
                skill_dir,
                None,
            )

        # New / changed Skills.
        for skill_dir in sorted(current):
            skill_file = skill_dir / SKILL_FILENAME
            fingerprint = self._fingerprint(skill_file)

            previous = self._workspace_fingerprints.get(
                skill_dir
            )

            previous_error = self._workspace_errors.get(
                skill_dir
            )

            if (
                previous == fingerprint
                and previous_error is not None
            ):
                continue

            if (
                previous == fingerprint
                and skill_dir in self._workspace_fingerprints
            ):
                continue

            self._workspace_fingerprints[
                skill_dir
            ] = fingerprint

            self._workspace_errors.pop(
                skill_dir,
                None,
            )

            try:
                self._register_skill_directory(
                    skill_dir,
                    origin="workspace",
                )

            except Exception as exc:
                self._workspace_errors[
                    skill_dir
                ] = exc

    def _register_skill_directory(
        self,
        root: Path,
        *,
        origin: str,
    ) -> None:
        metadata = self._read_metadata(
            root,
            origin=origin,
        )

        existing = self._skills.get(
            metadata.name
        )

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
            self._generations.get(
                metadata.name,
                -1,
            )
            + 1
        )

        self._generations[
            metadata.name
        ] = generation

        self._skills[
            metadata.name
        ] = _SkillRecord(
            metadata=metadata,
            root=root,
            generation=generation,
            fingerprint=(
                self._fingerprint(
                    root / SKILL_FILENAME
                )
                if origin == "workspace"
                else None
            ),
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
            sorted(
                (
                    record.metadata
                    for record in self._skills.values()
                ),
                key=lambda item: item.name,
            )
        )

    def get_metadata(
        self,
        name: str,
    ) -> SkillMetadata | None:
        record = self._skills.get(name)

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
        record = self._skills.get(name)

        if record is None:
            raise KeyError(
                f"Unknown Skill: {name}"
            )

        skill_file = record.root / SKILL_FILENAME

        instructions = self._read_instructions(
            skill_file
        )

        scripts = self._resource_files(
            record.root / "scripts"
        )

        references = self._resource_files(
            record.root / "references"
        )

        assets = self._resource_files(
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

    # ==================================================================
    # Workspace refresh
    # ==================================================================

    def refresh(self) -> None:
        """
        Re-scan workspace Skills.

        Builtins are not repeatedly reloaded.
        """
        self._discover_workspace()

    # ==================================================================
    # File parsing
    # ==================================================================

    @classmethod
    def _read_metadata(
        cls,
        root: Path,
        *,
        origin: str,
    ) -> SkillMetadata:
        skill_file = root / SKILL_FILENAME

        if not skill_file.is_file():
            raise SkillValidationError(
                f"Missing {SKILL_FILENAME}: {root}"
            )

        text = skill_file.read_text(
            encoding="utf-8",
        )

        frontmatter, _ = cls._split_skill_file(
            text
        )

        if not isinstance(frontmatter, dict):
            raise SkillValidationError(
                f"Invalid frontmatter: {skill_file}"
            )

        name = frontmatter.get("name")
        description = frontmatter.get(
            "description"
        )

        cls._validate_name(
            name,
            skill_file,
        )

        cls._validate_description(
            description,
            skill_file,
        )

        return SkillMetadata(
            name=name,
            description=description,
            source=skill_file,
            origin=origin,
            frontmatter=MappingProxyType(
                dict(frontmatter)
            ),
        )

    @staticmethod
    def _split_skill_file(
        text: str,
    ) -> tuple[dict[str, Any], str]:
        """
        Split YAML frontmatter and Markdown body.

        Frontmatter must start at the very beginning with --- and
        terminate with the next --- line.
        """
        lines = text.splitlines()

        if not lines or lines[0].strip() != "---":
            raise SkillValidationError(
                "SKILL.md must start with YAML frontmatter"
            )

        end_index: int | None = None

        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                end_index = index
                break

        if end_index is None:
            raise SkillValidationError(
                "SKILL.md frontmatter is not terminated"
            )

        frontmatter_text = "\n".join(
            lines[1:end_index]
        )

        body = "\n".join(
            lines[end_index + 1:]
        )

        parsed = yaml.safe_load(
            frontmatter_text
        ) or {}

        if not isinstance(parsed, dict):
            raise SkillValidationError(
                "SKILL.md frontmatter must be a mapping"
            )

        return parsed, body

    @staticmethod
    def _read_instructions(
        path: Path,
    ) -> str:
        text = path.read_text(
            encoding="utf-8",
        )

        _, body = SkillRuntime._split_skill_file(
            text
        )

        return body.strip()

    @staticmethod
    def _validate_name(
        name: Any,
        path: Path,
    ) -> None:
        if not isinstance(name, str):
            raise SkillValidationError(
                f"Skill name must be a string: {path}"
            )

        if not name:
            raise SkillValidationError(
                f"Skill name cannot be empty: {path}"
            )

        if len(name) > MAX_NAME_LENGTH:
            raise SkillValidationError(
                f"Skill name exceeds "
                f"{MAX_NAME_LENGTH} characters: {path}"
            )

        if _NAME_RE.fullmatch(name) is None:
            raise SkillValidationError(
                f"Invalid Skill name '{name}': {path}"
            )

    @staticmethod
    def _validate_description(
        description: Any,
        path: Path,
    ) -> None:
        if not isinstance(description, str):
            raise SkillValidationError(
                f"Skill description must be a string: {path}"
            )

        if not description.strip():
            raise SkillValidationError(
                f"Skill description cannot be empty: {path}"
            )

        if len(description) > MAX_DESCRIPTION_LENGTH:
            raise SkillValidationError(
                f"Skill description exceeds "
                f"{MAX_DESCRIPTION_LENGTH} characters: {path}"
            )

    @staticmethod
    def _resource_files(
        directory: Path,
    ) -> tuple[Path, ...]:
        if not directory.is_dir():
            return ()

        return tuple(
            sorted(
                path.resolve()
                for path in directory.rglob("*")
                if path.is_file()
                and not any(
                    part.startswith(".")
                    for part in path.relative_to(
                        directory
                    ).parts
                )
            )
        )

    @staticmethod
    def _fingerprint(
        path: Path,
    ) -> tuple[int, int]:
        stat = path.stat()

        return (
            stat.st_mtime_ns,
            stat.st_size,
        )

    def _record_by_workspace_dir(
        self,
        directory: Path,
    ) -> _SkillRecord | None:
        directory = directory.resolve()

        for record in self._skills.values():
            if record.metadata.origin != "workspace":
                continue

            if record.root.resolve() == directory:
                return record

        return None
"""
Skill runtime facade.

The agent-facing surface of the skills package:

    - discover(): one-shot discovery at boot
    - refresh(): re-scan every skill root on demand (called by
      the agent loop at every turn start; builtin and workspace
      skills share the same hot-reload logic)
    - catalog()/names()/get_metadata(): lightweight lookups that
      never load skill bodies
    - resource_paths()/invoke(): progressive disclosure of
      bundled resources; scripts execute, other resources
      return as text

Discovery is synchronous (metadata files only); invoking a
script is async.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Iterable

from .model import (
    SkillMetadata,
    UnknownSkillError,
    SkillValidationError,
)
from .parsing import (
    resource_files,
)
from .registry import (
    SkillRecord,
    SkillRegistry,
)


# Scripts run through a fixed interpreter table; anything not
# listed is executed directly (shebang + executable bit).
_SCRIPT_COMMANDS: dict[str, tuple[str, ...]] = {
    ".py": (sys.executable,),
    ".sh": ("bash",),
    ".bash": ("bash",),
    ".js": ("node",),
}

RESOURCE_GROUPS = ("scripts", "references", "assets")


class SkillRuntime:
    """
    Agent Skills runtime.

    Responsibilities:
        - discover builtin and workspace Skills with one
          identical hot-reload scan per root
        - validate SKILL.md
        - maintain lightweight metadata catalog
        - expose bundled resources / execute scripts

    It deliberately does NOT:
        - manage Agents
        - manage Tools
        - decide which Skill an Agent should use
    """

    def __init__(
        self,
        workspace_skills: str | Path | None = None,
        builtin_skills: str | Path | None = None,
        *,
        resource_char_limit: int = 100_000,
        script_timeout: float = 300.0,
    ) -> None:
        project_root = Path(__file__).resolve().parents[3]

        self.builtin_skills = (
            Path(builtin_skills).resolve()
            if builtin_skills is not None
            else (
                project_root
                / "builtin"
                / "skills"
            ).resolve()
        )

        self.workspace_skills = (
            Path(workspace_skills).resolve()
            if workspace_skills is not None
            else (
                project_root
                / "workspace"
                / "skills"
            ).resolve()
        )

        self.resource_char_limit = (
            resource_char_limit
        )

        self.script_timeout = (
            script_timeout
        )

        self._registry = SkillRegistry()

    # ==================================================================
    # Discovery
    # ==================================================================

    def discover(self) -> None:
        """
        Discover all Skills synchronously.

        This is intentionally synchronous because discovery only
        reads metadata files. The caller may run it in a thread
        if needed.
        """
        self._registry.discover_roots(
            (
                self.builtin_skills,
                self.workspace_skills,
            )
        )

    def refresh(self) -> None:
        """
        Re-scan every skill root.

        Builtin and workspace skills are hot-reloadable through
        exactly the same path.
        """
        self.discover()

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
    # Resources
    # ==================================================================

    def resource_paths(
        self,
        name: str,
    ) -> dict[str, tuple[str, ...]]:
        """
        Relative paths of one Skill's bundled resources,
        grouped by folder.
        """
        record = self._record(name)

        return {
            group: tuple(
                path.relative_to(
                    record.root
                ).as_posix()
                for path in resource_files(
                    record.root / group
                )
            )
            for group in RESOURCE_GROUPS
        }

    async def invoke(
        self,
        name: str,
        path: str,
        args: Iterable[str] = (),
    ) -> str:
        """
        Invoke one Skill resource.

        Paths under scripts/ are executed with the given
        arguments; any other resource is returned as text.
        Output is truncated to the resource character limit.
        """
        record = self._record(name)

        resource = self._resolve_resource(
            record,
            path,
        )

        if (
            resource.relative_to(record.root).parts[0]
            == "scripts"
        ):
            return await self._run_script(
                record,
                resource,
                list(args),
            )

        return self._read_resource(resource)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _record(
        self,
        name: str,
    ) -> SkillRecord:
        record = self._registry.get_record(name)

        if record is None:
            raise UnknownSkillError(
                f"Unknown Skill: {name}"
            )

        return record

    @staticmethod
    def _resolve_resource(
        record: SkillRecord,
        path: str,
    ) -> Path:
        if not path.strip():
            raise SkillValidationError(
                "Resource path is empty."
            )

        resource = (
            record.root / path
        ).resolve()

        if not resource.is_relative_to(
            record.root
        ):
            raise SkillValidationError(
                f"Resource path escapes the "
                f"Skill root: {path}"
            )

        if not resource.is_file():
            raise SkillValidationError(
                f"Resource not found: {path}"
            )

        return resource

    def _read_resource(
        self,
        resource: Path,
    ) -> str:
        text = resource.read_text(
            encoding="utf-8",
            errors="replace",
        )

        return text[: self.resource_char_limit]

    async def _run_script(
        self,
        record: SkillRecord,
        script: Path,
        args: list[str],
    ) -> str:
        command = _SCRIPT_COMMANDS.get(
            script.suffix.lower(),
            (),
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                str(script),
                *args,
                cwd=record.root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

        except OSError as exc:
            return (
                f"{type(exc).__name__}: {exc}"
            )

        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=self.script_timeout,
            )

        except asyncio.TimeoutError:
            process.kill()

            await process.wait()

            return (
                f"Script timed out after "
                f"{self.script_timeout:g} seconds."
            )

        text = stdout.decode(
            "utf-8",
            errors="replace",
        )[: self.resource_char_limit]

        if process.returncode != 0:
            text = (
                f"{text}\n"
                f"[script exited with code "
                f"{process.returncode}]"
            )

        return text

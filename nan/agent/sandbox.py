"""
Execution-level sandbox for agent shell commands.

macOS wraps commands in ``sandbox-exec -f <profile>``; the
Seatbelt profile is generated once at core startup: write access
only to the workspace and temp directories, the nan package and
the builtin tree explicitly read-only, reads and network open.
Linux uses the bubblewrap equivalent. Services and the
autostart script run with user authority, outside the sandbox.

If no sandbox tool exists on the host, ``sandbox.required``
decides: true -- refuse to execute (raise); false -- degrade to
an unwrapped command with a warning, once.
"""

from __future__ import annotations

import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Literal

from loguru import logger


class SandboxUnavailableError(RuntimeError):
    """
    A sandbox is required but no sandbox tool exists on this
    host. The command is not executed.
    """


class Sandbox:
    """
    Wraps shell-verb commands. Value-ish object: one per core
    process, shared by every agent (the profile is global).
    """

    def __init__(
        self,
        *,
        workspace: Path,
        package_dir: Path,
        builtin_dir: Path | None = None,
        enabled: bool = True,
        required: bool = False,
        platform: str | None = None,
    ) -> None:
        self.workspace = workspace
        self.package_dir = package_dir
        self.builtin_dir = builtin_dir
        self.enabled = enabled
        self.required = required
        self.platform = platform or sys.platform

        self._tool: Literal["sandbox-exec", "bwrap"] | None = self._detect_tool()
        self._profile_path: Path | None = None
        self._warned = False

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_tool(self) -> Literal["sandbox-exec", "bwrap"] | None:
        if self.platform == "darwin" and shutil.which("sandbox-exec"):
            return "sandbox-exec"

        if self.platform.startswith("linux") and shutil.which("bwrap"):
            return "bwrap"

        return None

    @property
    def available(self) -> bool:
        return self._tool is not None

    # ------------------------------------------------------------------
    # Profile (macOS Seatbelt)
    # ------------------------------------------------------------------

    def ensure_profile(self) -> Path:
        """
        Generate the Seatbelt profile once and return its path.
        Called at core startup; also called lazily by wrap().
        """
        if self._profile_path is not None:
            return self._profile_path

        profile = _seatbelt_profile(
            workspace=self.workspace,
            package_dir=self.package_dir,
            builtin_dir=self.builtin_dir,
        )

        path = Path(tempfile.gettempdir()) / f"nan-sandbox-{id(self):x}.sb"

        path.write_text(profile, encoding="utf-8")

        self._profile_path = path

        return path

    # ------------------------------------------------------------------
    # Wrapping
    # ------------------------------------------------------------------

    def wrap(self, command: str) -> str:
        """
        The command the model asked to run, wrapped for
        execution. Raises SandboxUnavailableError when a sandbox
        is required but no tool exists.
        """
        if not self.enabled:
            if not self._warned:
                self._warned = True

                logger.warning("sandbox disabled by config; shell commands run UNSANDBOXED")

            return command

        if self._tool is None:
            if self.required:
                raise SandboxUnavailableError(
                    "sandbox.sandbox_required=true but no sandbox tool found "
                    "(sandbox-exec on macOS / bwrap on Linux); refusing to execute"
                )

            if not self._warned:
                self._warned = True

                logger.warning(
                    "No sandbox tool found (sandbox-exec/bwrap); shell commands run unsandboxed"
                )

            return command

        if self._tool == "sandbox-exec":
            profile = self.ensure_profile()

            return f"sandbox-exec -f {shlex.quote(str(profile))} bash -c {shlex.quote(command)}"

        return _bwrap_command(
            command,
            workspace=self.workspace,
            package_dir=self.package_dir,
            builtin_dir=self.builtin_dir,
        )


# ----------------------------------------------------------------------
# Profile / command builders
# ----------------------------------------------------------------------


def _seatbelt_profile(
    *,
    workspace: Path,
    package_dir: Path,
    builtin_dir: Path | None,
) -> str:
    """
    deny default; reads and network open; writes only to the
    workspace and temp locations; the nan package and builtin
    tree stay read-only even if a later edit broadens the write
    list (deny beats allow).
    """
    tmpdir = tempfile.gettempdir()

    write_paths = [
        workspace.resolve(),
        Path("/private/tmp"),
        Path("/var/folders"),
        Path(tmpdir).resolve(),
    ]

    lines = [
        "(version 1)",
        "(deny default)",
        "(allow file-read*)",
        "(allow file-write*",
    ]

    lines += [f'    (subpath "{p}")' for p in dict.fromkeys(str(p) for p in write_paths)]

    lines += [
        '    (literal "/dev/null")',
        '    (literal "/dev/dtrace-helper")',
        ")",
        "(deny file-write*",
        f'    (subpath "{package_dir.resolve()}")',
    ]

    if builtin_dir is not None:
        lines.append(f'    (subpath "{builtin_dir.resolve()}")')

    lines += [
        ")",
        "(allow network*)",
        "(allow process*)",
        "(allow file-ioctl)",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow system-socket)",
        "(allow iokit-get-properties)",
    ]

    return "\n".join(lines) + "\n"


def _bwrap_command(
    command: str,
    *,
    workspace: Path,
    package_dir: Path,
    builtin_dir: Path | None,
) -> str:
    """
    Bubblewrap equivalent: / read-only, /dev and /proc provided,
    /tmp and the workspace writable on top, network untouched.
    """
    parts = [
        "bwrap",
        "--ro-bind / /",
        "--dev-bind /dev /dev",
        "--proc /proc",
        "--tmpfs /tmp",
        f"--bind {shlex.quote(str(workspace.resolve()))} {shlex.quote(str(workspace.resolve()))}",
        # already ro via the / bind; pinned again here so it stays
        # readable even if the / bind changes
        f"--ro-bind {shlex.quote(str(package_dir.resolve()))} {shlex.quote(str(package_dir.resolve()))}",
    ]

    if builtin_dir is not None:
        parts.append(
            f"--ro-bind {shlex.quote(str(builtin_dir.resolve()))} "
            f"{shlex.quote(str(builtin_dir.resolve()))}"
        )

    parts += ["--", "bash", "-c", shlex.quote(command)]

    return " ".join(parts)

"""
Execution-level sandbox for agent shell commands.

macOS wraps commands in ``sandbox-exec -p <profile>``; the
Seatbelt profile is built in memory on every wrap: write access
only to the workspace and temp directories, the nan package and
the builtin tree explicitly read-only, reads and network open.
Linux uses the bubblewrap equivalent.

Two autostart scripts exist, with different authority:

    builtin/autostart.sh    maintained by the user, runs outside
                            this sandbox with the user's full
                            rights.
    workspace/autostart.sh  writable by agents and executed
                            inside this sandbox: an agent can set
                            up its own boot services, but only
                            within the same write scope as its
                            shell commands.

If no sandbox tool exists on the host, ``sandbox.required``
decides: true -- refuse to start; false -- degrade to an
unwrapped command with a warning, once.
"""

from __future__ import annotations

import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Literal

from loguru import logger

from nan.config import settings


class SandboxUnavailableError(RuntimeError):
    """
    A sandbox is required but no sandbox tool exists on this
    host. The command is not executed.
    """


class Sandbox:
    """
    Wraps shell-verb commands. Value-ish object: one per core
    process, shared by every agent (the profile is global).
    Reads the layout and policy from the process-wide settings.
    """

    def __init__(self) -> None:
        self._tool: Literal["sandbox-exec", "bwrap"] | None = self._detect_tool()
        self._warned = False

        if self.required and self._tool is None:
            raise SandboxUnavailableError(
                "sandbox.required=true but no sandbox tool found "
                "(sandbox-exec on macOS / bwrap on Linux); refusing to start"
            )

    # ------------------------------------------------------------------
    # Policy / layout (live reads of the global settings)
    # ------------------------------------------------------------------

    @property
    def workspace(self) -> Path:
        return settings.paths.workspace

    @property
    def package_dir(self) -> Path:
        return settings.paths.package

    @property
    def builtin_dir(self) -> Path:
        return settings.paths.builtin

    @property
    def required(self) -> bool:
        return settings.sandbox.required

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_tool(self) -> Literal["sandbox-exec", "bwrap"] | None:
        if sys.platform == "darwin" and shutil.which("sandbox-exec"):
            return "sandbox-exec"

        if sys.platform.startswith("linux") and shutil.which("bwrap"):
            return "bwrap"

        return None

    @property
    def available(self) -> bool:
        return self._tool is not None

    # ------------------------------------------------------------------
    # Wrapping
    # ------------------------------------------------------------------

    def wrap(self, command: str) -> str:
        """
        The command the model asked to run, wrapped for
        execution.
        """
        if self._tool is None:
            if not self._warned:
                self._warned = True

                logger.warning(
                    "No sandbox tool found (sandbox-exec/bwrap); shell commands run unsandboxed"
                )

            return command

        if self._tool == "sandbox-exec":
            # -p takes the profile inline.
            profile = _seatbelt_profile(
                workspace=self.workspace,
                package_dir=self.package_dir,
                builtin_dir=self.builtin_dir,
            )

            return f"sandbox-exec -p {shlex.quote(profile)} bash -c {shlex.quote(command)}"

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
    builtin_dir: Path,
) -> str:
    """
    deny default; reads and network open; writes only to the
    workspace and temp locations; the nan package and builtin
    tree stay read-only.
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
        f'    (subpath "{builtin_dir.resolve()}")',
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
    builtin_dir: Path,
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
        f"--ro-bind {shlex.quote(str(package_dir.resolve()))} {shlex.quote(str(package_dir.resolve()))}",
    ]

    if builtin_dir.exists():
        parts.append(
            f"--ro-bind {shlex.quote(str(builtin_dir.resolve()))} "
            f"{shlex.quote(str(builtin_dir.resolve()))}"
        )

    parts += ["--", "bash", "-c", shlex.quote(command)]

    return " ".join(parts)

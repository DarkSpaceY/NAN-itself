from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nan.agent.verbs import Verb, VerbOutcome, format_shell_output
from nan.execution import Job, kill_group
from nan.llm.types import VerbDefinition

if TYPE_CHECKING:
    from nan.agent.core import Agent

# How long a background command gets to prove it started before the
# call reports "started" or its immediate exit.
_BACKGROUND_GRACE = 0.1

# How long a cancelled command gets between SIGTERM and SIGKILL.
_CANCEL_GRACE = 1.0


class ShellVerb(Verb):
    name = "shell"

    def definition(self) -> VerbDefinition:
        return VerbDefinition(
            name=self.name,
            description=(
                "Run a bash command in your workspace. Its cwd is the "
                "workspace root unless you pass 'cwd' (relative to the "
                "root and it must stay inside it; prefer 'cwd' over a "
                "leading 'cd'). Commands are non-interactive and have a "
                "fixed timeout set by the user, so a genuinely long task "
                "must pass background=true: the call then returns a job "
                "id, pid and log path at once, and later shell calls poll "
                "the log. Output is truncated (head and tail kept)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command."},
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Directory to run in, relative to the workspace "
                            "root; it must stay inside the workspace."
                        ),
                    },
                    "env": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Extra environment variables for this command.",
                    },
                    "stdin": {
                        "type": "string",
                        "description": (
                            "Text written once to the command's stdin. Not allowed with background."
                        ),
                    },
                    "background": {
                        "type": "boolean",
                        "description": (
                            "Start the command in the background and return "
                            "immediately with its job id, pid and log path."
                        ),
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        )

    async def execute(self, *, call: Any, agent: Agent) -> VerbOutcome:
        arguments = call.arguments

        command = arguments.get("command")

        if not isinstance(command, str) or not command.strip():
            return VerbOutcome("shell requires a non-empty 'command'.")

        cwd, error = _cwd(agent, arguments.get("cwd"))

        if error is not None:
            return VerbOutcome(error)

        env, error = _environment(arguments.get("env"))

        if error is not None:
            return VerbOutcome(error)

        stdin = arguments.get("stdin")

        if stdin is not None and not isinstance(stdin, str):
            return VerbOutcome("'stdin' must be a string.")

        background = arguments.get("background", False)

        if not isinstance(background, bool):
            return VerbOutcome("'background' must be a boolean.")

        if background and stdin is not None:
            return VerbOutcome(
                "'stdin' cannot be used with background: the command runs without input."
            )

        # Command-level deny list: a hit is reported back as the
        # verb result so the model can rewrite the command.
        refused = agent.sandbox.refusal_reason(command)

        if refused is not None:
            return VerbOutcome(
                f"Refused by sandbox rule: {refused}. The command was not executed; rewrite it.",
                {"refused": True, "reason": refused},
            )

        # Execution-level sandbox: the command the model wrote is
        # wrapped by the sandbox profile. wrap() may raise when
        # required but unavailable.
        command = agent.sandbox.wrap(command)

        if background:
            return await self._background(agent, command, cwd, env)

        return await self._foreground(agent, command, cwd, env, stdin)

    async def _foreground(
        self,
        agent: Agent,
        command: str,
        cwd: Path,
        env: dict[str, str] | None,
        stdin: str | None,
    ) -> VerbOutcome:
        """Run the command and wait; the fixed timeout is the user's."""
        timeout = agent.settings.agent.shell_timeout
        limit = agent.settings.agent.shell_output_limit

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                env=env,
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                # Own session: the whole tree can be signalled.
                start_new_session=True,
            )

            collector = _Collector()
            reader = asyncio.create_task(collector.pump(proc.stdout))
            waited = asyncio.create_task(proc.wait())
            feeder = (
                asyncio.create_task(_feed(proc.stdin, stdin))
                if stdin is not None and proc.stdin is not None
                else None
            )

            try:
                # EOF is the end of the command: its exit closes the
                # pipe. A leftover child still holding it counts as
                # overrunning the timeout.
                await asyncio.wait_for(reader, timeout=timeout)

                await waited

                if feeder is not None:
                    await feeder

            except TimeoutError:
                kill_group(proc.pid, signal.SIGKILL)

                await waited

                if feeder is not None:
                    feeder.cancel()

                    with contextlib.suppress(asyncio.CancelledError):
                        await feeder

                # The output read so far survives the cancelled reader.
                return VerbOutcome(
                    f"{format_shell_output(collector.text, proc.returncode, limit)}\n\n"
                    f"Timed out after {timeout:g}s (command killed).",
                    {"timed_out": True},
                )

            except asyncio.CancelledError:
                # Pause/death cancellation kills the child's
                # process tree.
                kill_group(proc.pid, signal.SIGTERM)

                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(waited, timeout=_CANCEL_GRACE)

                if proc.returncode is None:
                    kill_group(proc.pid, signal.SIGKILL)

                for task in (reader, feeder):
                    if task is not None:
                        task.cancel()

                raise

        except Exception as exc:
            return VerbOutcome(f"{type(exc).__name__}: {exc}")

        text = format_shell_output(collector.text, proc.returncode, limit)

        return VerbOutcome(text, {"exit_code": proc.returncode, "timed_out": False})

    async def _background(
        self,
        agent: Agent,
        command: str,
        cwd: Path,
        env: dict[str, str] | None,
    ) -> VerbOutcome:
        """
        Start the command without waiting. Its output goes to a log
        file the model can poll; the job is registered on the agent
        so death and shutdown can kill it.
        """
        workspace = agent.settings.paths.workspace
        jobs_dir = workspace / ".nan" / "jobs" / agent.agent_id

        job_id = agent.jobs.new_id()
        log_path = jobs_dir / f"{job_id}.log"

        try:
            jobs_dir.mkdir(parents=True, exist_ok=True)

            with open(log_path, "wb") as log:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    cwd=str(cwd),
                    env=env,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )

            # A command that dies at once (a typo, a missing
            # binary) must say so, not pretend it started.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=_BACKGROUND_GRACE)

        except Exception as exc:
            return VerbOutcome(f"{type(exc).__name__}: {exc}")

        if proc.returncode is not None:
            output = _read_log(log_path)
            limit = agent.settings.agent.shell_output_limit

            return VerbOutcome(
                "The command exited immediately.\n"
                f"{format_shell_output(output, proc.returncode, limit)}",
                {"exit_code": proc.returncode, "timed_out": False},
            )

        agent.jobs.add(Job(job_id=job_id, proc=proc, pgid=proc.pid, log_path=log_path))

        relative = log_path.relative_to(workspace)

        return VerbOutcome(
            f"Started {job_id} (pid {proc.pid}, pgid {proc.pid}). "
            f"Log: {relative}\n"
            f"Poll the log with a later shell call; stop it with "
            f"kill -TERM -- -{proc.pid}.",
            {"job_id": job_id, "pid": proc.pid, "pgid": proc.pid},
        )


# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------


def _cwd(agent: Agent, value: Any) -> tuple[Path, str | None]:
    """
    The command's cwd: the workspace root by default, and a
    directory inside it when asked for.
    """
    workspace = agent.settings.paths.workspace.resolve()

    if value is None:
        return workspace, None

    if not isinstance(value, str) or not value.strip():
        return workspace, "'cwd' must be a non-empty string."

    target = Path(value)

    resolved = (target if target.is_absolute() else workspace / target).resolve()

    if not resolved.is_relative_to(workspace):
        return workspace, f"'cwd' must stay inside the workspace: {resolved}"

    if not resolved.is_dir():
        return workspace, f"'cwd' is not a directory: {resolved}"

    return resolved, None


def _environment(value: Any) -> tuple[dict[str, str] | None, str | None]:
    """
    The command's environment: the inherited one plus the model's
    overlay. None means "inherit untouched".
    """
    if value is None:
        return None, None

    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        return None, "'env' must be an object of string values."

    return {**os.environ, **value}, None


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------


class _Collector:
    """Output read so far; it survives the reader being cancelled."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []

    async def pump(self, stream: asyncio.StreamReader | None) -> None:
        """Drain a pipe to EOF; a kill ends it like a normal exit."""
        if stream is None:
            return

        while True:
            chunk = await stream.read(65536)

            if not chunk:
                return

            self._chunks.append(chunk)

    @property
    def text(self) -> str:
        return b"".join(self._chunks).decode("utf-8", errors="replace")


async def _feed(stream: asyncio.StreamWriter, text: str) -> None:
    """Write the one-shot stdin; a command that never reads it is fine."""
    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
        stream.write(text.encode("utf-8"))

        await stream.drain()

    stream.close()


def _read_log(path: Path) -> str:
    with contextlib.suppress(OSError):
        return path.read_bytes().decode("utf-8", errors="replace")

    return ""

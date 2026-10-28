from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from typing import TYPE_CHECKING, Any

from nan.agent.verbs import Verb, VerbOutcome, truncate_output
from nan.config import settings
from nan.llm import ToolDefinition

if TYPE_CHECKING:
    from nan.agent.core import Agent


def _kill_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    """
    Signal the command's whole process group. The command starts
    in its own session, so its children (subshells, background
    jobs) die with it.
    """
    # ProcessLookupError: already gone, nothing left to kill.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), sig)


class ShellVerb(Verb):
    name = "shell"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Run a bash command in your workspace (the command's "
                "cwd is the workspace root; keep all file activity "
                "under it). Commands have a fixed timeout; for "
                "genuinely long tasks do NOT block: launch in "
                'background, e.g. "nohup python job.py > job.log '
                '2>&1 &" then poll job.log with later shell calls. '
                "Output is truncated (head+tail kept)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command."},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        )

    async def execute(self, *, call: Any, agent: Agent) -> VerbOutcome:
        command = call.arguments.get("command")

        if not isinstance(command, str) or not command.strip():
            return VerbOutcome("shell requires a non-empty 'command'.")

        timeout = settings.agent.shell_timeout

        # Execution-level sandbox: the command the model wrote is
        # wrapped by the sandbox profile. wrap() may raise when
        # required but unavailable.
        command = agent.sandbox.wrap(command)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(settings.paths.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                # Own session: the whole tree can be signalled.
                start_new_session=True,
            )

            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            except TimeoutError:
                _kill_group(proc, signal.SIGKILL)

                await proc.wait()

                return VerbOutcome(
                    f"Timed out after {timeout:g}s (command killed).",
                    {"timed_out": True},
                )

            except asyncio.CancelledError:
                # Pause/death cancellation kills the child's
                # process tree.
                _kill_group(proc, signal.SIGTERM)

                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=1.0)

                if proc.returncode is None:
                    _kill_group(proc, signal.SIGKILL)

                raise

        except Exception as exc:
            return VerbOutcome(f"{type(exc).__name__}: {exc}")

        output = stdout.decode("utf-8", errors="replace")

        text = truncate_output(output, proc.returncode, settings.agent.shell_output_limit)

        return VerbOutcome(text, {"exit_code": proc.returncode, "timed_out": False})

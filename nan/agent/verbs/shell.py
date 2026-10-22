from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from nan.agent.verbs import Verb, truncate_output
from nan.llm import ToolDefinition

if TYPE_CHECKING:
    from nan.agent.core import Agent


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
                "background, e.g. \"nohup python job.py > job.log "
                "2>&1 &\" then poll job.log with later shell calls. "
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

    async def execute(self, *, call: Any, agent: Agent) -> str:
        command = call.arguments.get("command")

        if not isinstance(command, str) or not command.strip():
            return "shell requires a non-empty 'command'."

        # The timeout comes from the user's configuration.

        timeout = agent.config.shell_timeout

        # Execution-level sandbox: the command the model wrote is
        # wrapped by the sandbox profile. wrap() may raise
        # (required but unavailable): refusing to execute is the
        # point.
        command = agent.sandbox.wrap(command)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(agent.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            except TimeoutError:
                proc.kill()

                await proc.wait()

                return f"Timed out after {timeout:g}s (command killed)."

            except asyncio.CancelledError:
                # Pause/death cancelling the turn must not leave
                # the child running.
                proc.terminate()

                raise

        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

        output = stdout.decode("utf-8", errors="replace")

        return truncate_output(output, proc.returncode, agent.config.shell_output_limit)

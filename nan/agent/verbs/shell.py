from __future__ import annotations

import asyncio
import contextlib
import fcntl
import os
import pty
import termios
import tty
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nan.agent.verbs import Verb, VerbOutcome
from nan.config.models import Settings
from nan.execution import (
    Job,
    home_environment,
    job_directory,
    read_output,
    start_pump,
    write_stdin,
)
from nan.llm.types import VerbDefinition

if TYPE_CHECKING:
    from nan.agent.core import Agent

# How long a command gets to prove it finished before the call hands
# it to the background (or reports an immediate exit for an explicit
# background start). A pty spawn plus shell startup costs ~50-120ms on
# a loaded machine, so the window needs headroom over that.
_BACKGROUND_GRACE = 0.3


def _own_terminal() -> None:
    """Take the pty as controlling terminal; setsid already ran."""
    fcntl.ioctl(1, termios.TIOCSCTTY, 0)


class ShellVerb(Verb):
    name = "shell"

    def definition(self, settings: Settings) -> VerbDefinition:
        wait = f"{settings.agent.shell_timeout:g}s"
        limit = f"{settings.agent.shell_output_limit:,}"

        return VerbDefinition(
            name=self.name,
            description=(
                "Run a bash command in your workspace. Its cwd is the "
                "workspace root unless you pass 'cwd' (relative to the "
                "root and it must stay inside it; prefer 'cwd' over a "
                "leading 'cd').\n"
                f"Commands are non-interactive on the terminal, but their "
                f"stdin stays open: a command that reads stdin waits until "
                f"you write to it or send EOF with the 'jobs' verb. A command "
                f"that finishes within {wait} returns its output and exit "
                f"code directly. A command still running after {wait} keeps "
                "going as a background job: the call returns its job id and "
                "log path, and the 'jobs' verb can list it, read new output, "
                "write its stdin, wait for it or stop it. Pass background=true "
                "when you know the command should outlive the call -- for "
                "example a server or debugger you keep open and inspect "
                "across several steps.\n"
                f"Output is truncated to {limit} characters (head and tail "
                "kept), so write long output to a file and read it back with "
                "grep/head. Prefer chaining shell commands together to "
                "reduce the number of calls."
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
                            "Text written to the command's stdin when it starts. "
                            "The pipe stays open; send more, or EOF, with jobs input."
                        ),
                    },
                    "background": {
                        "type": "boolean",
                        "description": (
                            "Return at once with a job id instead of waiting. "
                            "Use it for servers and debuggers you keep open."
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

        env, error = _environment(
            arguments.get("env"), agent.agent_id, agent.settings.paths.workspace
        )

        if error is not None:
            return VerbOutcome(error)

        stdin = arguments.get("stdin")

        if stdin is not None and not isinstance(stdin, str):
            return VerbOutcome("'stdin' must be a string.")

        background = arguments.get("background", False)

        if not isinstance(background, bool):
            return VerbOutcome("'background' must be a boolean.")

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
        wrapped = agent.sandbox.wrap(command)

        timeout = agent.settings.agent.shell_timeout

        try:
            job = await _start_job(agent, wrapped, command, cwd, env, stdin)
        except Exception as exc:
            return VerbOutcome(f"{type(exc).__name__}: {exc}")

        if background:
            try:
                # A command that dies at once (a typo, a missing
                # binary) must say so, not pretend it started.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(job.proc.wait(), timeout=_BACKGROUND_GRACE)
            except asyncio.CancelledError:
                # A cancelled start has no id to report; leaving it
                # alive would be a job nobody can poll or stop.
                await agent.jobs.stop(job.job_id)

                raise

            if not job.running:
                return await _completed(agent, job, immediate=True)

            return _job_outcome(agent, job, promoted=False)

        try:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(job.proc.wait(), timeout=timeout)
        except asyncio.CancelledError:
            # A job belongs to the agent: a cancelled turn leaves it
            # running, exactly like an explicit background start.
            raise

        if not job.running:
            return await _completed(agent, job, immediate=False)

        return _job_outcome(agent, job, promoted=True, waited=timeout)


async def _start_job(
    agent: Agent,
    command: str,
    display: str,
    cwd: Path,
    env: dict[str, str] | None,
    stdin: str | None,
) -> Job:
    """
    Start the command on its own pty. The core keeps the master open:
    an ungraceful core death hangs up the job's session, and the pump
    copies the pty into the log.
    """
    workspace = agent.settings.paths.workspace
    jobs_dir = job_directory(workspace, agent.agent_id)

    job_id = agent.jobs.new_id()
    log_path = jobs_dir / f"{job_id}.log"

    job: Job | None = None
    master = -1
    slave = -1

    try:
        jobs_dir.mkdir(parents=True, exist_ok=True)

        # Raw mode: no echo and no CR translation, so the log holds
        # exactly what the command wrote.
        master, slave = pty.openpty()
        tty.setraw(slave)

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            env=env,
            # A pipe that stays open for the job's life: a command that
            # reads stdin waits for `jobs input` (or EOF) instead of
            # seeing end-of-file at once.
            stdin=asyncio.subprocess.PIPE,
            stdout=slave,
            stderr=slave,
            start_new_session=True,
            preexec_fn=_own_terminal,
        )

        os.close(slave)
        slave = -1

        job = Job(
            job_id=job_id,
            proc=proc,
            pgid=proc.pid,
            log_path=log_path,
            master=master,
            pump=start_pump(master, log_path, agent.settings.agent.job_log_limit),
            command=display,
            stdin=proc.stdin,
        )
        master = -1

        # Registered before any await: every cancellation point from
        # here on has a job that stop_all can kill.
        agent.jobs.add(job)

        if stdin is not None:
            await write_stdin(job, stdin)

        return job

    except BaseException:
        if job is not None:
            await agent.jobs.stop(job_id)

        raise

    finally:
        if slave >= 0:
            os.close(slave)

        if master >= 0:
            os.close(master)


async def _completed(agent: Agent, job: Job, *, immediate: bool) -> VerbOutcome:
    """The command exited within the window; hand back its output."""
    limit = agent.settings.agent.shell_output_limit

    # The pty may still hold the final output; drain it with a bound.
    if not job.pump.done():
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(job.pump), timeout=_BACKGROUND_GRACE)

    output = read_output(job.log_path, limit)
    code = job.proc.returncode

    agent.jobs.discard(job.job_id)

    prefix = "The command exited immediately.\n" if immediate else ""

    return VerbOutcome(f"{prefix}exit code: {code}\n{output}", {"exit_code": code})


def _job_outcome(
    agent: Agent, job: Job, *, promoted: bool, waited: float | None = None
) -> VerbOutcome:
    """The command is still running; hand back its job."""
    relative = job.log_path.relative_to(agent.settings.paths.workspace)

    if promoted:
        intro = (
            f"Still running after {waited:g}s; it continues as job {job.job_id} in the background. "
        )
    else:
        intro = f"Started {job.job_id}. "

    return VerbOutcome(
        f"{intro}pid {job.proc.pid}, pgid {job.pgid}. Log: {relative}\n"
        f"Use the 'jobs' verb to work with it: list, logs {job.job_id}, "
        f"wait {job.job_id}, input {job.job_id} (write stdin or EOF), "
        f"stop {job.job_id}.",
        {"job_id": job.job_id, "pid": job.proc.pid, "pgid": job.pgid, "promoted": promoted},
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


def _environment(
    value: Any, agent_id: str, workspace: Path
) -> tuple[dict[str, str] | None, str | None]:
    """
    The command's environment: the inherited one, then a quiet
    terminal (the pty would otherwise invite colors and progress
    bars into the log), then the model's overlay, then the core's
    own settings last so a command cannot impersonate another agent
    or reach the user's home. `<workspace>/bin` and the agent's
    `~/.local/bin` trail PATH: tools there are callable by bare
    name, but can never shadow a system command (the model can write
    into the workspace).
    """
    if value is not None and (
        not isinstance(value, dict)
        or not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items())
    ):
        return None, "'env' must be an object of string values."

    overlay = value if isinstance(value, dict) else {}
    env = {
        **os.environ,
        "TERM": "dumb",
        "NO_COLOR": "1",
        "CLICOLOR": "0",
        **overlay,
        **home_environment(workspace),
        "NAN_AGENT_ID": agent_id,
    }

    (workspace / "home").mkdir(parents=True, exist_ok=True)

    trailing = [workspace / "home" / ".local" / "bin", workspace / "bin"]

    env["PATH"] = os.pathsep.join([env.get("PATH", ""), *(str(p) for p in trailing)])

    return env, None

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

from nan.agent.verbs import Verb, VerbOutcome
from nan.execution import Job, close_stdin, read_slice, write_stdin
from nan.llm.types import VerbDefinition

if TYPE_CHECKING:
    from nan.agent.core import Agent
    from nan.config.models import Settings

_ACTIONS = ("list", "status", "logs", "wait", "input", "stop")

_DEFAULT_BYTES = 8_000
_MAX_BYTES = 64_000
_DEFAULT_WAIT = 10.0
_MAX_WAIT = 300.0


class JobsVerb(Verb):
    """The model's console for the commands that kept running."""

    name = "jobs"

    def definition(self, _settings: Settings) -> VerbDefinition:
        return VerbDefinition(
            name=self.name,
            description=(
                "Work with the background jobs your shell calls started. "
                "'list' shows every job you started in this session, running "
                "or finished; 'logs' returns new output from a byte offset; "
                "'wait' blocks until a job exits or the timeout; 'input' "
                "writes to a job's stdin, or sends EOF by closing it; 'stop' "
                "kills its process group. Jobs belong to you: they outlive "
                "the turn that started them, and a pause does not kill them."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": list(_ACTIONS),
                        "description": "What to do.",
                    },
                    "job_id": {
                        "type": "string",
                        "description": "The job's short id, e.g. j1. Required except for 'list'.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "logs: byte offset to read from (default 0).",
                    },
                    "max_bytes": {
                        "type": "integer",
                        "description": (
                            f"logs: most bytes to return this time "
                            f"(default {_DEFAULT_BYTES}, max {_MAX_BYTES})."
                        ),
                    },
                    "timeout": {
                        "type": "number",
                        "description": (
                            f"wait: seconds to block before reporting it is "
                            f"still running (default {_DEFAULT_WAIT:g}, "
                            f"max {_MAX_WAIT:g})."
                        ),
                    },
                    "text": {
                        "type": "string",
                        "description": (
                            "input: text written to the job's stdin. The pipe "
                            "stays open afterwards unless eof is true; include "
                            "a trailing newline if the command expects a line."
                        ),
                    },
                    "eof": {
                        "type": "boolean",
                        "description": (
                            "input: close the job's stdin after writing text, "
                            "so its next read sees end-of-file."
                        ),
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        )

    async def execute(self, *, call: Any, agent: Agent) -> VerbOutcome:
        arguments = call.arguments

        action = arguments.get("action")

        if not isinstance(action, str) or action not in _ACTIONS:
            return VerbOutcome(f"'action' must be one of: {', '.join(_ACTIONS)}.")

        if action == "list":
            return _list(agent)

        job_id = arguments.get("job_id")

        if not isinstance(job_id, str) or not job_id.strip():
            return VerbOutcome(f"'job_id' is required for '{action}'.")

        job = agent.jobs.get(job_id)

        if job is None:
            return VerbOutcome(f"Unknown job id: {job_id}. Run jobs list to see your jobs.")

        if action == "status":
            return _status(agent, job)

        if action == "logs":
            return _logs(agent, job, arguments)

        if action == "wait":
            return await _wait(agent, job, arguments)

        if action == "input":
            return await _input(job, arguments)

        return await _stop(agent, job)


def _list(agent: Agent) -> VerbOutcome:
    jobs = sorted(agent.jobs.all, key=lambda job: job.started_at)

    if not jobs:
        return VerbOutcome("No jobs: everything you started has already returned.")

    lines = [
        f"{job.job_id:<4} {job.state:<10} {_age(job):>7}  $ {_preview(job.command)}" for job in jobs
    ]

    return VerbOutcome("\n".join(lines), {"action": "list"})


def _status(agent: Agent, job: Job) -> VerbOutcome:
    lines = [
        f"{job.job_id} {job.state}",
        f"  command: {_preview(job.command, 200)}",
        f"  pid {job.proc.pid}, pgid {job.pgid}",
        f"  started {_age(job)} ago",
    ]

    if not job.running:
        lines.append(f"  ended {_age(job, since_end=True)} ago, exit {job.proc.returncode}")

    lines.append(f"  log: {_relative(agent, job)}")
    lines.append(f"  output: {_size(job)} bytes")

    return VerbOutcome("\n".join(lines), {"action": "status", "job_id": job.job_id})


def _logs(agent: Agent, job: Job, arguments: dict[str, Any]) -> VerbOutcome:
    offset = arguments.get("offset", 0)
    max_bytes = arguments.get("max_bytes", _DEFAULT_BYTES)

    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        return VerbOutcome("'offset' must be an integer >= 0.")

    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        return VerbOutcome("'max_bytes' must be a positive integer.")

    max_bytes = min(max_bytes, _MAX_BYTES)

    size = _size(job)

    if offset >= size:
        return VerbOutcome(
            f"{job.job_id} has no new output (offset {offset} of {size}).",
            {"action": "logs", "job_id": job.job_id, "offset": size, "size": size},
        )

    chunk, next_offset, size = read_slice(job.log_path, offset, max_bytes)

    return VerbOutcome(
        f"{job.job_id} output [offset {offset} -> {next_offset} of {size}]\n{chunk}",
        {"action": "logs", "job_id": job.job_id, "offset": next_offset, "size": size},
    )


async def _wait(agent: Agent, job: Job, arguments: dict[str, Any]) -> VerbOutcome:
    timeout = arguments.get("timeout", _DEFAULT_WAIT)

    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout < 0:
        return VerbOutcome("'timeout' must be a number >= 0.")

    timeout = min(float(timeout), _MAX_WAIT)

    if job.running:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(job.proc.wait(), timeout=timeout)

    if job.running:
        return VerbOutcome(
            f"{job.job_id} is still running after {timeout:g}s (pid {job.proc.pid}).",
            {"action": "wait", "job_id": job.job_id, "exited": False},
        )

    return VerbOutcome(
        f"{job.job_id} exited {job.proc.returncode} after {_runtime(job):.1f}s.",
        {
            "action": "wait",
            "job_id": job.job_id,
            "exited": True,
            "exit_code": job.proc.returncode,
        },
    )


async def _input(job: Job, arguments: dict[str, Any]) -> VerbOutcome:
    text = arguments.get("text")
    eof = arguments.get("eof", False)

    if text is not None and not isinstance(text, str):
        return VerbOutcome("'text' must be a string.")

    if not isinstance(eof, bool):
        return VerbOutcome("'eof' must be a boolean.")

    if not text and not eof:
        return VerbOutcome("'input' needs 'text', 'eof': true, or both.")

    if not job.running:
        return VerbOutcome(
            f"{job.job_id} has exited {job.proc.returncode}; its stdin is closed.",
            {
                "action": "input",
                "job_id": job.job_id,
                "written": 0,
                "eof": False,
            },
        )

    written = 0

    if text:
        if not await write_stdin(job, text):
            return VerbOutcome(
                f"{job.job_id} is no longer reading stdin.",
                {
                    "action": "input",
                    "job_id": job.job_id,
                    "written": 0,
                    "eof": False,
                },
            )

        written = len(text.encode("utf-8"))

    if eof:
        close_stdin(job)

    if written and eof:
        message = f"Wrote {written} bytes to {job.job_id}'s stdin, then closed it (EOF)."
    elif written:
        message = f"Wrote {written} bytes to {job.job_id}'s stdin."
    else:
        message = f"Closed {job.job_id}'s stdin (EOF)."

    return VerbOutcome(
        message,
        {"action": "input", "job_id": job.job_id, "written": written, "eof": eof},
    )


async def _stop(agent: Agent, job: Job) -> VerbOutcome:
    was_running = job.running

    stopped = await agent.jobs.stop(job.job_id)

    if stopped is None:
        return VerbOutcome(f"Unknown job id: {job.job_id}. Run jobs list to see your jobs.")

    if not was_running:
        return VerbOutcome(
            f"{job.job_id} had already exited {job.proc.returncode}.",
            {"action": "stop", "job_id": job.job_id, "exit_code": job.proc.returncode},
        )

    return VerbOutcome(
        f"Stopped {job.job_id} ({job.state}).",
        {"action": "stop", "job_id": job.job_id, "exit_code": job.proc.returncode},
    )


def _preview(command: str, limit: int = 80) -> str:
    line = command.strip().splitlines()[0] if command.strip() else ""

    return line if len(line) <= limit else f"{line[:limit]}…"


def _runtime(job: Job) -> float:
    return (job.ended_at or time.time()) - job.started_at


def _age(job: Job, *, since_end: bool = False) -> str:
    seconds = _runtime(job) if not since_end else time.time() - (job.ended_at or time.time())

    if seconds < 60:
        return f"{seconds:.0f}s"

    minutes = seconds / 60

    if minutes < 60:
        return f"{minutes:.0f}m"

    return f"{minutes / 60:.1f}h"


def _relative(agent: Agent, job: Job) -> str:
    return str(job.log_path.relative_to(agent.settings.paths.workspace))


def _size(job: Job) -> int:
    try:
        return job.log_path.stat().st_size
    except OSError:
        return 0

"""
Background jobs, owned by the agent that started them.

A foreground command dies with the turn that ran it; a background
job outlives the turn and belongs to the agent. The registry is
the agent's list of live jobs: on death and shutdown it kills
their process groups, so no command outlives the agent that
started it. Pause leaves jobs alone -- a pause cancels the
in-flight turn, not the agent's processes.

A job's stdout and stderr go to its log file under
``<workspace>/.nan/jobs/<agent_id>/``; the model polls it with
later shell calls.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from dataclasses import dataclass
from pathlib import Path

# How long a job's group gets between SIGTERM and SIGKILL.
_STOP_GRACE = 1.0


def kill_group(pgid: int, sig: int) -> None:
    """Signal a whole process group; an already-dead group is nothing to do."""
    # ProcessLookupError: the group is already gone.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, sig)


@dataclass
class Job:
    """One background command, still owned by its agent."""

    job_id: str
    proc: asyncio.subprocess.Process
    pgid: int
    log_path: Path

    @property
    def running(self) -> bool:
        return self.proc.returncode is None


class JobRegistry:
    """The agent's live jobs, in start order. Not shared between agents."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._next = 0

    def new_id(self) -> str:
        self._next += 1

        return f"j{self._next}"

    def add(self, job: Job) -> None:
        # A long-lived agent starts many jobs; finished ones do not
        # need to be kept.
        self._forget_exited()

        self._jobs[job.job_id] = job

    @property
    def live(self) -> list[Job]:
        return [job for job in self._jobs.values() if job.running]

    async def stop_all(self) -> None:
        """
        Kill every live job: SIGTERM the group, then SIGKILL the
        survivors. Called on death and shutdown, never on pause.
        """
        jobs = list(self._jobs.values())

        self._jobs.clear()

        pending = [job for job in jobs if job.running]

        for job in pending:
            kill_group(job.pgid, signal.SIGTERM)

        waiting = [asyncio.create_task(job.proc.wait()) for job in pending]

        if waiting:
            await asyncio.wait(waiting, timeout=_STOP_GRACE)

        for job in pending:
            if job.proc.returncode is None:
                kill_group(job.pgid, signal.SIGKILL)

        if waiting:
            await asyncio.gather(*waiting, return_exceptions=True)

    def _forget_exited(self) -> None:
        for job_id in [job.job_id for job in self._jobs.values() if not job.running]:
            self._jobs.pop(job_id, None)

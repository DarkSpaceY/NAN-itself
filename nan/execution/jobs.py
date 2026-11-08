"""
Background jobs, owned by the agent that started them.

A foreground command dies with the turn that ran it; a background
job outlives the turn and belongs to the agent. The registry is
the agent's list of live jobs: on death and shutdown it kills
their process groups, so no command outlives the agent that
started it. Pause leaves jobs alone -- a pause cancels the
in-flight turn, not the agent's processes.

Each job runs on its own pty and the agent keeps the master open.
A core that dies without unwinding (crash, SIGKILL) closes that
master, and the kernel hangs up the job's session the way a
closing terminal does -- so an ungraceful death leaks nothing.

The pty is pumped into the job's log file under
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
    master: int
    pump: asyncio.Task[None]

    @property
    def running(self) -> bool:
        return self.proc.returncode is None

    def close(self) -> None:
        """Drop the pty and its pump; the process is gone or being killed."""
        if not self.pump.done():
            self.pump.cancel()
        elif not self.pump.cancelled():
            self.pump.exception()

        with contextlib.suppress(OSError):
            os.close(self.master)


async def _pump(master: int, log_path: Path) -> None:
    """Copy the pty into the log until the command closes its side."""
    try:
        stream = os.fdopen(os.dup(master), "rb", buffering=0)
    except OSError:
        return

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    transport = None

    try:
        transport, _ = await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), stream
        )

        with open(log_path, "wb", buffering=0) as log:
            while True:
                try:
                    chunk = await reader.read(65536)
                except OSError:
                    return

                if not chunk:
                    return

                log.write(chunk)

    finally:
        if transport is None:
            stream.close()
        else:
            transport.close()


def start_pump(master: int, log_path: Path) -> asyncio.Task[None]:
    """Start copying the job's pty into its log."""
    return asyncio.create_task(_pump(master, log_path), name=f"job-pump:{log_path.stem}")


async def _terminate(jobs: list[Job]) -> None:
    """SIGTERM the groups, SIGKILL the survivors, then close every pty."""
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

    for job in jobs:
        if not job.pump.done():
            job.pump.cancel()

    if jobs:
        await asyncio.gather(*(job.pump for job in jobs), return_exceptions=True)

    for job in jobs:
        job.close()


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

        await _terminate(jobs)

    async def stop(self, job_id: str) -> None:
        """Kill one job; a start that was cancelled has no id to report."""
        job = self._jobs.pop(job_id, None)

        if job is not None:
            await _terminate([job])

    def discard(self, job_id: str) -> None:
        """Drop a job that already exited; its pty goes with it."""
        job = self._jobs.pop(job_id, None)

        if job is not None:
            job.close()

    def _forget_exited(self) -> None:
        for job_id, job in list(self._jobs.items()):
            if job.running:
                continue

            self._jobs.pop(job_id, None)
            job.close()

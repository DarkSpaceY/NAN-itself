"""
Jobs, owned by the agent that started them.

Every shell command runs as a job on its own pty: a command that
finishes inside the inline window returns its output directly, one
that outlives it stays behind as a background job. A job outlives
its turn and belongs to the agent. The registry is the agent's list
of jobs: on death and shutdown it kills their process groups, so no
command outlives the agent that started it. Pause leaves jobs alone
-- a pause cancels the in-flight turn, not the agent's processes.

Each job's pty is pumped into a log file under
``<workspace>/.nan/jobs/<agent_id>/``. When the command ends, a
watcher records the exit facts in ``<job>.json`` and keeps the
finished job in the registry (bounded) so its exit code can still
be queried.

A core that dies without unwinding (crash, SIGKILL) closes the
master, and the kernel hangs up the job's session the way a closing
terminal does -- so an ungraceful death leaks nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

# How long a job's group gets between SIGTERM and SIGKILL.
_STOP_GRACE = 1.0

# How long the watcher waits for the pump to drain the last bytes
# after the command exits (a grandchild may hold the pty open).
_DRAIN_GRACE = 0.3

# Finished jobs kept in the registry for `jobs status`; their logs
# stay on disk either way.
_KEEP_FINISHED = 20

# One token per core process: log paths never collide with a previous
# run's, and an old log is never truncated by a fresh j1.
_BOOT_ID = secrets.token_hex(3)

_TRUNCATED = b"\n[nan: log truncated at the configured limit]\n"


def kill_group(pgid: int, sig: int) -> None:
    """Signal a whole process group; an already-dead group is nothing to do."""
    # ProcessLookupError: the group is already gone.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, sig)


def job_directory(workspace: Path, agent_id: str) -> Path:
    """This boot's job directory for one agent."""
    return workspace / ".nan" / "jobs" / agent_id / _BOOT_ID


@dataclass
class Job:
    """One command, either still running or finished but still known."""

    job_id: str
    proc: asyncio.subprocess.Process
    pgid: int
    log_path: Path
    master: int
    pump: asyncio.Task[None]
    command: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    watch: asyncio.Task[None] | None = field(default=None, repr=False)
    stdin: asyncio.StreamWriter | None = field(default=None, repr=False)
    stdin_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def running(self) -> bool:
        return self.proc.returncode is None

    @property
    def state(self) -> str:
        if self.running:
            return "running"

        return f"exited {self.proc.returncode}"

    @property
    def runtime(self) -> float:
        return (self.ended_at or time.time()) - self.started_at

    @property
    def status_path(self) -> Path:
        return self.log_path.with_suffix(".json")

    def write_status(self) -> None:
        """Persist the exit facts next to the log, atomically."""
        payload = {
            "job_id": self.job_id,
            "command": self.command,
            "pid": self.proc.pid,
            "pgid": self.pgid,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "exit_code": self.proc.returncode,
            "log": str(self.log_path),
        }

        try:
            tmp = self.log_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            tmp.replace(self.status_path)
        except OSError:
            # Losing the sidecar must not take the watcher down.
            pass

    def close(self) -> None:
        """Drop the pty and its pump; the process is gone or being killed."""
        close_stdin(self)

        if not self.pump.done():
            self.pump.cancel()
        else:
            # Consume a pump exception so it is not reported as
            # "never retrieved"; it changes nothing here.
            with contextlib.suppress(BaseException):
                self.pump.exception()

        with contextlib.suppress(OSError):
            os.close(self.master)


async def _watch(job: Job) -> None:
    """Record the exit facts once the command ends."""
    await job.proc.wait()

    job.ended_at = time.time()

    close_stdin(job)

    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(job.pump), timeout=_DRAIN_GRACE)

    job.write_status()


async def write_stdin(job: Job, text: str) -> bool:
    """
    Write to the command's stdin, leaving it open. False when the
    command already closed its side (or exited) -- nothing to do.
    """
    writer = job.stdin

    if writer is None or writer.is_closing():
        return False

    async with job.stdin_lock:
        try:
            writer.write(text.encode("utf-8"))

            await writer.drain()
        except (BrokenPipeError, ConnectionResetError, OSError, RuntimeError):
            return False

    return True


def close_stdin(job: Job) -> None:
    """Send EOF: the command's next read returns end-of-file."""
    writer = job.stdin

    if writer is None:
        return

    with contextlib.suppress(Exception):
        writer.close()


async def _pump(master: int, log_path: Path, limit: int) -> None:
    """
    Copy the pty into the log until the command closes its side. Past
    `limit` the log stops growing but the pty keeps draining, so a
    chatty command never blocks on a full terminal buffer.
    """
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

        written = 0

        with open(log_path, "wb", buffering=0) as log:
            while True:
                try:
                    chunk = await reader.read(65536)
                except OSError:
                    return

                if not chunk:
                    return

                if written >= limit:
                    continue

                room = limit - written
                log.write(chunk[:room])
                written += min(len(chunk), room)

                if written >= limit:
                    log.write(_TRUNCATED)

    finally:
        if transport is None:
            stream.close()
        else:
            transport.close()


def start_pump(master: int, log_path: Path, limit: int) -> asyncio.Task[None]:
    """Start copying the job's pty into its log."""
    return asyncio.create_task(_pump(master, log_path, limit), name=f"job-pump:{log_path.stem}")


def read_output(log_path: Path, limit: int) -> str:
    """
    The command's output, capped at `limit` characters: the head and
    the tail, so both the start and the end of the run stay visible.
    The file is read in place rather than loaded whole.
    """
    try:
        size = log_path.stat().st_size
    except OSError:
        return ""

    if size == 0:
        return ""

    if size <= limit:
        return log_path.read_text(errors="replace")

    half = max(limit // 2, 1)

    with log_path.open("rb") as file:
        head = file.read(half)
        file.seek(size - half)
        tail = file.read(half)

    omitted = size - len(head) - len(tail)

    return (
        head.decode(errors="replace")
        + f"\n...[truncated {omitted} bytes]...\n"
        + tail.decode(errors="replace")
    )


def read_slice(log_path: Path, offset: int, max_bytes: int) -> tuple[str, int, int]:
    """
    A byte range of the log, decoded leniently. Returns the text, the
    offset just past it, and the file's size.
    """
    try:
        size = log_path.stat().st_size
    except OSError:
        return "", 0, 0

    start = min(max(offset, 0), size)
    end = min(start + max_bytes, size)

    with log_path.open("rb") as file:
        file.seek(start)
        chunk = file.read(end - start)

    return chunk.decode(errors="replace"), end, size


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
        if job.ended_at is None:
            job.ended_at = time.time()

    for job in jobs:
        if not job.pump.done():
            job.pump.cancel()

    if jobs:
        await asyncio.gather(*(job.pump for job in jobs), return_exceptions=True)

    for job in jobs:
        job.close()


class JobRegistry:
    """The agent's jobs, in start order. Not shared between agents."""

    def __init__(self, keep_finished: int = _KEEP_FINISHED) -> None:
        self._jobs: dict[str, Job] = {}
        self._next = 0
        self._keep = keep_finished

    def new_id(self) -> str:
        self._next += 1

        return f"j{self._next}"

    def add(self, job: Job) -> None:
        self._jobs[job.job_id] = job

        job.watch = asyncio.create_task(_watch(job), name=f"job-watch:{job.job_id}")

        self._evict_finished()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    @property
    def all(self) -> list[Job]:
        return list(self._jobs.values())

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

    async def stop(self, job_id: str) -> Job | None:
        """
        Kill one job's process group but keep its record, so the exit
        status can still be read. A start that was cancelled before it
        could be reported is stopped the same way.
        """
        job = self._jobs.get(job_id)

        if job is None:
            return None

        await _terminate([job])

        return job

    def discard(self, job_id: str) -> None:
        """Drop a job that already exited; its pty goes with it."""
        job = self._jobs.pop(job_id, None)

        if job is None:
            return

        if job.watch is not None:
            job.watch.cancel()

        job.close()

        with contextlib.suppress(OSError):
            job.status_path.unlink()

    def _evict_finished(self) -> None:
        finished = sorted(
            (job for job in self._jobs.values() if not job.running and job.pump.done()),
            key=lambda job: job.started_at,
        )

        for job in finished[: max(len(finished) - self._keep, 0)]:
            self._jobs.pop(job.job_id, None)
            job.close()

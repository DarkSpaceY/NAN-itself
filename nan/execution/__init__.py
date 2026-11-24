"""Execution adapter: the OS-level sandbox for shell commands."""

from __future__ import annotations

from .jobs import (
    Job,
    JobRegistry,
    close_stdin,
    job_directory,
    kill_group,
    read_output,
    read_slice,
    start_pump,
    write_stdin,
)
from .sandbox import Sandbox, SandboxUnavailableError

__all__ = [
    "Job",
    "JobRegistry",
    "Sandbox",
    "SandboxUnavailableError",
    "close_stdin",
    "job_directory",
    "kill_group",
    "read_output",
    "read_slice",
    "start_pump",
    "write_stdin",
]

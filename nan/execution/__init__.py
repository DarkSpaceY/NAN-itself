"""Execution adapter: the OS-level sandbox for shell commands."""

from __future__ import annotations

from .home import home_environment
from .jobs import (
    Job,
    JobRegistry,
    close_stdin,
    job_directory,
    kill_group,
    read_output,
    read_slice,
    start_pump,
    trim_job_logs,
    write_stdin,
)
from .sandbox import Sandbox, SandboxUnavailableError

__all__ = [
    "Job",
    "JobRegistry",
    "Sandbox",
    "SandboxUnavailableError",
    "close_stdin",
    "home_environment",
    "job_directory",
    "kill_group",
    "read_output",
    "read_slice",
    "start_pump",
    "trim_job_logs",
    "write_stdin",
]

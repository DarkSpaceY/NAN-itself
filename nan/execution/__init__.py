"""Execution adapter: the OS-level sandbox for shell commands."""

from __future__ import annotations

from .jobs import Job, JobRegistry, kill_group
from .sandbox import Sandbox, SandboxUnavailableError

__all__ = ["Job", "JobRegistry", "Sandbox", "SandboxUnavailableError", "kill_group"]

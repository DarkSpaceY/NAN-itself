"""Execution adapter: the OS-level sandbox for shell commands."""

from __future__ import annotations

from .sandbox import Sandbox, SandboxUnavailableError

__all__ = ["Sandbox", "SandboxUnavailableError"]

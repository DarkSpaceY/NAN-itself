"""Execution-level sandbox: Seatbelt profile generation, command
wrapping, missing-tool behavior (required vs degrade)."""

from __future__ import annotations

import shlex
import shutil
import stat
from pathlib import Path
from typing import Any

import pytest

from nan.agent.sandbox import Sandbox, SandboxUnavailableError

FAKE_SANDBOX_EXEC = """#!/bin/bash
# fake sandbox-exec: verify a profile arg, then run the command
if [ "$1" = "-f" ]; then
    test -f "$2" || { echo "profile missing"; exit 3; }
    shift 3
fi
exec "$@"
"""


@pytest.fixture
def with_sandbox_exec(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A fake sandbox-exec on PATH: validates the -f profile then
    executes the wrapped command."""
    fake = tmp_path / "sandbox-exec"

    fake.write_text(FAKE_SANDBOX_EXEC)

    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)

    monkeypatch.setattr(shutil, "which", lambda name: str(fake) if name == "sandbox-exec" else None)

    return fake


def _sandbox(tmp_path: Path, **kwargs: Any) -> Sandbox:
    defaults: dict[str, Any] = {
        "workspace": tmp_path / "workspace",
        "package_dir": tmp_path / "pkg" / "nan",
        "builtin_dir": tmp_path / "builtin",
        "platform": "darwin",
    }

    defaults.update(kwargs)

    return Sandbox(**defaults)


def test_seatbelt_profile_generation(tmp_path: Path, with_sandbox_exec: Path) -> None:
    workspace = tmp_path / "workspace"

    workspace.mkdir()

    sb = _sandbox(tmp_path, workspace=workspace)

    profile_path = sb.ensure_profile()

    text = profile_path.read_text(encoding="utf-8")

    assert "(version 1)" in text
    assert "(deny default)" in text
    assert "(allow file-read*)" in text
    assert "(allow network*)" in text
    assert f'(subpath "{workspace.resolve()}")' in text
    assert '(subpath "/private/tmp")' in text
    assert '(subpath "/var/folders")' in text

    # the read-only guarantee: package + builtin denied for writes
    deny_section = text[text.index("(deny file-write*") :]

    assert f'(subpath "{sb.package_dir.resolve()}")' in deny_section
    assert f'(subpath "{sb.builtin_dir.resolve()}")' in deny_section

    # workspace is writable: its only mention sits before the deny
    assert text.index(f'(subpath "{workspace.resolve()}")') < text.index("(deny file-write*")


def test_wrap_darwin_quotes_command(tmp_path: Path, with_sandbox_exec: Path) -> None:
    sb = _sandbox(tmp_path)

    wrapped = sb.wrap("echo hi > out.txt")

    parts = shlex.split(wrapped)

    assert parts[:3] == ["sandbox-exec", "-f", str(sb.ensure_profile())]
    assert parts[3:] == ["bash", "-c", "echo hi > out.txt"]


def test_wrap_linux_bwrap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/bwrap" if name == "bwrap" else None)

    sb = _sandbox(tmp_path, platform="linux")

    wrapped = sb.wrap("echo hi")

    assert wrapped.startswith("bwrap --ro-bind / /")
    assert "--tmpfs /tmp" in wrapped
    assert f"--bind {sb.workspace.resolve()} {sb.workspace.resolve()}" in wrapped
    assert shlex.split(wrapped)[-3:] == ["bash", "-c", "echo hi"]


def test_missing_tool_required_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    sb = _sandbox(tmp_path, required=True)

    with pytest.raises(SandboxUnavailableError):
        sb.wrap("echo hi")


def test_missing_tool_optional_degrades_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    sb = _sandbox(tmp_path, required=False)

    assert sb.wrap("echo hi") == "echo hi"  # degraded, runs unwrapped
    assert sb.wrap("echo again") == "echo again"


def test_disabled_passes_through(tmp_path: Path, with_sandbox_exec: Path) -> None:
    sb = _sandbox(tmp_path, enabled=False)

    assert sb.wrap("echo hi") == "echo hi"


@pytest.mark.asyncio
async def test_shell_verb_runs_wrapped(tmp_path: Path, with_sandbox_exec: Path) -> None:
    """End-to-end through the shell verb with the mocked
    sandbox-exec: the profile is generated and the command runs."""
    from nan_fakes import FakeLLM, make_agent

    workspace = tmp_path / "workspace"

    workspace.mkdir()

    sb = _sandbox(tmp_path, workspace=workspace)

    agent = make_agent(tmp_path=tmp_path, llm=FakeLLM(), sandbox=sb)

    class Call:
        name = "shell"

        arguments = {"command": "echo sandbox-ok"}

    result = await agent._run_call(Call())

    assert "sandbox-ok" in result

    assert result.startswith("exit code: 0")

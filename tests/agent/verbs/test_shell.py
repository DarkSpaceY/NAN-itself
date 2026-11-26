"""The shell verb beyond the sandbox profile: cwd, env, stdin,
background jobs and their lifecycle. The sandbox itself is
covered in tests/execution."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nan_fakes import make_agent

from nan.agent.verbs.shell import ShellVerb
from nan.execution import job_directory

pytestmark = pytest.mark.asyncio


async def _shell(agent: Any, command: str, **arguments: object) -> Any:
    call = SimpleNamespace(name="shell", arguments={"command": command, **arguments}, error=None)

    return await ShellVerb().execute(call=call, agent=agent)


async def test_cwd_defaults_to_the_workspace(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, "pwd")

    assert result.text.startswith("exit code: 0\n")
    assert str(tmp_path / "workspace") in result.text


async def test_cwd_may_be_relative_inside_the_workspace(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    sub = tmp_path / "workspace" / "sub"

    sub.mkdir()

    result = await _shell(root, "pwd", cwd="sub")

    assert result.text.startswith("exit code: 0\n")
    assert str(sub) in result.text


async def test_cwd_outside_the_workspace_is_refused(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, "touch escaped", cwd=str(tmp_path))

    assert result.text.startswith("'cwd' must stay inside the workspace")
    assert not (tmp_path / "escaped").exists()


async def test_cwd_must_be_a_directory(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, "pwd", cwd="missing")

    assert result.text.startswith("'cwd' is not a directory")


@pytest.mark.parametrize("value", ["", 1, ["sub"], {"dir": "sub"}])
async def test_cwd_must_be_a_string(tmp_path: Path, no_sink: Any, value: object) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, "pwd", cwd=value)

    assert result.text == "'cwd' must be a non-empty string."


@pytest.mark.parametrize("value", [{"A": 1}, "A=1", ["A=1"]])
async def test_env_must_be_an_object_of_strings(
    tmp_path: Path, no_sink: Any, value: object
) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, "pwd", env=value)

    assert result.text == "'env' must be an object of string values."


async def test_env_is_merged_into_the_inherited_environment(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, 'echo "value=$NAN_TEST_VALUE"', env={"NAN_TEST_VALUE": "hello"})

    assert "value=hello" in result.text


async def test_commands_carry_the_agent_identity(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, 'echo "agent=$NAN_AGENT_ID"')

    assert f"agent={root.agent_id}" in result.text


async def test_workspace_bin_leads_the_path(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    tools = tmp_path / "workspace" / "bin"

    tools.mkdir()

    script = tools / "hello-tool"

    script.write_text("#!/bin/sh\necho hello-from-tool\n")
    script.chmod(0o755)

    result = await _shell(root, "hello-tool")

    assert "hello-from-tool" in result.text


async def test_the_model_cannot_shadow_workspace_bin(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    tools = tmp_path / "workspace" / "bin"

    tools.mkdir()

    script = tools / "echo"

    script.write_text("#!/bin/sh\necho shadowed\n")
    script.chmod(0o755)

    result = await _shell(root, "echo real")

    assert "real" in result.text
    assert "shadowed" not in result.text


async def test_the_model_cannot_override_the_agent_identity(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, 'echo "agent=$NAN_AGENT_ID"', env={"NAN_AGENT_ID": "spoofed"})

    assert f"agent={root.agent_id}" in result.text


async def test_home_points_inside_the_workspace(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, 'echo "$HOME"; echo "$XDG_CACHE_HOME"')

    home = (tmp_path / "workspace" / "home").resolve()

    assert str(home) in result.text
    assert str(home / ".cache") in result.text
    assert home.is_dir()


async def test_the_model_cannot_override_home(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, 'echo "$HOME"', env={"HOME": str(tmp_path)})

    assert str(tmp_path / "workspace" / "home") in result.text
    assert result.text.strip().endswith(str((tmp_path / "workspace" / "home").resolve()))


async def test_stdin_is_written_when_the_command_starts(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    # head reads one line and exits; the pipe stays open after it, so
    # a command that consumed all of stdin would block instead.
    result = await _shell(root, "head -1", stdin="hello stdin\n")

    assert result.text.startswith("exit code: 0\n")
    assert "hello stdin" in result.text


@pytest.mark.parametrize("value", [1, ["hi"]])
async def test_stdin_must_be_a_string(tmp_path: Path, no_sink: Any, value: object) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, "cat", stdin=value)

    assert result.text == "'stdin' must be a string."


async def test_stdin_with_background_is_allowed(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, "head -1", stdin="hi\n", background=True)

    assert result.detail == {"exit_code": 0}
    assert "hi" in result.text


@pytest.mark.parametrize("value", ["true", 1])
async def test_background_must_be_a_boolean(tmp_path: Path, no_sink: Any, value: object) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, "true", background=value)

    assert result.text == "'background' must be a boolean."


async def test_a_command_that_outlives_the_window_becomes_a_job(
    tmp_path: Path, no_sink: Any
) -> None:
    root = make_agent(tmp_path=tmp_path, agent={"shell_timeout": 0.5})

    try:
        result = await _shell(root, "echo before; sleep 30")

        assert result.detail["job_id"] == "j1"
        assert result.detail["promoted"] is True
        assert "Still running after 0.5s" in result.text

        log = job_directory(tmp_path / "workspace", root.agent_id) / "j1.log"

        for _ in range(20):
            if "before" in log.read_text():
                break

            await asyncio.sleep(0.05)

        assert "before" in log.read_text()

    finally:
        await root.jobs.stop_all()


async def test_a_finished_command_leaves_no_job(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, "echo hi")

    assert "job_id" not in result.detail
    assert root.jobs.all == []


async def test_background_returns_a_job_and_its_log(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    try:
        result = await _shell(root, "sleep 30", background=True)

        assert result.detail["job_id"] == "j1"
        assert result.detail["pgid"] == result.detail["pid"]

        log = job_directory(tmp_path / "workspace", root.agent_id) / "j1.log"

        assert log.exists()
        assert str(log.relative_to(tmp_path / "workspace")) in result.text

    finally:
        await root.jobs.stop_all()


async def test_stopping_jobs_kills_their_process_groups(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, "sleep 30", background=True)

    await root.jobs.stop_all()

    assert root.jobs.live == []

    with pytest.raises(ProcessLookupError):
        os.kill(result.detail["pid"], 0)


async def test_a_background_command_that_exits_at_once_reports_the_exit(
    tmp_path: Path, no_sink: Any
) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, "echo boom; exit 3", background=True)

    assert result.detail == {"exit_code": 3}
    assert "exited immediately" in result.text
    assert "boom" in result.text


async def test_death_kills_background_jobs(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    await _shell(root, "sleep 30", background=True)

    root.request_death("test cleanup")

    await root._dispose()

    assert root.jobs.live == []


async def test_shutdown_kills_background_jobs(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    await _shell(root, "sleep 30", background=True)

    root.suspend()

    await root._suspend()

    assert root.jobs.live == []


async def test_pause_leaves_background_jobs_alone(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    try:
        result = await _shell(root, "sleep 30", background=True)

        root.pause()

        # A pause cancels the in-flight turn, not the agent's processes.
        assert root.jobs.live != []

        os.kill(result.detail["pid"], 0)

    finally:
        await root.jobs.stop_all()


async def test_a_crashed_core_hangs_up_the_job(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    try:
        await _shell(root, "sleep 30", background=True)

        job = next(iter(root.jobs._jobs.values()))

        # A core that dies without unwinding closes every fd: the pump's
        # duplicate and the master. The kernel hangs up the session.
        job.pump.cancel()

        with pytest.raises(asyncio.CancelledError):
            await job.pump

        job.close()

        await asyncio.wait_for(job.proc.wait(), timeout=2.0)

        assert job.proc.returncode == -signal.SIGHUP

    finally:
        await root.jobs.stop_all()


async def test_cancelling_the_start_kills_the_job(
    tmp_path: Path, no_sink: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nan.agent.verbs.shell._BACKGROUND_GRACE", 5.0)

    root = make_agent(tmp_path=tmp_path)
    pids: list[int] = []

    create = asyncio.create_subprocess_shell

    async def record(*args: Any, **kwargs: Any) -> Any:
        proc = await create(*args, **kwargs)
        pids.append(proc.pid)

        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", record)

    task = asyncio.create_task(_shell(root, "sleep 30", background=True))

    for _ in range(200):
        if pids:
            break

        await asyncio.sleep(0.01)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert root.jobs.live == []

    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)


async def test_the_log_is_raw_output(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    try:
        result = await _shell(root, "printf 'a\\nb\\n'; sleep 2", background=True)

        log = (
            job_directory(tmp_path / "workspace", root.agent_id) / f"{result.detail['job_id']}.log"
        )

        for _ in range(200):
            if log.exists() and log.read_bytes() == b"a\nb\n":
                break

            await asyncio.sleep(0.01)

        assert log.read_bytes() == b"a\nb\n"

    finally:
        await root.jobs.stop_all()


async def test_the_log_stops_at_the_configured_limit(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path, agent={"job_log_limit": 1024})

    result = await _shell(root, "head -c 5000 /dev/zero | tr '\\0' 'x'")

    assert result.text.startswith("exit code: 0\n")

    log = job_directory(tmp_path / "workspace", root.agent_id) / "j1.log"
    data = log.read_bytes()

    assert data.startswith(b"x" * 1024)
    assert b"log truncated" in data
    assert len(data) < 2000


async def test_a_previous_runs_log_is_not_overwritten(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    stale = tmp_path / "workspace" / ".nan" / "jobs" / root.agent_id / "j1.log"
    stale.parent.mkdir(parents=True)
    stale.write_text("old run")

    result = await _shell(root, "echo fresh")

    assert result.text.startswith("exit code: 0\n")
    assert stale.read_text() == "old run"

    log = job_directory(tmp_path / "workspace", root.agent_id) / "j1.log"

    assert "fresh" in log.read_text()

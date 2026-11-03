"""The shell verb beyond the sandbox profile: cwd, env, stdin,
background jobs and their lifecycle. The sandbox itself is
covered in tests/execution."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nan_fakes import make_agent

from nan.agent.verbs.shell import ShellVerb

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


async def test_stdin_is_fed_once(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, "cat", stdin="hello stdin")

    assert result.text.startswith("exit code: 0\n")
    assert "hello stdin" in result.text


@pytest.mark.parametrize("value", [1, ["hi"]])
async def test_stdin_must_be_a_string(tmp_path: Path, no_sink: Any, value: object) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, "cat", stdin=value)

    assert result.text == "'stdin' must be a string."


async def test_stdin_with_background_is_refused(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, "cat", stdin="hi", background=True)

    assert result.text.startswith("'stdin' cannot be used with background")


@pytest.mark.parametrize("value", ["true", 1])
async def test_background_must_be_a_boolean(tmp_path: Path, no_sink: Any, value: object) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, "true", background=value)

    assert result.text == "'background' must be a boolean."


async def test_a_timeout_keeps_the_output_so_far(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path, agent={"shell_timeout": 0.5})

    result = await _shell(root, "echo before; sleep 30")

    assert result.detail == {"timed_out": True}
    assert "before" in result.text
    assert "Timed out after 0.5s" in result.text


async def test_background_returns_a_job_and_its_log(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    try:
        result = await _shell(root, "sleep 30", background=True)

        assert result.detail["job_id"] == "j1"
        assert result.detail["pgid"] == result.detail["pid"]

        log = tmp_path / "workspace" / ".nan" / "jobs" / root.agent_id / "j1.log"

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

    assert result.detail == {"exit_code": 3, "timed_out": False}
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

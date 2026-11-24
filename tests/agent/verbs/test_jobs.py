"""The jobs verb: the console for commands that kept running."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nan_fakes import make_agent

from nan.agent.verbs.jobs import JobsVerb
from nan.agent.verbs.shell import ShellVerb

pytestmark = pytest.mark.asyncio


async def _shell(agent: Any, command: str, **arguments: object) -> Any:
    call = SimpleNamespace(name="shell", arguments={"command": command, **arguments}, error=None)

    return await ShellVerb().execute(call=call, agent=agent)


async def _jobs(agent: Any, **arguments: object) -> Any:
    call = SimpleNamespace(name="jobs", arguments=arguments, error=None)

    return await JobsVerb().execute(call=call, agent=agent)


async def _background(agent: Any, command: str) -> str:
    result = await _shell(agent, command, background=True)

    return result.detail["job_id"]


async def _read_until(agent: Any, job_id: str, needle: str) -> Any:
    """Poll the log until the pump has caught up with the command."""
    result = await _jobs(agent, action="logs", job_id=job_id)

    for _ in range(40):
        if needle in result.text:
            return result

        await asyncio.sleep(0.05)

        result = await _jobs(agent, action="logs", job_id=job_id)

    return result


async def test_list_shows_started_jobs(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    try:
        job_id = await _background(root, "sleep 30")

        result = await _jobs(root, action="list")

        assert job_id in result.text
        assert "running" in result.text
        assert "sleep 30" in result.text

    finally:
        await root.jobs.stop_all()


async def test_list_says_when_there_is_nothing(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _jobs(root, action="list")

    assert result.text.startswith("No jobs")


async def test_logs_reads_in_offsets(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    try:
        job_id = await _background(root, "printf 'hello'; sleep 30")

        first = await _read_until(root, job_id, "hello")

        assert "hello" in first.text
        assert first.detail["offset"] == 5

        again = await _jobs(root, action="logs", job_id=job_id, offset=first.detail["offset"])

        assert "no new output" in again.text

    finally:
        await root.jobs.stop_all()


async def test_logs_validates_the_offset(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    try:
        job_id = await _background(root, "sleep 30")

        result = await _jobs(root, action="logs", job_id=job_id, offset=-1)

        assert result.text == "'offset' must be an integer >= 0."

    finally:
        await root.jobs.stop_all()


async def test_wait_reports_a_job_that_is_still_running(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    try:
        job_id = await _background(root, "sleep 30")

        result = await _jobs(root, action="wait", job_id=job_id, timeout=0.1)

        assert "still running" in result.text
        assert result.detail["exited"] is False

    finally:
        await root.jobs.stop_all()


async def test_wait_returns_the_exit_code(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    job_id = await _background(root, "sleep 0.5; exit 7")

    result = await _jobs(root, action="wait", job_id=job_id, timeout=10)

    assert result.detail == {"action": "wait", "job_id": job_id, "exited": True, "exit_code": 7}
    assert "exited 7" in result.text


async def test_stop_kills_the_group_and_keeps_the_record(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _shell(root, "sleep 30", background=True)

    job_id = result.detail["job_id"]
    pid = result.detail["pid"]

    stopped = await _jobs(root, action="stop", job_id=job_id)

    assert "Stopped" in stopped.text

    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)

    status = await _jobs(root, action="status", job_id=job_id)

    assert "exited" in status.text


async def test_status_names_pid_and_log(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    try:
        job_id = await _background(root, "sleep 30")

        result = await _jobs(root, action="status", job_id=job_id)

        assert "pid" in result.text
        assert f"{job_id}.log" in result.text

    finally:
        await root.jobs.stop_all()


async def test_unknown_job_is_reported(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _jobs(root, action="status", job_id="j99")

    assert "Unknown job id: j99" in result.text


async def test_action_must_be_known(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _jobs(root, action="dance")

    assert result.text == "'action' must be one of: list, status, logs, wait, stop."


async def test_job_id_is_required_outside_list(tmp_path: Path, no_sink: Any) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await _jobs(root, action="logs")

    assert result.text == "'job_id' is required for 'logs'."

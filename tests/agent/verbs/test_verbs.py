"""The six verbs: argument validation, routing and their domain
facts. The shell verb's sandbox is covered in tests/execution."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from nan_fakes import await_until, make_agent

from nan.agent.verbs.finish import FinishVerb
from nan.agent.verbs.kill import KillVerb
from nan.agent.verbs.send import SendVerb
from nan.agent.verbs.sleep import SleepVerb
from nan.agent.verbs.spawn import SpawnVerb

pytestmark = pytest.mark.asyncio


def _call(name: str, **arguments: object) -> SimpleNamespace:
    return SimpleNamespace(name=name, arguments=arguments, error=None)


async def test_a_malformed_tool_call_never_runs_and_returns_its_error(
    tmp_path: Path, no_sink: None
) -> None:
    root = make_agent(tmp_path=tmp_path)

    marker = tmp_path / "workspace" / "ran"

    result = await root._execute_call(
        SimpleNamespace(
            name="shell",
            arguments={"command": f"touch {marker}"},
            error="Arguments for 'shell' are not valid JSON",
        )
    )

    # The adapter's complaint is this call's result: the model reads
    # it and corrects itself on the next pass. Nothing executes.
    assert result == "Arguments for 'shell' are not valid JSON"
    assert not marker.exists()


async def _cleanup(*agents: object) -> None:
    for agent in agents:
        agent.request_death("test cleanup")

        if agent.task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await agent.task


async def test_spawn_requires_a_task(tmp_path: Path, no_sink: None) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await SpawnVerb().execute(call=_call("spawn", task="   "), agent=root)

    assert result.text == "spawn requires a non-empty 'task'."
    assert result.detail == {}


async def test_spawn_returns_the_child_identity(tmp_path: Path, no_sink: None) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await SpawnVerb().execute(call=_call("spawn", task="do work"), agent=root)

    child = root.children[result.detail["child_id"]]

    assert result.detail == {"child_id": child.agent_id, "depth": child.depth}
    assert result.text == f"Subagent spawned.\nid: {child.agent_id}\ndepth: {child.depth}"

    await _cleanup(child, root)


@pytest.mark.parametrize("seconds", ["5", None, True, [1]])
async def test_sleep_rejects_non_numbers(tmp_path: Path, no_sink: None, seconds: object) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await SleepVerb().execute(call=_call("sleep", seconds=seconds), agent=root)

    assert result.text == "'seconds' must be a number."


async def test_sleep_rejects_negative(tmp_path: Path, no_sink: None) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await SleepVerb().execute(call=_call("sleep", seconds=-1), agent=root)

    assert result.text == "'seconds' must be >= 0."


async def test_sleep_waits_and_reports_the_interval(tmp_path: Path, no_sink: None) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await SleepVerb().execute(call=_call("sleep", seconds=0), agent=root)

    assert result.text == "Waited 0 seconds."
    assert result.detail == {"seconds": 0.0}


async def test_send_rejects_a_bad_direction_or_empty_text(tmp_path: Path, no_sink: None) -> None:
    root = make_agent(tmp_path=tmp_path)

    bad_direction = await SendVerb().execute(
        call=_call("send", direction="sideways", text="hi"), agent=root
    )

    assert bad_direction.text == "'direction' must be 'up' or 'down'."

    empty = await SendVerb().execute(call=_call("send", direction="up", text="  "), agent=root)

    assert empty.text == "send requires a non-empty 'text'."


async def test_the_root_cannot_send_up(tmp_path: Path, no_sink: None) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await SendVerb().execute(call=_call("send", direction="up", text="hi"), agent=root)

    assert result.text == "No parent: the root cannot send up."


async def test_a_child_sends_up_into_the_parent_inbox(tmp_path: Path, no_sink: None) -> None:
    root = make_agent(tmp_path=tmp_path)

    child, err = await root.spawn_child("child")

    assert err is None and child is not None

    result = await SendVerb().execute(call=_call("send", direction="up", text="hi"), agent=child)

    assert result.text == "Sent up."
    assert result.detail == {"direction": "up"}

    message = root._inbox.pop(0)

    assert (message.kind, message.source, message.text) == ("subagent", child.agent_id, "hi")

    await _cleanup(child, root)


async def test_send_down_requires_a_known_child(tmp_path: Path, no_sink: None) -> None:
    root = make_agent(tmp_path=tmp_path)

    child, err = await root.spawn_child("child")

    assert err is None and child is not None

    result = await SendVerb().execute(
        call=_call("send", direction="down", child_id="missing", text="hi"), agent=root
    )

    assert result.text == f"Unknown child 'missing'. Children: {child.agent_id}"

    await _cleanup(child, root)


async def test_send_down_into_the_child_inbox(tmp_path: Path, no_sink: None) -> None:
    root = make_agent(tmp_path=tmp_path)

    child, err = await root.spawn_child("child")

    assert err is None and child is not None

    result = await SendVerb().execute(
        call=_call("send", direction="down", child_id=child.agent_id, text="hi"), agent=root
    )

    assert result.text == f"Sent down to {child.agent_id}."
    assert result.detail == {"direction": "down", "to": child.agent_id}

    messages = [message for message in child._inbox if message.text == "hi"]

    assert messages and messages[0].kind == "parentagent"

    await _cleanup(child, root)


async def test_kill_requires_a_known_child(tmp_path: Path, no_sink: None) -> None:
    root = make_agent(tmp_path=tmp_path)

    child, err = await root.spawn_child("child")

    assert err is None and child is not None

    result = await KillVerb().execute(call=_call("kill", child_id="missing"), agent=root)

    assert result.text == f"Unknown child 'missing'. Children: {child.agent_id}"

    await _cleanup(child, root)


async def test_kill_requests_the_child_death(tmp_path: Path, no_sink: None) -> None:
    root = make_agent(tmp_path=tmp_path)

    child, err = await root.spawn_child("child")

    assert err is None and child is not None

    result = await KillVerb().execute(call=_call("kill", child_id=child.agent_id), agent=root)

    assert result.text == f"Kill requested for {child.agent_id}."
    assert result.detail == {"target_id": child.agent_id}

    await await_until(lambda: child._death is not None)

    assert child._death is not None
    assert child._death.reason == f"killed by holder {root.agent_id}"

    await _cleanup(root)


async def test_finish_requires_a_report(tmp_path: Path, no_sink: None) -> None:
    root = make_agent(tmp_path=tmp_path)

    child, err = await root.spawn_child("child")

    assert err is None and child is not None

    result = await FinishVerb().execute(call=_call("finish", report="  "), agent=child)

    assert result.text == "finish requires a non-empty 'report'."
    assert child._death is None

    await _cleanup(child, root)


async def test_finish_records_the_report(tmp_path: Path, no_sink: None) -> None:
    root = make_agent(tmp_path=tmp_path)

    child, err = await root.spawn_child("child")

    assert err is None and child is not None

    result = await FinishVerb().execute(call=_call("finish", report="done"), agent=child)

    assert result.text == "Report submitted upstream; task finished."
    assert child._death is not None
    assert child._death.reason == "finished"
    assert child._death.report == "done"

    await _cleanup(child, root)


async def test_a_finished_child_reports_into_the_parent_inbox(
    tmp_path: Path, no_sink: None
) -> None:
    root = make_agent(tmp_path=tmp_path)

    child, err = await root.spawn_child("child")

    assert err is None and child is not None

    await FinishVerb().execute(call=_call("finish", report="done"), agent=child)

    await _cleanup(child)

    message = root._inbox.pop(0)

    assert message.kind == "subagent"
    assert message.event == "finish"
    assert message.text == "done"
    assert child.agent_id not in root.children

    await _cleanup(root)


async def test_the_root_does_not_own_the_finish_verb(tmp_path: Path, no_sink: None) -> None:
    root = make_agent(tmp_path=tmp_path)

    result = await root._execute_call(_call("finish", report="nope"))

    assert result.startswith("Unknown action 'finish'.")
    assert root._death is None

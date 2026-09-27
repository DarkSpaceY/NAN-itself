"""
Let-it-crash supervision semantics for ask / tell / feed.

One non-Cancelled failure out of ask(), tell() or feed()
crashes the module exactly like a start() crash: DOWN with
the recorded error, its live start task cancelled, and a
supervised backoff restart. The failure never propagates to
the caller: the agent's query completes normally and the
model only ever sees a result string.

Every Facade is pointed at tmp dirs only (never the real
builtin/modules -- real modules would start).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from nan_itself.modules.model import (
    ModuleState,
    Turn,
)
from nan_itself.modules.runtime import (
    Facade,
)


def run(coro):
    return asyncio.run(coro)


def _turn() -> Turn:
    return Turn(
        agent_hash="hash",
        parent_hash=None,
        depth=0,
        task=None,
        world={},
    )


def _facade(tmp_path: Path) -> Facade:
    return Facade(
        workspace_modules=tmp_path / "modules",
        builtin_modules_dir=tmp_path / "builtin",
        data_dir=tmp_path / "data",
        retry_interval=0.05,
    )


async def _load_and_start(
    facade: Facade,
    tmp_path: Path,
    filename: str,
    source: str,
):
    path = tmp_path / filename

    path.write_text(source, encoding="utf-8")

    await facade._load_or_reload_file(
        path,
        facade._fingerprint(path),
    )

    record = facade._find_record_by_source(path)

    await facade._try_start(record)

    return record


async def _settle(record) -> None:
    """
    Let the crash-cancelled start task finish dying.
    """
    for _ in range(10):
        task = record.task

        if task is None:
            break

        if task.done():
            break

        await asyncio.sleep(0.01)


# ============================================================================
# ask() failure
# ============================================================================


_ASK_CRASH_SOURCE = """# @module
import asyncio


class AskCrash(Module):
    id = "ask-crash"

    def __init__(self):
        self.starts = 0

    async def start(self):
        self.starts += 1
        await asyncio.Event().wait()

    async def ask(self, turn):
        raise RuntimeError("ask boom")
"""


_HEALTHY_SOURCE = """# @module
import asyncio


class Healthy(Module):
    id = "healthy"

    async def start(self):
        await asyncio.Event().wait()

    async def ask(self, turn):
        return "ambient-ok"

    async def tell(self, turn):
        pass
"""


async def _ask_crash_full_cycle(tmp_path: Path) -> None:
    facade = _facade(tmp_path)

    crashy = await _load_and_start(
        facade,
        tmp_path,
        "ask_crash.py",
        _ASK_CRASH_SOURCE,
    )

    healthy = await _load_and_start(
        facade,
        tmp_path,
        "healthy.py",
        _HEALTHY_SOURCE,
    )

    assert crashy.state is ModuleState.RUNNING

    assert healthy.state is ModuleState.RUNNING

    old_task = crashy.task

    # The agent's query completes normally: the crashed module
    # contributes nothing, the healthy one still answers.
    results = await facade.query_snapshot(_turn())

    assert results == ["ambient-ok"]

    # One failure = DOWN with the recorded error.
    assert crashy.state is ModuleState.DOWN

    assert isinstance(crashy.error, RuntimeError)

    assert "ask boom" in str(crashy.error)

    assert crashy.retry_at > 0.0

    # The live start task (any realtime service) was killed,
    # and its cleanup did not wipe the recorded error.
    await _settle(crashy)

    assert old_task.cancelled()

    assert crashy.task is None

    # A peer module is untouched.
    assert healthy.state is ModuleState.RUNNING

    assert not healthy.task.done()

    # The supervisor restarts the module after backoff.
    await asyncio.sleep(0.1)

    await facade._reconcile()

    assert crashy.state is ModuleState.RUNNING

    assert crashy.error is None

    assert crashy.task is not None

    assert crashy.task is not old_task

    assert crashy.instance.starts == 2

    await facade.stop()


def test_ask_failure_crashes_and_restarts(tmp_path):
    run(_ask_crash_full_cycle(tmp_path))


# ============================================================================
# tell() failure
# ============================================================================


_TELL_CRASH_SOURCE = """# @module
import asyncio


class TellCrash(Module):
    id = "tell-crash"

    async def start(self):
        await asyncio.Event().wait()

    async def ask(self, turn):
        return None

    async def tell(self, turn):
        raise RuntimeError("tell boom")
"""


async def _tell_crash_deliver_turn(tmp_path: Path) -> None:
    facade = _facade(tmp_path)

    crashy = await _load_and_start(
        facade,
        tmp_path,
        "tell_crash.py",
        _TELL_CRASH_SOURCE,
    )

    healthy = await _load_and_start(
        facade,
        tmp_path,
        "healthy.py",
        _HEALTHY_SOURCE,
    )

    old_task = crashy.task

    facade.deliver_turn(_turn())

    pending = set(Facade._DELIVERY_TASKS)

    await asyncio.gather(*pending)

    assert crashy.state is ModuleState.DOWN

    assert isinstance(crashy.error, RuntimeError)

    assert "tell boom" in str(crashy.error)

    await _settle(crashy)

    assert old_task.cancelled()

    # A peer module is untouched by the crash.
    assert healthy.state is ModuleState.RUNNING

    assert not healthy.task.done()

    await facade.stop()


def test_tell_failure_crashes_module(tmp_path):
    run(_tell_crash_deliver_turn(tmp_path))


# ============================================================================
# feed() failure
# ============================================================================


_FEED_CRASH_SOURCE = """# @module
import asyncio

from pydantic import BaseModel

from nan_itself.modules import ChannelSpec


class Goal(BaseModel):
    text: str


class FeedCrash(Module):
    id = "feed-crash"

    channels = {
        "goal": ChannelSpec(
            Goal,
            description="Navigation goal",
        ),
    }

    async def start(self):
        await asyncio.Event().wait()

    async def ask(self, turn):
        return None

    def feed(self, channel, payload):
        raise RuntimeError("feed boom")
"""


async def _feed_crash_channel_write(tmp_path: Path) -> None:
    facade = _facade(tmp_path)

    record = await _load_and_start(
        facade,
        tmp_path,
        "feed_crash.py",
        _FEED_CRASH_SOURCE,
    )

    old_task = record.task

    # A schema-invalid payload is a model input error, not a
    # crash: rejected at the boundary, module keeps running.
    rejected = facade.write_module_channel(
        "feed-crash", "goal", {"priority": 3}
    )

    assert rejected.startswith("rejected")

    assert record.state is ModuleState.RUNNING

    assert record.error is None

    # A feed() exception crashes the module; the model still
    # gets a plain failure string naming the module.
    result = facade.write_module_channel(
        "feed-crash", "goal", {"text": "go"}
    )

    assert "crashed" in result

    assert "feed-crash" in result

    assert "restarted" in result

    assert record.state is ModuleState.DOWN

    assert isinstance(record.error, RuntimeError)

    assert "feed boom" in str(record.error)

    await _settle(record)

    assert old_task.cancelled()

    # The supervisor restarts the module after backoff.
    await asyncio.sleep(0.1)

    await facade._reconcile()

    assert record.state is ModuleState.RUNNING

    assert record.error is None

    await facade.stop()


def test_feed_failure_crashes_module_and_reports(tmp_path):
    run(_feed_crash_channel_write(tmp_path))


# ============================================================================
# Regression: a normal shutdown cancel records no error
# ============================================================================


async def _shutdown_cancel_records_no_error(
    tmp_path: Path,
) -> None:
    facade = _facade(tmp_path)

    record = await _load_and_start(
        facade,
        tmp_path,
        "healthy.py",
        _HEALTHY_SOURCE,
    )

    assert record.state is ModuleState.RUNNING

    await facade.stop()

    # The stopping-path cancel must not fabricate an error.
    assert record.error is None

    assert record.state is ModuleState.STOPPING


def test_shutdown_cancel_records_no_error(tmp_path):
    run(_shutdown_cancel_records_no_error(tmp_path))

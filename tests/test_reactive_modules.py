"""
Channel downlink mechanism tests.

Covers:
    - ChannelSpec schema derivation and validation
    - Module.feed base contract ('written' by default)
    - Facade routing (never raises; failure modes as strings)
    - verb triple end-to-end against a live Facade

Every Facade is pointed at tmp dirs only (never the real
builtin/modules -- real modules would start).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel

from nan_itself.modules.model import (
    ChannelSpec,
    Module,
    ModuleState,
)
from nan_itself.modules.runtime import (
    Facade,
)
from nan_itself.agent.verbs import (
    InvokeChannelsVerb,
    ListChannelsVerb,
    ShowChannelsVerb,
)


def run(coro):
    return asyncio.run(coro)


class Goal(BaseModel):
    text: str
    priority: int = 1


# ============================================================================
# ChannelSpec
# ============================================================================


def test_channel_spec_schema_and_validation():
    spec = ChannelSpec(
        Goal,
        description="Navigation goal",
    )

    schema = spec.json_schema()

    assert schema is not None

    assert "title" not in schema

    assert schema["properties"]["text"] == {
        "type": "string"
    }

    payload, error = spec.validate(
        {"text": "go", "priority": 2}
    )

    assert error is None

    assert payload == {"text": "go", "priority": 2}

    payload, error = spec.validate({"priority": 2})

    assert payload is None

    assert error is not None


def test_channel_spec_unvalidated_passthrough():
    spec = ChannelSpec()

    assert spec.json_schema() is None

    payload = {"anything": [1, 2]}

    validated, error = spec.validate(payload)

    assert error is None

    assert validated is payload


def test_module_base_feed_accepts_by_default():
    # The runtime validates channel existence and schema before
    # calling feed(); the base default accepts whatever it gets.
    assert Module().feed("goal", {"text": "x"}) == "written"


# ============================================================================
# Facade routing (never raises)
# ============================================================================

_PROBE_SOURCE = """# @module
import asyncio

from pydantic import BaseModel

from nan_itself.modules import ChannelSpec


class Goal(BaseModel):
    text: str
    priority: int = 1


class Probe(Module):
    id = "probe"

    channels = {
        "goal": ChannelSpec(Goal, description="Navigation goal"),
        "burst": ChannelSpec(description="Command burst"),
    }

    def __init__(self):
        self.fed = []

    def feed(self, channel, payload):
        self.fed.append((channel, payload))
        return "written"

    async def start(self):
        await asyncio.Event().wait()

    async def ask(self, turn):
        return None
"""


def _write_probe(path: Path) -> None:
    path.write_text(_PROBE_SOURCE, encoding="utf-8")


def _facade(tmp_path: Path) -> Facade:
    return Facade(
        workspace_modules=tmp_path / "modules",
        builtin_modules_dir=tmp_path / "builtin",
        data_dir=tmp_path / "data",
    )


async def _running_probe(
    tmp_path: Path,
) -> tuple[Facade, object]:
    facade = _facade(tmp_path)

    source = tmp_path / "probe.py"

    _write_probe(source)

    await facade._load_or_reload_file(
        source,
        facade._fingerprint(source),
    )

    record = facade._find_record_by_source(source)

    await facade._try_start(record)

    return facade, record.instance


def test_facade_routing_failure_modes(tmp_path):
    facade = _facade(tmp_path)

    # Unknown module: never raises.
    assert "Unknown module" in (
        facade.write_module_channel(
            "ghost", "goal", {"text": "x"}
        )
    )

    assert "Unknown module" in (
        facade.show_module_channels("ghost")
    )

    # A channel-free Module exposes no channels.
    plain = Module()

    from nan_itself.modules.model import (
        DataSpace,
        ModuleRecord,
        ModuleState,
    )

    record = ModuleRecord(
        id="plain",
        cls=Module,
        instance=plain,
        data=DataSpace("plain"),
        source_path="memory",
    )

    record.state = ModuleState.RUNNING

    facade.modules["plain"] = record

    assert "exposes no channels" in (
        facade.write_module_channel(
            "plain", "goal", {"text": "x"}
        )
    )

    assert "exposes no channels" in (
        facade.show_module_channels("plain")
    )


# ============================================================================
# Facade lifecycle: provisioning failure -> DOWN + retry_at
# ============================================================================


_CRASHING_SOURCE = """# @module
class Crashing(Module):
    id = "crashing"

    async def start(self):
        raise RuntimeError("no weights: drop model.onnx")

    async def ask(self, turn):
        return None
"""


async def _provisioning_failure_down(tmp_path: Path) -> None:
    facade = _facade(tmp_path)

    source = tmp_path / "crashing.py"

    source.write_text(_CRASHING_SOURCE, encoding="utf-8")

    await facade._load_or_reload_file(
        source,
        facade._fingerprint(source),
    )

    record = facade._find_record_by_source(source)

    await facade._try_start(record)

    task = record.task

    if task is not None:
        await task

    # The exception out of start() is not swallowed: the Facade
    # records the module DOWN with the error and schedules a
    # backoff retry instead of leaving an "unavailable" limbo.
    assert record.state is ModuleState.DOWN

    assert "no weights" in str(record.error)

    assert record.retry_at > 0.0


def test_facade_provisioning_failure_records_down(tmp_path):
    run(_provisioning_failure_down(tmp_path))


async def _full_chain(tmp_path):
    facade, instance = await _running_probe(
        tmp_path
    )

    # list: bare 'module/channel' lines, no depth/occupancy.
    listing = facade.list_module_channels()

    assert "probe/goal" in listing

    assert "probe/burst" in listing

    assert "depth" not in listing

    # show: description + schema only.
    detail = facade.show_module_channels(
        "probe", "goal"
    )

    assert "Navigation goal" in detail

    assert '"text"' in detail

    assert "occupancy" not in detail

    assert "depth" not in detail

    # invoke: every validated feed is 'written' (the module's
    # consumption policy is private now).
    assert (
        facade.write_module_channel(
            "probe", "goal", {"text": "a"}
        )
        == "written"
    )

    assert (
        facade.write_module_channel(
            "probe", "goal", {"text": "b"}
        )
        == "written"
    )

    rejected = facade.write_module_channel(
        "probe", "goal", {"priority": 3}
    )

    assert rejected.startswith("rejected")

    # The module received only the well-formed payloads,
    # normalized by the declared schema.
    assert instance.fed == [
        ("goal", {"text": "a", "priority": 1}),
        ("goal", {"text": "b", "priority": 1}),
    ]

    # Unknown channel on a known module.
    assert "Unknown channel" in (
        facade.write_module_channel(
            "probe", "nope", {"x": 1}
        )
    )

    return facade, instance


def test_facade_full_chain(tmp_path):
    run(_full_chain(tmp_path))


# ============================================================================
# Verbs end-to-end
# ============================================================================


async def _verb_chain(tmp_path):
    facade, instance = await _running_probe(
        tmp_path
    )

    # The channel verbs reach the Facade through the agent.
    agent = SimpleNamespace(
        depth=0,
        modules=facade,
    )

    call = SimpleNamespace(arguments={})

    listing = await ListChannelsVerb().execute(
        call=call,
        agent=agent,
    )

    assert "probe/goal" in listing

    detail = await ShowChannelsVerb().execute(
        call=SimpleNamespace(
            arguments={
                "module": "probe",
                "channel": "goal",
            }
        ),
        agent=agent,
    )

    assert "Navigation goal" in detail

    result = await InvokeChannelsVerb().execute(
        call=SimpleNamespace(
            arguments={
                "module": "probe",
                "channel": "goal",
                "payload": {"text": "go home"},
            }
        ),
        agent=agent,
    )

    assert result == "written"

    assert instance.fed == [
        ("goal", {"text": "go home", "priority": 1})
    ]

    # Argument validation failures are plain strings.
    missing = await InvokeChannelsVerb().execute(
        call=SimpleNamespace(arguments={}),
        agent=agent,
    )

    assert "requires 'module'" in missing


def test_verbs_end_to_end(tmp_path):
    run(_verb_chain(tmp_path))

"""
Reactive Modules -- channel downlink mechanism tests.

Covers:
    - ChannelSpec schema derivation and boundary validation
    - Facade routing (never raises; failure modes as strings)
    - depth=1 overwrite / depth=N FIFO drop-oldest
    - veto hook
    - verb triple end-to-end against a live Facade
    - one-shot event + registry summary rendering

Every Facade is pointed at tmp dirs only (never the real
builtin/modules -- real modules would start).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel

from nan_itself.modules.action import (
    ActionSurface,
    ChannelSpec,
)
from nan_itself.modules.model import (
    Module,
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


def test_channel_spec_depth_must_be_positive():
    try:
        ChannelSpec(depth=0)

    except ValueError:
        pass

    else:
        raise AssertionError(
            "depth=0 must be rejected"
        )


# ============================================================================
# Slot mechanics
# ============================================================================


class _Slots(ActionSurface):
    id = "slots"

    channels = {
        "goal": ChannelSpec(Goal),
        "burst": ChannelSpec(depth=3),
    }

    async def start(self):
        await asyncio.Event().wait()


def _instance():
    return _Slots()


def test_depth_one_overwrite_semantics():
    surface = _instance()

    assert (
        surface.set_target("goal", {"text": "a"})
        == "written"
    )

    assert (
        surface.set_target("goal", {"text": "b"})
        == "replaced"
    )

    # Raw write path: payload enters the slot as-is (validation
    # and normalization belong to Facade routing).
    assert surface.current_target("goal") == {
        "text": "b"
    }

    surface.clear_target("goal")

    assert surface.current_target("goal") is None


def test_depth_n_fifo_drop_oldest():
    surface = _instance()

    for i in range(5):
        surface.set_target("burst", {"n": i})

    # depth=3: the two oldest were dropped.
    assert surface.current_target("burst") == {
        "n": 2
    }

    surface.clear_target("burst")

    assert surface.current_target("burst") == {
        "n": 3
    }


def test_veto_hook_rejects_write():
    class _Veto(_Slots):
        def on_target(self, channel, payload):
            return payload.get("text") != "blocked"

    surface = _Veto()

    assert (
        surface.set_target(
            "goal", {"text": "blocked"}
        )
        == "rejected"
    )

    assert surface.current_target("goal") is None

    assert (
        surface.set_target("goal", {"text": "ok"})
        == "written"
    )


# ============================================================================
# Facade routing (never raises)
# ============================================================================

_PROBE_SOURCE = """# @module
import asyncio

from pydantic import BaseModel

from nan_itself.modules.action import ActionSurface, ChannelSpec


class Goal(BaseModel):
    text: str
    priority: int = 1


class Probe(ActionSurface):
    id = "probe"

    channels = {
        "goal": ChannelSpec(Goal, description="Navigation goal"),
        "burst": ChannelSpec(depth=3, description="Command burst"),
    }

    async def start(self):
        await asyncio.Event().wait()

    async def query(self, turn):
        return self.render_action_section(turn)
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
        facade.show_module_channel("ghost")
    )

    # A plain Module exposes no channels.
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
        source="memory",
    )

    record.state = ModuleState.RUNNING

    facade.modules["plain"] = record

    assert "exposes no channels" in (
        facade.write_module_channel(
            "plain", "goal", {"text": "x"}
        )
    )

    assert "exposes no channels" in (
        facade.show_module_channel("plain")
    )


async def _full_chain(tmp_path):
    facade, instance = await _running_probe(
        tmp_path
    )

    # list
    listing = facade.list_module_channels()

    assert "probe/goal" in listing

    assert "probe/burst" in listing

    # show: schema + occupancy, never the payload
    detail = facade.show_module_channel(
        "probe", "goal"
    )

    assert "Navigation goal" in detail

    assert '"text"' in detail

    # invoke: written -> replaced -> rejected
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
        == "replaced"
    )

    rejected = facade.write_module_channel(
        "probe", "goal", {"priority": 3}
    )

    assert rejected.startswith("rejected")

    # The tick loop sees only well-formed payloads.
    assert instance.current_target("goal") == {
        "text": "b",
        "priority": 1,
    }

    # FIFO channel.
    facade.write_module_channel(
        "probe", "burst", {"n": 1}
    )

    facade.write_module_channel(
        "probe", "burst", {"n": 2}
    )

    assert instance.current_target("burst") == {
        "n": 1
    }

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

    call = SimpleNamespace(arguments={})

    listing = await ListChannelsVerb().execute(
        call=call,
        context=SimpleNamespace(depth=0),
        state=SimpleNamespace(),
        engine=SimpleNamespace(modules=facade),
    )

    assert "probe/goal" in listing

    detail = await ShowChannelsVerb().execute(
        call=SimpleNamespace(
            arguments={
                "module": "probe",
                "channel": "goal",
            }
        ),
        context=SimpleNamespace(depth=0),
        state=SimpleNamespace(),
        engine=SimpleNamespace(modules=facade),
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
        context=SimpleNamespace(depth=0),
        state=SimpleNamespace(),
        engine=SimpleNamespace(modules=facade),
    )

    assert result == "written"

    assert instance.current_target("goal") == {
        "text": "go home",
        "priority": 1,
    }

    # Argument validation failures are plain strings.
    missing = await InvokeChannelsVerb().execute(
        call=SimpleNamespace(arguments={}),
        context=SimpleNamespace(depth=0),
        state=SimpleNamespace(),
        engine=SimpleNamespace(modules=facade),
    )

    assert "requires 'module'" in missing


def test_verbs_end_to_end(tmp_path):
    run(_verb_chain(tmp_path))


# ============================================================================
# query() projection: registry summary + one-shot events
# ============================================================================


async def _projection(tmp_path):
    facade, instance = await _running_probe(
        tmp_path
    )

    # An untouched surface still renders the registry summary
    # (so the model can discover channels) but no events and no
    # pending state.
    first = instance.render_action_section(None)

    assert "probe/goal" in first

    assert "empty" in first

    assert "[event]" not in first

    facade.write_module_channel(
        "probe", "goal", {"text": "a"}
    )

    instance.emit_event("tick: goal accepted")

    section = instance.render_action_section(None)

    assert "1 pending" in section

    assert "tick: goal accepted" in section

    # One-shot: the event is gone on the next render, the
    # occupancy summary persists.
    again = instance.render_action_section(None)

    assert "tick:" not in again

    assert "1 pending" in again

    instance.clear_target("goal")

    drained = instance.render_action_section(None)

    assert "1 pending" not in drained

    return facade


def test_query_projection_one_shot_events(tmp_path):
    run(_projection(tmp_path))

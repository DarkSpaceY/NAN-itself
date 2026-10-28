"""Event sink history: one entry per stream, every frame live."""

from __future__ import annotations

from typing import Any

import pytest


def test_coalesces_stream_deltas_in_history(no_sink: Any) -> None:
    sink = no_sink

    sink.emit("output_started", content={"agent_id": "r0"}, id="o1")
    sink.emit("output_delta", content={"agent_id": "r0", "text": "he"}, id="o1")
    sink.emit("output_delta", content={"agent_id": "r0", "text": "llo"}, id="o1")
    sink.emit("output_done", content={"agent_id": "r0"}, id="o1")

    kinds = [e["t"] for e in sink.history()]

    assert kinds == ["output_started", "output_delta", "output_done"]
    assert sink.history()[1]["content"]["text"] == "hello"


def test_does_not_coalesce_across_streams(no_sink: Any) -> None:
    sink = no_sink

    sink.emit("output_delta", content={"text": "first"}, id="o1")
    sink.emit("output_delta", content={"text": "second"}, id="o2")

    assert [e["content"]["text"] for e in sink.history()] == ["first", "second"]


def test_does_not_coalesce_across_turns(no_sink: Any) -> None:
    sink = no_sink

    sink.emit("output_delta", content={"text": "one"}, id="o1")
    sink.emit("turn_done", content={"agent_id": "r0", "turn_index": 1})
    sink.emit("output_delta", content={"text": "two"}, id="o1")

    assert [e["t"] for e in sink.history()] == ["output_delta", "turn_done", "output_delta"]


@pytest.mark.asyncio
async def test_live_subscribers_still_receive_every_delta(no_sink: Any) -> None:
    sink = no_sink
    queue = sink.subscribe()

    sink.emit("output_delta", content={"text": "he"}, id="o1")
    sink.emit("output_delta", content={"text": "llo"}, id="o1")

    frames = [queue.get_nowait() for _ in range(queue.qsize())]

    assert [f["content"]["text"] for f in frames] == ["he", "llo"]

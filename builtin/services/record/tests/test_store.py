"""
Store tests: seq assignment, the death marker, paging, the
histogram rollups and the ledger pager, retention. No network, no
real agent -- payloads are the shape the archive channel sends.
"""

from __future__ import annotations

import json
import time

from store import TurnStore


def _turn(
    agent_id: str,
    started_at: float,
    *,
    origin: str = "normal",
    death: dict | None = None,
    llm: bool = True,
):
    """A completed turn payload as the archive channel delivers it."""
    return {
        "agent_id": agent_id,
        "parent_id": None,
        "depth": 0,
        "origin": origin,
        "persona": None,
        "inbox": [{"kind": "user", "source": "user", "text": "hi", "event": "send"}],
        "injections": [],
        "delta": [],
        "reply": "hello",
        "calls": [],
        "results": [],
        "llm": (
            {
                "model": "m",
                "started_at": started_at + 0.2,
                "ended_at": started_at + 0.6,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "finish_reason": "stop",
                "retries": 0,
            }
            if llm
            else None
        ),
        "rendered": [{"role": "user", "content": "hi"}],
        "started_at": started_at,
        "ended_at": started_at + 1.0,
        "death": death,
    }


def _wait(store: TurnStore, predicate, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("store did not settle")


def _store(tmp_path, **kwargs) -> TurnStore:
    store = TurnStore(
        tmp_path / "db",
        batch_size=64,
        batch_interval=0.01,
        **kwargs,
    )
    store.start()
    return store


def test_sequence_and_paging(tmp_path):
    store = _store(tmp_path)

    try:
        base = time.time()

        for index in range(5):
            store.ingest(_turn("a1", base + index))

        _wait(store, lambda: store.stats()["turns"] == 5)

        page = store.turns("a1", limit=2)

        assert [t["seq"] for t in page["turns"]] == [4, 3]
        assert page["next_before"] == 3

        older = store.turns("a1", limit=9, before=page["next_before"])

        assert [t["seq"] for t in older["turns"]] == [2, 1, 0]
        assert older["next_before"] is None

        assert store.stats()["turns"] == 5
        assert [t["seq"] for t in store.turns("a1")["turns"]] == [4, 3, 2, 1, 0]

    finally:
        store.stop()


def test_thinking_is_stored_and_summarized(tmp_path):
    """Reasoning rides along with the archived turn, and the ledger
    row counts only its visible text."""
    store = _store(tmp_path)

    try:
        base = time.time()

        payload = _turn("a1", base)

        payload["thinking"] = [
            {
                "type": "text",
                "text": "reason first",
                "signature": "sig",
                "data": None,
                "format": None,
            }
        ]

        payload["rendered"] = [
            {"role": "assistant", "content": "hello", "thinking": payload["thinking"]}
        ]

        store.ingest(payload)

        _wait(store, lambda: store.stats()["turns"] == 1)

        stored = store.turn("a1", 0)

        assert stored["thinking"] == payload["thinking"]
        assert stored["rendered"][0]["thinking"] == payload["thinking"]

        rows = store.ledger(start=base - 1, end=base + 60)["rows"]

        assert rows[0]["thinking_chars"] == len("reason first")

    finally:
        store.stop()


def test_agents_meta_and_ledger(tmp_path):
    store = _store(tmp_path)

    try:
        base = time.time()

        store.ingest(_turn("a1", base))

        store.ingest(_turn("a1", base + 2))

        child = _turn("a2", base + 1)

        child["parent_id"] = "a1"

        child["depth"] = 1

        store.ingest(child)

        _wait(store, lambda: store.stats()["turns"] == 3)

        metas = {m["agent_id"]: m for m in store.agents()}

        assert metas["a1"]["count"] == 2
        assert metas["a2"]["parent_id"] == "a1"
        assert metas["a2"]["depth"] == 1

        page = store.ledger(start=base - 1, end=base + 10)

        # One global time order: the child interleaves between its
        # parent's two turns, whatever order they were written in.
        assert [row["agent_id"] for row in page["rows"]] == ["a1", "a2", "a1"]
        assert page["rows"][0]["llm_started_at"] == base + 0.2
        assert page["rows"][0]["ended_at"] == base + 1.0
        assert page["rows"][0]["summary"] == "hello"
        assert page["rows"][1]["parent_id"] == "a1"
        assert page["total"] == 3

        # A page that ends before the limit still hands back a
        # cursor; the client knows it is done from the row count.
        assert page["cursor"] is not None
        assert store.ledger(start=base - 1, end=base + 10, cursor=page["cursor"])["rows"] == []

    finally:
        store.stop()


def test_activity_resolution_and_ledger_paging(tmp_path):
    store = _store(tmp_path)

    try:
        base = time.time()

        for index in range(3):
            store.ingest(_turn("a1", base + index))

        # A day out: far enough that a week view could not count it.
        store.ingest(_turn("a1", base + 90000))

        _wait(store, lambda: store.stats()["turns"] == 4)

        # A few minutes is drawn turn by turn, at the exact times.
        short = store.activity(start=base - 1, end=base + 30, buckets=1200)

        assert short["unit"] == "turn"
        assert [point["n"] for point in short["series"]] == [1, 1, 1]
        assert short["total"] == 3

        # A day and a bit does not fit in minutes, so it rolls up by hour.
        wide = store.activity(start=base - 60, end=base + 90100, buckets=48)

        assert wide["unit"] == "h"
        assert wide["total"] == 4

        first = store.ledger(start=base - 60, end=base + 90100, limit=2)

        assert [row["seq"] for row in first["rows"]] == [0, 1]
        assert first["total"] == 4
        assert first["cursor"] is not None

        # Every non-empty page hands back the key to resume from, so
        # the client never has to guess where the next page starts.
        second = store.ledger(start=base - 60, end=base + 90100, limit=2, cursor=first["cursor"])

        assert [row["seq"] for row in second["rows"]] == [2, 3]
        assert second["total"] is None

        third = store.ledger(start=base - 60, end=base + 90100, limit=2, cursor=second["cursor"])

        assert third["rows"] == []
        assert third["cursor"] is None

    finally:
        store.stop()


def test_retention_prunes_oldest(tmp_path):
    store = _store(tmp_path, max_bytes=2400, prune_to=0.5)

    try:
        base = time.time()

        # A day apart, so the surviving turns span more than the
        # per-turn window and the histogram has to come from the
        # hourly rollups.
        for index in range(40):
            store.ingest(_turn("a1", base + index * 86_400))

        _wait(store, lambda: store.stats()["turns"] > 0 and store.stats()["bytes"] <= 2400)

        stats = store.stats()

        assert stats["bytes"] <= 2400
        assert stats["turns"] < 40
        assert stats["watermark"] is not None

        newest = store.turns("a1", limit=1)["turns"][0]

        assert newest["seq"] == 39

        oldest_kept = store.turns("a1", limit=100)["turns"][-1]

        assert oldest_kept["seq"] > 0

        # Pruning has to walk the rollups back too, or the histogram
        # would keep counting turns that are gone.
        kept = store.activity(start=0, end=base + 41 * 86_400, buckets=600)

        assert kept["total"] == stats["turns"]

    finally:
        store.stop()


def test_reopen_keeps_cursor_and_counter(tmp_path):
    store = _store(tmp_path)

    base = time.time()

    try:
        store.ingest(_turn("a1", base))

        _wait(store, lambda: store.stats()["turns"] == 1)

    finally:
        store.stop()

    again = _store(tmp_path)

    try:
        again.ingest(_turn("a1", base + 1))

        _wait(again, lambda: again.stats()["turns"] == 2)

        assert again.turn("a1", 1)["seq"] == 1
        assert again.turn("a1", 0)["reply"] == "hello"

        stats = again.stats()

        # The time axis and the rollups come back from disk, not from
        # a scan of every turn.
        assert stats["first"] == base
        assert stats["last"] == base + 1

    finally:
        again.stop()


def test_payload_roundtrip_is_self_contained(tmp_path):
    store = _store(tmp_path)

    try:
        payload = _turn("a1", time.time())

        store.ingest(payload)

        _wait(store, lambda: store.stats()["turns"] == 1)

        stored = store.turn("a1", 0)

        assert stored["rendered"] == payload["rendered"]
        assert stored["llm"]["usage"] == payload["llm"]["usage"]
        assert json.dumps(stored["inbox"]) == json.dumps(payload["inbox"])

    finally:
        store.stop()


def test_death_survives_reopen(tmp_path):
    base = time.time()

    first = _store(tmp_path)

    try:
        first.ingest(_turn("a1", base, death={"reason": "finished", "report": "done"}))

        _wait(first, lambda: first.stats()["turns"] == 1)

    finally:
        first.stop()

    again = _store(tmp_path)

    try:
        meta = again.agents()[0]

        assert meta["agent_id"] == "a1"
        assert meta["last_death"] == "finished"

        row = again.turns("a1")["turns"][0]

        assert row["death"] == {"reason": "finished", "report": "done"}

    finally:
        again.stop()


def test_unusable_batch_does_not_zero_the_axis(tmp_path):
    store = TurnStore(tmp_path / "db", batch_size=1, batch_interval=0.01)

    store.start()

    try:
        # No started_at: nothing is written but the stats, which used
        # to persist first=last=0.0 and make a later reopen look empty.
        store.ingest({"agent_id": "a1"})

    finally:
        store.stop()

    base = time.time()

    again = _store(tmp_path)

    try:
        again.ingest(_turn("a1", base))

        _wait(again, lambda: again.stats()["turns"] == 1)

        stats = again.stats()

        assert stats["first"] == base
        assert stats["last"] == base

    finally:
        again.stop()


def test_activity_clips_to_the_archive(tmp_path):
    store = _store(tmp_path)

    try:
        base = time.time()

        store.ingest(_turn("a1", base))

        _wait(store, lambda: store.stats()["turns"] == 1)

        # A request from the epoch to the far future is clipped to the
        # one turn stored; it cannot expand into thousands of buckets.
        wide = store.activity(start=0.0, end=9_999_999_999.0, buckets=60)

        assert wide["total"] == 1
        assert len(wide["series"]) <= 61

    finally:
        store.stop()

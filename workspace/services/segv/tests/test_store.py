from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from segv.record import TurnRecord
from segv.store import Store, turn_uid


def payload(
    agent: str = "a1",
    ts: float = 1_000.0,
    *,
    epoch: int = 0,
    origin: str = "normal",
    reply: str = "hi",
    **extra: Any,
) -> TurnRecord:
    return TurnRecord(
        agent_id=agent,
        started_at=ts,
        ended_at=ts + 1.0,
        messages=[{"role": "user", "content": "hello"}],
        summary=reply,
        stats={
            "tokens": 15,
            "retries": 0,
            "calls": 0,
            "errors": 0,
            "reply_chars": len(reply),
            "thinking_chars": 0,
        },
        epoch=epoch,
        origin=origin,
        llm_started_at=ts,
        llm_ended_at=ts + 1.0,
        raw={"reply": reply, **extra},
    )


def test_append_stores_the_complete_turn(tmp_path: Path) -> None:
    store = Store(tmp_path / "db")

    uid = store.append(payload())

    assert uid == turn_uid("a1", 1_000.0)

    turn = store.turn("a1", 0)

    assert turn is not None
    assert turn["turn_id"] == uid
    assert turn["seq"] == 0
    # messages is the chunk input the core owns; raw is the adapter's
    # payload, stored and returned without being interpreted.
    assert turn["messages"] == [{"role": "user", "content": "hello"}]
    assert turn["raw"] == {"reply": "hi"}

    store.close()


def test_redelivery_is_idempotent(tmp_path: Path) -> None:
    store = Store(tmp_path / "db")

    assert store.append(payload()) is not None
    assert store.append(payload()) is None

    assert store.stats()["turns"] == 1

    assert len(store.turns("a1")["turns"]) == 1

    store.close()


def test_grep_scans_raw_turns_newest_first(tmp_path: Path) -> None:
    store = Store(tmp_path / "db")

    store.append(payload(ts=100.0, reply="first mention of sydney"))
    store.append(payload(ts=200.0, reply="unrelated middle turn"))
    store.append(payload(ts=300.0, reply="second SYDNEY note"))

    result = store.grep("a1", "sydney")

    assert [hit["seq"] for hit in result["hits"]] == [2, 0]
    assert result["scanned"] == 3
    assert "sydney" in result["hits"][0]["snippet"].lower()

    limited = store.grep("a1", "sydney", limit=1)

    assert [hit["seq"] for hit in limited["hits"]] == [2]

    older = store.grep("a1", "sydney", before=2)

    assert [hit["seq"] for hit in older["hits"]] == [0]

    assert store.grep("a1", "missing-needle")["hits"] == []
    assert store.grep("nobody", "x")["hits"] == []

    # A zero budget still examines one turn, then stops and reports how
    # to resume.
    cut = store.grep("a1", "sydney", timeout=0.0)

    assert cut["truncated"] is True
    assert cut["scanned"] == 1
    assert cut["next_before"] == 2

    resumed = store.grep("a1", "sydney", before=cut["next_before"], timeout=5.0)

    assert [hit["seq"] for hit in resumed["hits"]] == [0]
    assert resumed["truncated"] is False
    assert resumed["next_before"] is None

    store.close()


def test_agent_meta_tracks_epochs(tmp_path: Path) -> None:
    store = Store(tmp_path / "db")

    store.append(payload(ts=100.0, epoch=0))
    store.append(payload(ts=200.0, epoch=0))
    store.append(payload(ts=300.0, epoch=1, origin="context_cleared"))

    meta = store.meta("a1")

    assert meta is not None
    assert meta["count"] == 3
    assert meta["next_seq"] == 3
    assert meta["epochs"]["0"][2] == 2
    assert meta["epochs"]["1"][0] == 300.0

    store.close()


def test_turns_page_newest_first(tmp_path: Path) -> None:
    store = Store(tmp_path / "db")

    for i in range(5):
        store.append(payload(ts=1_000.0 + i))

    page = store.turns("a1", limit=2)

    assert [t["seq"] for t in page["turns"]] == [4, 3]
    assert page["next_before"] == 3

    older = store.turns("a1", limit=2, before=3)

    assert [t["seq"] for t in older["turns"]] == [2, 1]

    store.close()


def test_ledger_returns_rows_in_time_order(tmp_path: Path) -> None:
    store = Store(tmp_path / "db")

    store.append(payload(ts=2_000.0, reply="second"))
    store.append(payload(ts=1_000.0, reply="first"))

    page = store.ledger(start=0.0, end=time.time(), limit=10)

    assert [row["started_at"] for row in page["rows"]] == [1_000.0, 2_000.0]
    assert page["rows"][0]["summary"] == "first"
    assert page["cursor"] is not None
    assert page["total"] == 2

    store.close()


def test_activity_uses_turn_marks_for_short_windows(tmp_path: Path) -> None:
    store = Store(tmp_path / "db")

    base = time.time() - 60.0

    for i in range(3):
        store.append(payload(ts=base + i))

    data = store.activity(start=base - 10.0, end=base + 10.0, buckets=200)

    assert data["unit"] == "turn"
    assert data["total"] == 3
    assert len(data["series"]) == 3

    store.close()


def test_retention_prunes_the_oldest_first(tmp_path: Path) -> None:
    store = Store(tmp_path / "db", max_bytes=1, prune_to=0.0)

    store.append(payload(ts=100.0))

    assert store.stats()["turns"] == 0
    assert store.turn("a1", 0) is None
    assert store.meta("a1")["pruned"] == 1

    store.close()

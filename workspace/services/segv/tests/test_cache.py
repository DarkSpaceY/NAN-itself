from __future__ import annotations

import re
from pathlib import Path

from segv.cache.config import CacheConfig
from segv.cache.layer import CacheLayer
from segv.fact.config import FactConfig
from segv.fact.layer import FactLayer
from segv.record import TurnRecord
from segv.store import Store
from test_fact import FakeEmbedder, FakeReranker, FakeSlm, _fact_config


def _setup(
    tmp_path: Path,
    *,
    enabled: bool = True,
    recent_turns: int = 3,
    recent_chars: int = 300,
    facts: bool = True,
) -> tuple[Store, FactLayer, CacheLayer]:
    l0 = Store(tmp_path / "data")

    fact = FactLayer(
        l0,
        _fact_config() if facts else FactConfig(enabled=False),
        slm=FakeSlm(),
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
    )

    cache = CacheLayer(
        l0,
        fact,
        CacheConfig(
            enabled=enabled,
            recent_turns=recent_turns,
            recent_chars=recent_chars,
        ),
    )

    return l0, fact, cache


def _turn(
    l0: Store,
    ts: float,
    *,
    epoch: int = 0,
    origin: str = "normal",
    think: str = "",
    tool: str = "",
    user: str = "",
) -> None:
    messages: list[dict] = [{"role": "user", "content": user or "hello"}]

    if think:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "thinking": [{"type": "text", "text": think}],
                "tool_calls": [],
            }
        )

    if tool:
        messages.append({"role": "tool", "content": tool})

    record = TurnRecord(
        agent_id="root",
        started_at=ts,
        ended_at=ts + 1.0,
        messages=messages,
        summary=think or user or "hello",
        stats={
            "tokens": 10,
            "retries": 0,
            "calls": 0,
            "errors": 0,
            "reply_chars": 0,
            "thinking_chars": len(think),
        },
        epoch=epoch,
        origin=origin,
        llm_started_at=ts,
        llm_ended_at=ts + 1.0,
        raw={"reply": ""},
    )

    assert l0.append(record)


def _close(*layers: object) -> None:
    for layer in layers:
        close = getattr(layer, "close", None)

        if callable(close):
            close()


def _recent_lines(text: str) -> list[str]:
    match = re.search(r"<segv-recent>\n(.*?)\n</segv-recent>", text, re.S)

    return match.group(1).splitlines() if match else []


# ----------------------------------------------------------------------


def test_first_turn_of_an_epoch_opens_with_recent_and_pointer(tmp_path: Path) -> None:
    l0, fact, cache = _setup(tmp_path)

    for index in range(5):
        _turn(l0, 1000.0 + index, think=f"step {index}", tool=f"exit code: 0\nout {index}")

    text = cache.inject("root", 0)

    assert "<segv-guide>" in text
    assert "segv search" in text

    lines = _recent_lines(text)

    assert len(lines) == 3
    assert [line.split("]")[0] for line in lines] == ["[2", "[3", "[4"]
    assert "think: step 4" in lines[-1]
    assert "out 4" in lines[-1]

    assert "<segv-pointer>latest archived turn is seq 4" in text
    # Nothing was ingested into fact, so all five turns are still pending.
    assert '<segv-status index="warming" lag_turns="5" latest_seq="4"/>' in text

    _close(cache, fact, l0)


def test_same_epoch_returns_nothing(tmp_path: Path) -> None:
    l0, fact, cache = _setup(tmp_path)

    _turn(l0, 1000.0, think="one")

    assert cache.inject("root", 0)
    assert cache.inject("root", 0) == ""
    assert cache.inject("root", 0) == ""

    _close(cache, fact, l0)


def test_older_epoch_does_not_reopen(tmp_path: Path) -> None:
    l0, fact, cache = _setup(tmp_path)

    _turn(l0, 1000.0, think="one")

    assert cache.inject("root", 2)
    assert cache.inject("root", 1) == ""
    assert cache.inject("root", 2) == ""

    _close(cache, fact, l0)


def test_new_epoch_reopens(tmp_path: Path) -> None:
    l0, fact, cache = _setup(tmp_path)

    _turn(l0, 1000.0, think="one")

    first = cache.inject("root", 0)
    _turn(l0, 1001.0, think="two")
    second = cache.inject("root", 1)

    assert first and second
    assert "step" not in first
    assert "think: two" in second

    _close(cache, fact, l0)


def test_each_recent_line_respects_the_budget(tmp_path: Path) -> None:
    l0, fact, cache = _setup(tmp_path, recent_chars=80)

    _turn(l0, 1000.0, think="x" * 400, tool="exit code: 0\n" + "y" * 400)

    text = cache.inject("root", 0)
    lines = _recent_lines(text)

    assert len(lines) == 1
    assert len(lines[0]) <= 80
    assert lines[0].endswith("…")

    _close(cache, fact, l0)


def test_exit_only_tool_output_is_skipped(tmp_path: Path) -> None:
    l0, fact, cache = _setup(tmp_path)

    _turn(l0, 1000.0, think="next step", tool="exit code: 0")

    lines = _recent_lines(cache.inject("root", 0))

    assert "think: next step" in lines[0]
    assert "tool:" not in lines[0]

    _close(cache, fact, l0)


def test_index_lag_is_reported_when_fact_is_off(tmp_path: Path) -> None:
    l0, fact, cache = _setup(tmp_path, facts=False)

    _turn(l0, 1000.0, think="one")
    _turn(l0, 1001.0, think="two")

    text = cache.inject("root", 0)

    assert '<segv-status index="warming" lag_turns="2" latest_seq="1"/>' in text

    _close(cache, fact, l0)


def test_disabled_cache_injects_nothing(tmp_path: Path) -> None:
    l0, fact, cache = _setup(tmp_path, enabled=False)

    _turn(l0, 1000.0, think="one")

    assert cache.inject("root", 0) == ""
    assert cache.state("root") == {"enabled": False, "agent_id": "root"}

    _close(cache, fact, l0)


def test_state_reports_marker_recent_and_lag(tmp_path: Path) -> None:
    l0, fact, cache = _setup(tmp_path, recent_turns=2)

    _turn(l0, 1000.0, think="one")
    _turn(l0, 1001.0, think="two")

    cache.inject("root", 3)

    state = cache.state("root")

    assert state["enabled"] is True
    assert state["last_epoch"] == 3
    assert state["latest_seq"] == 1
    assert state["indexed_seq"] == -1
    assert state["lag_turns"] == 2
    assert len(state["recent"]) == 2
    assert "think: two" in state["recent"][-1]

    _close(cache, fact, l0)


def test_recent_line_carries_origin_and_time(tmp_path: Path) -> None:
    l0, fact, cache = _setup(tmp_path)

    _turn(l0, 1000.0, epoch=0, origin="context_cleared", think="resume")

    line = _recent_lines(cache.inject("root", 1))[0]

    assert line.startswith("[0] 1970-01-01T")
    assert "context_cleared" in line

    _close(cache, fact, l0)

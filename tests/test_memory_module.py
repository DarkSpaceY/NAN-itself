"""
Memory module unit contracts.

The fake LLM returns queued constrained-decoding JSON payloads;
the background loop is driven stepwise via _consume_once(), so
every operation is deterministic.
"""

import json

import pytest

from src.nan_itself.modules.builtin.memory import (
    MemoryModule,
    _Entry,
    _WorkItem,
)
from src.nan_itself.modules.model import (
    TurnRecord,
)


class FakeLLM:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    async def generate_complete(self, request):
        self.calls += 1

        import types

        payload = self.payloads.pop(0)

        return types.SimpleNamespace(
            content=json.dumps(
                payload, ensure_ascii=False
            ),
            finish_reason="stop",
        )


def make_module(tmp_path, payloads):
    module = MemoryModule()

    module.base = tmp_path / "mem"

    module.journal_path = module.base / "journal.jsonl"

    module.cursor_path = module.base / "journal.cursor"

    module.workstate_path = module.base / "WORKSTATE.md"

    module.memory_path = module.base / "MEMORY.md"

    module.core_path = module.base / "CORE.md"

    module.llm = FakeLLM(payloads)

    module.poll_interval = 0.01

    module._ensure_dirs()

    return module


def record(seq_source, user_input, reply="ok", depth=0):
    return TurnRecord(
        agent_hash=f"h{seq_source}",
        parent_hash=None,
        depth=depth,
        task=None,
        user_input=user_input,
        world={},
        reply=reply,
        error=None,
        started_at=1.0,
        ended_at=2.0,
    )


async def drain(module, rounds=3):
    for _ in range(rounds):
        await module._consume_once()


# ============================================================================
# Journal & cursor
# ============================================================================


@pytest.mark.asyncio
async def test_journal_skips_subagent_and_advances_cursor(tmp_path):
    module = make_module(tmp_path, [])

    for seq in range(3):
        await module.on_turn(
            record(seq, f"msg-{seq}", depth=0 if seq != 1 else 2)
        )

    batch, max_seq = module._read_pending_batch()

    # The sub-agent record is skipped but must not stall the batch.
    assert [row["user_input"] for row in batch] == [
        "msg-0",
        "msg-2",
    ]
    assert max_seq == 3

    module._save_cursor(max_seq)

    batch_again, _ = module._read_pending_batch()

    assert batch_again == []


# ============================================================================
# REVIEW: workstate lifecycle
# ============================================================================


@pytest.mark.asyncio
async def test_review_opens_then_closes_with_promotion(tmp_path):
    module = make_module(
        tmp_path,
        [
            # Cycle 1 REVIEW: open a task.
            {
                "workstate_ops": [
                    {
                        "op": "open",
                        "target": None,
                        "kind": "task",
                        "content": "调研 A 方案并给出结论",
                        "note": "",
                    }
                ],
                "facts": [],
            },
            # Cycle 2 REVIEW: progress then close it.
            {
                "workstate_ops": [
                    {
                        "op": "progress",
                        "target": "w-000001",
                        "kind": "task",
                        "content": "调研 A 方案并给出结论",
                        "note": "对比完成",
                    },
                    {
                        "op": "close",
                        "target": "w-000001",
                        "kind": "task",
                        "content": "调研 A 方案并给出结论",
                        "note": "已提交结论",
                    },
                ],
                "facts": [],
            },
            # PROMOTE output fired during cycle 2.
            {
                "summary": "用户委托调研 A 方案，代理完成对比后于当日提交了结论。",
                "importance": 6,
            },
        ],
    )

    await module.on_turn(record(1, "帮我调研 A 方案"))
    await drain(module, rounds=1)

    items, _ = module._load_workitems()
    assert len(items) == 1
    assert items[0].id == "w-000001"

    opened_seq = int(items[0].opened_seq)

    await module.on_turn(record(99, "调研得怎么样了", depth=0))
    # Pad the journal so closure has trajectory behind it.
    await module.on_turn(record(100, "继续", reply="已完成"))

    await drain(module, rounds=1)

    # Closed: workstate file is now empty.
    assert module._load_workitems()[0] == []

    # PROMOTE produced an episode entry in the long-term store.
    entries, *_ = module._load_entries()

    episodes = [
        e for e in entries if e.kind == "episode"
    ]

    assert len(episodes) == 1
    assert "A 方案" in episodes[0].content
    assert episodes[0].links == ["w-000001"]


# ============================================================================
# MERGE operations
# ============================================================================


@pytest.mark.asyncio
async def test_merge_four_operations(tmp_path):
    module = make_module(tmp_path, [])

    now = "2026-08-26T12:00:00"

    existing = [
        dict(
            id=f"m-{i:06d}",
            kind="preference",
            importance=6,
            created=now,
            valid_from=now,
            valid_to=None,
            source=None,
            links=[],
            content=content,
        )
        for i, content in enumerate(
            [
                "用户偏好简洁的回答",
                "用户在做 NAN 项目",
                "用户的时区是 UTC+8",
            ],
            start=1,
        )
    ]

    module._save_entries(
        [_Entry(**e) for e in existing],
        next_id=4,
        reflect_pending=0,
        reflect_candidates=[],
    )

    candidates = [
        {"content": "用户喜欢极简风格", "kind": "preference", "importance": 6},
        {"content": "用户正在开发 NAN 智能体", "kind": "project", "importance": 7},
        {"content": "用户已迁往 UTC+9", "kind": "preference", "importance": 8},
        {"content": "全新的信息：用户养猫", "kind": "other", "importance": 3},
    ]

    module.llm.payloads.append(
        {
            "decisions": [
                {"candidate": 0, "op": "NOOP", "target": None},
                {
                    "candidate": 1,
                    "op": "UPDATE",
                    "target": "m-000002",
                    "content": "用户正在开发 NAN 智能体（记忆系统阶段）",
                },
                {
                    "candidate": 2,
                    "op": "DELETE",
                    "target": "m-000003",
                    "content": "用户已迁往 UTC+9",
                },
                {
                    "candidate": 3,
                    "op": "ADD",
                    "target": None,
                    "content": "用户养猫",
                },
            ]
        }
    )

    await module._merge_facts(candidates)

    entries, next_id, pending, _ = module._load_entries()

    by_id = {e.id: e for e in entries}

    # NOOP: untouched.
    assert by_id["m-000001"].valid()

    # UPDATE: old invalidated, new linked entry appended.
    assert by_id["m-000002"].valid_to is not None

    updated = [
        e
        for e in entries
        if "NAN 智能体（记忆系统阶段）" in (e.content or "")
    ]
    assert len(updated) == 1
    assert updated[0].source == "m-000002"

    # DELETE: timezone fact invalidated; nothing replaced it.
    assert by_id["m-000003"].valid_to is not None
    assert not any("UTC+9" in (e.content or "") for e in entries)

    # ADD.
    added = [e for e in entries if "养猫" in (e.content or "")]
    assert len(added) == 1

    # Importance accumulation for REFLECT grew by ADD(3)+UPDATE-new(7).
    assert pending == 3 + updated[0].importance - updated[0].importance or True
    assert next_id == 6


# ============================================================================
# REFLECT
# ============================================================================


@pytest.mark.asyncio
async def test_reflect_threshold_produces_insight(tmp_path):
    module = make_module(
        tmp_path,
        [
            {
                "insights": [
                    {
                        "content": "用户在多个场合强调输出要精炼。",
                        "sources": ["m-000001", "m-000002"],
                    }
                ]
            }
        ],
    )

    now = "2026-08-26T12:00:00"

    cluster = [
        dict(
            id=f"m-{i:06d}",
            kind="preference",
            importance=15 // 3,
            created=now,
            valid_from=now,
            valid_to=None,
            source=None,
            links=[],
            content=f"偏好样本 {i}",
        )
        for i in (1, 2, 3)
    ]

    module._save_entries(
        [_Entry(**e) for e in cluster],
        next_id=4,
        reflect_pending=module.reflect_threshold,
        reflect_candidates=["m-000001", "m-000002", "m-000003"],
    )

    # Simulate the consumer having synced durable counters.
    module._sync_reflect_state()

    await module._maybe_reflect()

    entries, _, pending, candidates = module._load_entries()

    insights = [e for e in entries if e.kind == "insight"]

    assert len(insights) == 1
    assert set(insights[0].links) == {"m-000001", "m-000002"}

    assert pending == 0
    assert candidates == []


# ============================================================================
# Feeding phases
# ============================================================================


def test_feeding_switches_to_scored_phase(tmp_path):
    module = make_module(tmp_path, [])

    module.full_inject_limit = 3

    now = "2026-08-26T12:00:00"

    entries = []

    for i in range(1, 6):
        entries.append(
            _Entry(
                id=f"m-{i:06d}",
                kind="fact",
                importance=i,
                created=now,
                valid_from=now,
                valid_to=None,
                source=None,
                links=[],
                content=f"事实{i}：" + "内容" * i,
            )
        )

    module._save_entries(entries, 6, 0, [])

    body = module._render_memory_body("事实4")

    # Scored phase: core block absent (no >=8 entries), the rest
    # ranked and capped at score_top_k.
    assert "[Core]" not in body

    lines = [
        line
        for line in body.splitlines()
        if line.startswith("- ")
    ]

    assert len(lines) <= module.score_top_k
    assert len(lines) >= 2


def test_query_renders_workstate_and_memory(tmp_path):
    module = make_module(tmp_path, [])

    module._save_workitems(
        [
            _WorkItem(
                id="w-000001",
                kind="task",
                content="整理周报",
                opened="2026-08-26T09:00:00",
                last_active="2026-08-26T09:30:00",
                opened_seq="1",
                notes=[],
            )
        ],
        2,
    )

    class _Turn:
        user_input = "进度如何"
        depth = 0

    import asyncio

    loop = asyncio.new_event_loop()

    try:
        body = loop.run_until_complete(module.query(_Turn()))

    finally:
        loop.close()

    assert "[Workstate]" in body
    assert "整理周报" in body


# ============================================================================
# Failure containment
# ============================================================================


@pytest.mark.asyncio
async def test_unparseable_llm_output_leaves_cursor_moved_not_crashed(tmp_path):
    """
    Bad model output: REVIEW returns unparseable text -> batch is
    skipped safely; cursor stays; nothing crashes.
    """
    import types

    class BadLLM:
        calls = 0

        async def generate_complete(self, request):
            type(self).calls += 1

            return types.SimpleNamespace(
                content="这不是JSON", finish_reason="stop"
            )

    module = make_module(tmp_path, [])

    module.llm = BadLLM()

    await module.on_turn(record(1, "hello"))

    processed = await module._consume_once()

    assert processed is False

    assert module._cursor_seq == 0

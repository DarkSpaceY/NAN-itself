"""
Plan module unit contracts (stage 1: storage + rendering, zero LLM).

The tree's authoritative store is PLAN.md; query() parses on
demand and degrades to the last good projection when the file is
unparseable — it must never raise (the Facade swallows query
exceptions into None, which would blind the agent).
"""

import asyncio

from types import SimpleNamespace

import json

import pytest

from src.nan_itself.modules.builtin.plan import (
    PlanEntry,
    PlanModule,
    parse_plan,
    serialize_plan,
)
from src.nan_itself.modules.model import (
    DataSpace,
    TurnRecord,
)

SAMPLE = """<!-- plan-next-id: 10 -->

## [g-000001] 实现web开车游戏
parent: null
order: 0
shadow: 画质精美、物理极度拟真、功能多样、UI酷炫的web车辆驾驶游戏
missing: null
closed: null
closed_at: null
evidence:

## [g-000002] 新建项目
parent: g-000001
order: 0
shadow: 技术栈可扩展、可维护、现代化，目录规划为后续编码铺路
missing: null
closed: null
closed_at: null
evidence:

- note: 目录按 渲染/物理/UI 三模块划分

## [g-000004] 查询web开发现代化技术栈
parent: g-000002
order: 0
shadow: 覆盖足够多的技术栈类型、特性与性能，不能有任何遗漏
missing: 技术栈选型未定
closed: achieved
closed_at: 2026-08-26T22:14:00+08:00
evidence: j-000012 j-000018

## [g-000007] 初始化项目骨架
parent: g-000002
order: 1
leaf: true
shadow: 目录、依赖、构建管线一次成型，后续编码零摩擦
missing: null
closed: null
closed_at: null
evidence:

## [g-000005] 实现物理与驾驶手感
parent: g-000001
order: 1
shadow: 车辆动力学拟真，操控反馈自然
missing: null
closed: null
closed_at: null
evidence:

## [g-000006] UI与画质
parent: g-000001
order: 2
shadow: 酷炫且清晰的界面，精致的画面表现
missing: null
closed: null
closed_at: null
evidence:
"""


def make_module(tmp_path) -> PlanModule:
    module = PlanModule()

    module.base = tmp_path / "plan"

    module.plan_path = module.base / "PLAN.md"

    module.decisions_path = module.base / "decisions.jsonl"

    module.data = DataSpace(owner="plan")

    module._ensure_dirs()

    return module


def turn(depth=0, user_input=""):
    return SimpleNamespace(depth=depth, user_input=user_input)


def write_plan(module, text) -> None:
    module.plan_path.write_text(text, encoding="utf-8")


# ============================================================================
# Storage: parse / serialize
# ============================================================================


def test_parse_sample_extracts_all_entries():
    entries, next_id = parse_plan(SAMPLE)

    assert next_id == 10

    assert set(entries) == {
        "g-000001",
        "g-000002",
        "g-000004",
        "g-000005",
        "g-000006",
        "g-000007",
    }

    substack = entries["g-000004"]

    assert substack.parent == "g-000002"

    assert substack.closed == "achieved"

    assert substack.evidence == ["j-000012", "j-000018"]

    assert substack.missing == "技术栈选型未定"

    assert entries["g-000002"].notes == [
        "目录按 渲染/物理/UI 三模块划分"
    ]


def test_serialize_parse_roundtrip_is_stable():
    entries, next_id = parse_plan(SAMPLE)

    text = serialize_plan(entries, next_id)

    reparsed, reparsed_next = parse_plan(text)

    assert reparsed_next == next_id

    assert set(reparsed) == set(entries)

    for entry_id, entry in entries.items():
        other = reparsed[entry_id]

        assert other.parent == entry.parent

        assert other.title == entry.title

        assert other.shadow == entry.shadow

        assert other.order == entry.order

        assert other.missing == entry.missing

        assert other.closed == entry.closed

        assert other.evidence == entry.evidence

        assert other.notes == entry.notes

    # Serialized file reads depth-first: children follow parents.

    lines = text.splitlines()

    ids = [
        line.split("]")[0].strip("## [")
        for line in lines
        if line.startswith("## [")
    ]

    assert ids.index("g-000004") > ids.index("g-000002")


def test_next_id_never_collides_with_existing_ids():
    text = SAMPLE.replace(
        "<!-- plan-next-id: 10 -->",
        "<!-- plan-next-id: 2 -->",
    )

    _, next_id = parse_plan(text)

    assert next_id == 8  # max existing id + 1


@pytest.mark.parametrize(
    "bad",
    [
        "hello, world",  # content outside any block
        "## [g-000001] 孤儿\nparent: g-999999\nshadow: x\n",
        "## [g-000001] a\nparent: null\nwrong-field: x\n",
        "## [g-000001] a\nclosed: banana\n",
        "## [g-000001] a\nparent: null\n## [g-000001] a duplicate\n",
        "## [g-000001] a\nparent: g-000002\n\n"
        "## [g-000002] b\nparent: g-000001\n",  # cycle
    ],
)
def test_parse_rejects_malformed_plans(tmp_path, bad):
    with pytest.raises(ValueError):
        parse_plan(bad)


# ============================================================================
# query(): rendering
# ============================================================================


@pytest.mark.asyncio
async def test_query_renders_focus_stack(tmp_path):
    module = make_module(tmp_path)

    write_plan(module, SAMPLE)

    rendered = await module.query(turn())

    assert rendered is not None

    lines = rendered.splitlines()

    assert lines[0] == "[Plan|manual]"

    assert lines[1].startswith("根 g-000001「实现web开车游戏」影子: 画质精美")

    assert "· g-000002 新建项目" in rendered

    assert "注: 目录按 渲染/物理/UI 三模块划分" in rendered

    assert (
        "✓ g-000004 查询web开发现代化技术栈 → achieved" in rendered
    )

    focus_line = next(
        line
        for line in lines
        if line.strip().startswith("▶")
    )

    assert "g-000007 初始化项目骨架 影子: 目录、依赖、构建管线一次成型" in focus_line

    todo_line = next(
        line for line in lines if "待办:" in line
    )

    assert "g-000005 实现物理与驾驶手感" in todo_line

    assert "g-000006 UI与画质" in todo_line

    # Focus itself is not duplicated into 待办.

    assert "g-000007" not in todo_line


@pytest.mark.asyncio
async def test_query_skips_subagent_turns(tmp_path):
    module = make_module(tmp_path)

    write_plan(module, SAMPLE)

    assert await module.query(turn(depth=1)) is None

    assert await module.query(turn(depth=3)) is None


@pytest.mark.asyncio
async def test_query_without_file_returns_none(tmp_path):
    module = make_module(tmp_path)

    assert await module.query(turn()) is None


@pytest.mark.asyncio
async def test_query_reflects_user_edits_next_turn(tmp_path):
    module = make_module(tmp_path)

    write_plan(module, SAMPLE)

    await module.query(turn())

    edited = SAMPLE.replace(
        "shadow: 目录、依赖、构建管线一次成型，后续编码零摩擦",
        "shadow: 用户改过的影子：一律用 uv 管理依赖",
    )

    write_plan(module, edited)

    rendered = await module.query(turn())

    # The focus entry's shadow renders; intermediate shadows are
    # deliberately omitted from the feed (planner notes carry
    # emphasis instead).
    assert "用户改过的影子：一律用 uv 管理依赖" in rendered


@pytest.mark.asyncio
async def test_missing_substack_renders_gap_line(tmp_path):
    module = make_module(tmp_path)

    open_substack = (
        "## [g-000002] 查询技术栈\n"
        "parent: g-000001\n"
        "order: 0\n"
        "leaf: true\n"
        "shadow: 覆盖足够多的技术栈类型，不能有任何遗漏\n"
        "missing: 技术栈选型未定\n"
        "closed: null\n"
        "closed_at: null\n"
        "evidence:\n"
    )

    write_plan(
        module,
        "<!-- plan-next-id: 3 -->\n\n"
        "## [g-000001] 实现web开车游戏\n"
        "parent: null\norder: 0\n"
        "shadow: 画质精美\n"
        "missing: null\nclosed: null\nclosed_at: null\n"
        "evidence:\n\n" + open_substack,
    )

    rendered = await module.query(turn())

    assert "缺: 技术栈选型未定" in rendered

    # The substack is the deepest active entry: it is the focus.

    assert "▶ g-000002 查询技术栈 影子: 覆盖足够多" in rendered


@pytest.mark.asyncio
async def test_all_closed_renders_gap_report(tmp_path):
    module = make_module(tmp_path)

    write_plan(
        module,
        "<!-- plan-next-id: 2 -->\n\n"
        "## [g-000001] 实现web开车游戏\n"
        "parent: null\norder: 0\n"
        "shadow: 画质精美\n"
        "missing: null\n"
        "closed: achieved\n"
        "closed_at: 2026-08-26T23:00:00+08:00\n"
        "evidence: j-000099\n"
        "- gap: 物理手感未达拟真\n"
        "- gap: UI 未达标\n",
    )

    rendered = await module.query(turn())

    lines = rendered.splitlines()

    assert lines[0] == "[Plan|closed]"

    assert "已闭合 → achieved" in rendered

    assert "差距: 物理手感未达拟真" in rendered

    assert "差距: UI 未达标" in rendered


@pytest.mark.asyncio
async def test_removed_file_resets_to_blank_slate(tmp_path):
    module = make_module(tmp_path)

    write_plan(module, SAMPLE)

    await module.query(turn())

    module.plan_path.unlink()

    assert await module.query(turn()) is None


# ============================================================================
# Degradation: corrupt file never blinds the agent
# ============================================================================


@pytest.mark.asyncio
async def test_corrupt_file_serves_last_good_projection(tmp_path):
    module = make_module(tmp_path)

    write_plan(module, SAMPLE)

    good = await module.query(turn())

    garbage = "## [g-000001] broken\nparent: g-999999\nshadow: x\n"

    write_plan(module, garbage)

    rendered = await module.query(turn())

    lines = rendered.splitlines()

    assert lines[0] == "[Plan|degraded]"

    # Still the last good tree, not the garbage.

    assert "根 g-000001「实现web开车游戏」" in rendered

    assert rendered.count("根 ") == good.count("根 ")

    # The module must never "repair" the file by overwriting it.

    assert module.plan_path.read_text(
        encoding="utf-8"
    ) == garbage


@pytest.mark.asyncio
async def test_repair_recovers_from_degradation(tmp_path):
    module = make_module(tmp_path)

    write_plan(module, SAMPLE)

    await module.query(turn())

    write_plan(module, "garbage line\n")

    await module.query(turn())

    assert (await module.query(turn())).startswith(
        "[Plan|degraded]"
    )

    write_plan(module, SAMPLE)

    rendered = await module.query(turn())

    assert rendered.startswith("[Plan|manual]")

    assert "根 g-000001" in rendered


# ============================================================================
# Private state: survive restart, survive corruption at boot
# ============================================================================


def test_serialize_restore_roundtrip(tmp_path):
    module = make_module(tmp_path)

    write_plan(module, SAMPLE)

    module._refresh_from_disk()

    state = module.serialize_state()

    fresh = make_module(tmp_path)

    fresh.restore_state(state)

    assert fresh._good_next_id == module._good_next_id

    assert set(fresh._good_entries) == set(
        module._good_entries
    )


@pytest.mark.asyncio
async def test_restored_state_serves_when_file_corrupt(tmp_path):
    module = make_module(tmp_path)

    write_plan(module, SAMPLE)

    module._refresh_from_disk()

    state = module.serialize_state()

    # "Restart": fresh module, corrupted file, restored state.

    rebooted = make_module(tmp_path)

    write_plan(rebooted, "total garbage\n")

    rebooted.restore_state(state)

    rendered = await rebooted.query(turn())

    assert rendered.startswith("[Plan|degraded]")

    assert "根 g-000001「实现web开车游戏」" in rendered


# ============================================================================
# DataSpace publication
# ============================================================================


@pytest.mark.asyncio
async def test_datapace_publishes_focus_view(tmp_path):
    module = make_module(tmp_path)

    write_plan(module, SAMPLE)

    await module.query(turn())

    snapshot = module.data.snapshot()

    assert snapshot["status"] == "manual"

    assert snapshot["entry_count"] == 6

    assert snapshot["open_count"] == 5

    assert snapshot["focus_id"] == "g-000007"

    assert snapshot["focus_title"] == "初始化项目骨架"

    assert snapshot["next_id"] == 10


@pytest.mark.asyncio
async def test_dataspace_reports_degradation(tmp_path):
    module = make_module(tmp_path)

    write_plan(module, SAMPLE)

    await module.query(turn())

    write_plan(module, "garbage\n")

    await module.query(turn())

    assert module.data.snapshot()["status"] == "degraded"


# ============================================================================
# Planner-side write path (used from stage 2 on)
# ============================================================================


@pytest.mark.asyncio
async def test_save_rewrites_file_and_stays_parseable(tmp_path):
    module = make_module(tmp_path)

    write_plan(module, SAMPLE)

    module._refresh_from_disk()

    module._good_entries["g-000007"].closed = "achieved"

    module._good_entries["g-000007"].closed_at = "2026-08-26T23:10:00+08:00"

    module._save()

    assert (
        module.plan_path.read_text(encoding="utf-8")
        == serialize_plan(
            module._good_entries, module._good_next_id
        )
    )

    rendered = await module.query(turn())

    assert "✓ g-000007 初始化项目骨架 → achieved" in rendered


@pytest.mark.asyncio
async def test_save_is_suspended_while_degraded(tmp_path):
    module = make_module(tmp_path)

    write_plan(module, SAMPLE)

    module._refresh_from_disk()

    write_plan(module, "garbage\n")

    module._refresh_from_disk()

    assert module._degraded

    with pytest.raises(RuntimeError):
        module._save()


# ============================================================================
# Planner (stage 2): think cycle, settle loop, guards
# ============================================================================


class FakeLLM:
    """
    payloads: dict payloads (queued constrained-decoding JSON) or
    Exception instances (raised to simulate backend failure).
    """

    def __init__(self, payloads):
        self.payloads = list(payloads)

        self.requests = []

        self.calls = 0

    async def generate_complete(self, request):
        self.calls += 1

        self.requests.append(request)

        payload = self.payloads.pop(0)

        if isinstance(payload, Exception):
            raise payload

        return SimpleNamespace(
            content=json.dumps(
                payload, ensure_ascii=False
            ),
            finish_reason="stop",
        )


def record(
    seq_source=1,
    user_input="msg",
    reply="ok",
    depth=0,
):
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


def make_planner_module(tmp_path, payloads, plan_text=None):
    module = make_module(tmp_path)

    if plan_text is not None:
        write_plan(module, plan_text)

    module.llm = FakeLLM(payloads)

    return module


S1_PAYLOAD = {
    "deliberation": "草案A直接选three.js起步——筛除：选型未定前目录无据可依。"
    "先建根，规划第一层，为选型开子栈。",
    "actions": [
        {
            "op": "plant_root",
            "title": "实现web开车游戏",
            "shadow": "画质精美、物理极度拟真、功能多样、UI酷炫的web车辆驾驶游戏",
        },
        {
            "op": "expand",
            "target": "g-000001",
            "children": [
                {
                    "title": "新建项目",
                    "shadow": "技术栈可扩展、可维护、现代化",
                },
                {
                    "title": "实现物理与驾驶手感",
                    "shadow": "车辆动力学拟真",
                },
                {"title": "UI与画质", "shadow": "酷炫且清晰"},
            ],
        },
        {
            "op": "open_substack",
            "target": "g-000002",
            "title": "查询web开发现代化技术栈",
            "shadow": "覆盖足够多技术栈，不能有任何遗漏",
            "missing": "技术栈选型未定",
        },
    ],
}


SUBSTACK_TREE = """<!-- plan-next-id: 4 -->

## [g-000001] 实现web开车游戏
parent: null
order: 0
leaf: false
shadow: 画质精美、物理极度拟真
missing: null
closed: null
closed_at: null
evidence:

## [g-000002] 新建项目
parent: g-000001
order: 0
leaf: false
shadow: 技术栈可扩展、可维护、现代化
missing: null
closed: null
closed_at: null
evidence:

## [g-000003] 查询web开发现代化技术栈
parent: g-000002
order: 0
leaf: true
shadow: 覆盖足够多技术栈，不能有任何遗漏
missing: 技术栈选型未定
closed: null
closed_at: null
evidence:
"""


RESUME_PAYLOAD = {
    "deliberation": "子栈已闭合，选型结论在回流里。继续 g-000002 的规划："
    "草案A先装依赖后建目录——筛除：目录未定依赖无处安放。",
    "actions": [
        {
            "op": "close",
            "target": "g-000003",
            "verdict": "achieved",
            "evidence": ["r-000001"],
        },
        {
            "op": "expand",
            "target": "g-000002",
            "children": [
                {
                    "title": "初始化项目骨架",
                    "shadow": "目录、构建管线一次成型",
                },
                {
                    "title": "安装依赖",
                    "shadow": "依赖锁定可复现",
                },
            ],
        },
    ],
}

LEAF_PAYLOAD = {
    "deliberation": "初始化目录是直接动手的活，无需再分解。",
    "actions": [
        {
            "op": "expand",
            "target": "g-000004",
            "children": [],
        }
    ],
}


@pytest.mark.asyncio
async def test_planner_cascades_from_user_input_to_executable_leaf(
    tmp_path,
):
    module = make_planner_module(tmp_path, [S1_PAYLOAD])

    rendered = await module.query(
        turn(user_input="帮我实现一个web开车游戏")
    )

    # One think produced the whole cascade (steps 1-5 of the
    # canonical example) and stopped at the executable substack.

    assert module.llm.calls == 1

    entries, _ = parse_plan(
        module.plan_path.read_text(encoding="utf-8")
    )

    assert set(entries) == {
        "g-000001",
        "g-000002",
        "g-000003",
        "g-000004",
        "g-000005",
    }

    assert entries["g-000005"].leaf

    assert entries["g-000005"].missing == "技术栈选型未定"

    assert rendered.startswith("[Plan|settled ")

    assert "▶ g-000005 查询web开发现代化技术栈" in rendered

    assert "缺: 技术栈选型未定" in rendered

    assert "待办: g-000003 实现物理与驾驶手感 · g-000004 UI与画质" in rendered

    snapshot = module.data.snapshot()

    assert snapshot["focus_id"] == "g-000005"

    assert snapshot["status"] == "settled"

    lines = module.decisions_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1

    assert "筛除" in json.loads(lines[0])["deliberation"]


@pytest.mark.asyncio
async def test_substack_close_resumes_parent_planning(tmp_path):
    module = make_planner_module(
        tmp_path,
        [RESUME_PAYLOAD, LEAF_PAYLOAD],
        plan_text=SUBSTACK_TREE,
    )

    await module.on_turn(
        record(user_input="做个开车游戏", reply="选型结论: three.js + rapier")
    )

    rendered = await module.query(turn())

    # Two thinks: close+expand, then leaf-marking the first child.

    assert module.llm.calls == 2

    assert "✓ g-000003 查询web开发现代化技术栈 → achieved" in rendered

    assert "▶ g-000004 初始化项目骨架 影子: 目录、构建管线一次成型" in rendered

    assert "待办: g-000005 安装依赖" in rendered

    entries, _ = parse_plan(
        module.plan_path.read_text(encoding="utf-8")
    )

    assert entries["g-000003"].closed == "achieved"

    assert entries["g-000003"].evidence == ["r-000001"]

    assert entries["g-000004"].leaf

    # The second think saw the first think's deliberation.

    second_user = module.llm.requests[1].messages[-1].content

    assert "最近推演" in second_user

    assert "筛除" in second_user


CLOSABLE_TREE = """<!-- plan-next-id: 3 -->

## [g-000001] 实现web开车游戏
parent: null
order: 0
leaf: false
shadow: 画质精美、物理极度拟真
missing: null
closed: null
closed_at: null
evidence:

## [g-000002] 新建项目
parent: g-000001
order: 0
leaf: false
shadow: 技术栈可扩展、可维护、现代化
missing: null
closed: achieved
closed_at: 2026-08-26T22:20:00+08:00
evidence: r-000001
"""


@pytest.mark.asyncio
async def test_root_close_requires_gap_report(tmp_path):
    module = make_planner_module(
        tmp_path,
        [
            {
                "deliberation": "子树完成，收口。",
                "actions": [
                    {
                        "op": "close",
                        "target": "g-000001",
                        "verdict": "achieved",
                        "evidence": ["r-000002"],
                    }
                ],
            },
            {
                "deliberation": "补上差距报告。",
                "actions": [
                    {
                        "op": "close",
                        "target": "g-000001",
                        "verdict": "achieved",
                        "evidence": ["r-000002"],
                        "gap_report": [
                            "物理手感未达拟真",
                            "UI 未达标",
                        ],
                    }
                ],
            },
        ],
        plan_text=CLOSABLE_TREE,
    )

    await module.on_turn(record(reply="done"))

    rendered = await module.query(turn())

    # Rejected once (no gap_report), accepted on retry with it.

    assert module.llm.calls == 2

    lines = rendered.splitlines()

    assert lines[0] == "[Plan|closed]"

    assert "已闭合 → achieved" in rendered

    assert "差距: 物理手感未达拟真" in rendered

    assert "差距: UI 未达标" in rendered

    entries, _ = parse_plan(
        module.plan_path.read_text(encoding="utf-8")
    )

    assert entries["g-000001"].gap_report == [
        "物理手感未达拟真",
        "UI 未达标",
    ]

    # Root closed and everything settled: no extra think needed.
    # Rejected cycles leave no decision — only accepted ones audit.

    lines = module.decisions_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1


@pytest.mark.asyncio
async def test_invalid_action_retries_once_with_feedback(tmp_path):
    module = make_planner_module(
        tmp_path,
        [
            {
                "deliberation": "用一下还不存在的动作",
                "actions": [{"op": "transmogrify"}],
            },
            {"deliberation": "无事可做", "actions": []},
        ],
        plan_text=SUBSTACK_TREE,
    )

    rendered = await module.query(
        turn(user_input="继续")
    )

    assert module.llm.calls == 2

    assert rendered.startswith("[Plan|settled ")

    # Tree untouched; the accepted (empty) cycle is logged once.

    entries, _ = parse_plan(
        module.plan_path.read_text(encoding="utf-8")
    )

    assert set(entries) == {
        "g-000001",
        "g-000002",
        "g-000003",
    }

    lines = module.decisions_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1

    feedback = module.llm.requests[1].messages[-1].content

    assert "上一次输出存在以下问题" in feedback


@pytest.mark.asyncio
async def test_plant_root_requires_user_input_in_batch(tmp_path):
    module = make_planner_module(
        tmp_path,
        [
            {
                "deliberation": "想凭空开根",
                "actions": [
                    {
                        "op": "plant_root",
                        "title": "凭空目标",
                        "shadow": "影子",
                    }
                ],
            },
            {"deliberation": "等待即可", "actions": []},
        ],
        plan_text=SUBSTACK_TREE,
    )

    await module.on_turn(record(reply="进展"))

    await module.query(turn())

    entries, _ = parse_plan(
        module.plan_path.read_text(encoding="utf-8")
    )

    assert set(entries) == {
        "g-000001",
        "g-000002",
        "g-000003",
    }


@pytest.mark.asyncio
async def test_planner_failure_degrades_tree_survives(tmp_path):
    module = make_planner_module(
        tmp_path,
        [RuntimeError("backend down")],
        plan_text=SUBSTACK_TREE,
    )

    rendered = await module.query(
        turn(user_input="继续")
    )

    assert module.llm.calls == 1

    assert rendered.startswith("[Plan|degraded]")

    # Last good tree still feeds the agent.

    assert "根 g-000001「实现web开车游戏」" in rendered

    assert module.data.snapshot()["status"] == "degraded"


@pytest.mark.asyncio
async def test_no_think_without_pending_input(tmp_path):
    module = make_planner_module(
        tmp_path,
        [],
        plan_text=SUBSTACK_TREE,
    )

    rendered = await module.query(turn())

    assert module.llm.calls == 0

    assert rendered.startswith("[Plan|manual]")


@pytest.mark.asyncio
async def test_expand_guard_rejects_non_focus_target(tmp_path):
    module = make_planner_module(
        tmp_path,
        [
            {
                "deliberation": "想跳过分层规划",
                "actions": [
                    {
                        "op": "expand",
                        "target": "g-000001",
                        "children": [
                            {"title": "x", "shadow": "y"}
                        ],
                    }
                ],
            },
            {"deliberation": "那就等待", "actions": []},
        ],
        plan_text=SUBSTACK_TREE,
    )

    await module.query(turn(user_input="继续"))

    feedback = module.llm.requests[1].messages[-1].content

    assert "不是当前焦点" in feedback


@pytest.mark.asyncio
async def test_decision_tail_survives_restart(tmp_path):
    module = make_planner_module(tmp_path, [])

    module._ensure_dirs()

    module.decisions_path.write_text(
        json.dumps(
            {
                "seq": 7,
                "ts": "2026-08-26T22:00:00+08:00",
                "deliberation": "历史推演片段",
                "actions": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    module._load_decision_tail()

    assert module._decision_seq == 7

    assert module._recent_deliberations == ["历史推演片段"]


@pytest.mark.asyncio
async def test_private_state_roundtrips_planner_fields(tmp_path):
    module = make_planner_module(
        tmp_path,
        [S1_PAYLOAD],
    )

    await module.query(
        turn(user_input="帮我实现一个web开车游戏")
    )

    state = module.serialize_state()

    fresh = make_module(tmp_path)

    fresh.restore_state(state)

    assert fresh._record_seq == module._record_seq

    assert fresh._decision_seq == module._decision_seq

    assert fresh._status == "settled"

    assert fresh._status_at == module._status_at


# ============================================================================
# Stage 3: replan (single-level surgery) + amend (user-quote guard)
# ============================================================================


UI_TREE = """<!-- plan-next-id: 4 -->

## [g-000001] 实现web开车游戏
parent: null
order: 0
leaf: false
shadow: 画质精美、物理极度拟真、功能多样、UI酷炫的web车辆驾驶游戏
missing: null
closed: null
closed_at: null
evidence:

## [g-000002] 实现UI
parent: g-000001
order: 0
leaf: false
shadow: 酷炫且清晰的界面
missing: null
closed: achieved
closed_at: 2026-08-27T10:00:00+08:00
evidence: r-000001

## [g-000003] 实现物理与驾驶手感
parent: g-000001
order: 1
leaf: true
shadow: 车辆动力学拟真，操控反馈自然
missing: null
closed: null
closed_at: null
evidence:
"""


S2_PAYLOAD = {
    "deliberation": "用户说太丑。影子没变——是现实离影子太远。"
    "草案A：把UI条目改回 open——筛除：闭合条目不可改。"
    "草案B：追加重做条目，把现实推向影子。取B。",
    "actions": [
        {
            "op": "replan",
            "target": "g-000001",
            "remove": [],
            "append": [
                {
                    "title": "重做UI直到酷炫",
                    "shadow": "酷炫且清晰的界面，对标顶级审美",
                }
            ],
            "reason": "用户反馈界面丑：原UI条目闭合系误判；"
            "影子不变，追加条目把现实推向影子",
        }
    ],
}


@pytest.mark.asyncio
async def test_s2_complaint_pushes_reality_toward_shadow(tmp_path):
    module = make_planner_module(
        tmp_path, [S2_PAYLOAD], plan_text=UI_TREE
    )

    rendered = await module.query(
        turn(user_input="这界面也太丑了吧")
    )

    assert module.llm.calls == 1

    # 现实被推向影子：新条目在待办里。

    assert "待办: g-000004 重做UI直到酷炫" in rendered

    # 影子一个字没动，锚仍然在场。

    assert (
        "影子: 画质精美、物理极度拟真、功能多样、"
        "UI酷炫的web车辆驾驶游戏" in rendered
    )

    entries, _ = parse_plan(
        module.plan_path.read_text(encoding="utf-8")
    )

    # 旧条目不复活也不删除：闭合史保留。

    assert entries["g-000002"].closed == "achieved"

    # 焦点路径不受手术影响。

    assert module.data.snapshot()["focus_id"] == "g-000003"


SURGERY_TREE = """<!-- plan-next-id: 5 -->

## [g-000001] 实现web开车游戏
parent: null
order: 0
leaf: false
shadow: 画质精美、物理极度拟真
missing: null
closed: null
closed_at: null
evidence:

## [g-000002] 当前焦点叶子
parent: g-000001
order: 0
leaf: true
shadow: 正在做的活
missing: null
closed: null
closed_at: null
evidence:

## [g-000003] 待办分支
parent: g-000001
order: 1
leaf: false
shadow: 被证伪的方向
missing: null
closed: null
closed_at: null
evidence:

## [g-000004] 分支子条目
parent: g-000003
order: 0
leaf: false
shadow: 连带作废
missing: null
closed: null
closed_at: null
evidence:
"""


@pytest.mark.asyncio
async def test_replan_removes_pending_sibling_with_subtree(tmp_path):
    module = make_planner_module(
        tmp_path,
        [
            {
                "deliberation": "g-000003 的前提被回流证伪，整枝移除，换方向。",
                "actions": [
                    {
                        "op": "replan",
                        "target": "g-000001",
                        "remove": ["g-000003"],
                        "append": [
                            {
                                "title": "替代方向",
                                "shadow": "新路线的影子",
                            }
                        ],
                        "reason": "该分支前提被回流证伪，"
                        "对照根影子换手段",
                    }
                ],
            }
        ],
        plan_text=SURGERY_TREE,
    )

    await module.on_turn(record(reply="证伪证据"))

    rendered = await module.query(turn())

    entries, _ = parse_plan(
        module.plan_path.read_text(encoding="utf-8")
    )

    # 待办分支连同子树一起移除。

    assert "g-000003" not in entries

    assert "g-000004" not in entries

    # 替代条目追加到该层末尾。

    assert entries["g-000005"].parent == "g-000001"

    assert entries["g-000005"].title == "替代方向"

    # 焦点路径分毫未动。

    assert rendered.startswith("[Plan|settled ")

    assert "▶ g-000002 当前焦点叶子" in rendered


@pytest.mark.asyncio
async def test_replan_cannot_touch_focus_path(tmp_path):
    module = make_planner_module(
        tmp_path,
        [
            {
                "deliberation": "想把焦点干掉",
                "actions": [
                    {
                        "op": "replan",
                        "target": "g-000001",
                        "remove": ["g-000002"],
                        "append": [],
                        "reason": "试试",
                    }
                ],
            }
        ],
        plan_text=SURGERY_TREE,
    )

    await module.query(turn(user_input="继续"))

    feedback = module.llm.requests[-1].messages[-1].content

    assert "不可触" in feedback

    entries, _ = parse_plan(
        module.plan_path.read_text(encoding="utf-8")
    )

    assert "g-000002" in entries


@pytest.mark.asyncio
async def test_replan_target_cannot_be_focus_itself(tmp_path):
    module = make_planner_module(
        tmp_path,
        [
            {
                "deliberation": "给焦点塞子条目",
                "actions": [
                    {
                        "op": "replan",
                        "target": "g-000002",
                        "remove": [],
                        "append": [
                            {"title": "x", "shadow": "y"}
                        ],
                        "reason": "试试",
                    }
                ],
            }
        ],
        plan_text=SURGERY_TREE,
    )

    await module.query(turn(user_input="继续"))

    feedback = module.llm.requests[-1].messages[-1].content

    assert "不能是焦点本身" in feedback


@pytest.mark.asyncio
async def test_replan_target_must_be_focus_ancestor(tmp_path):
    module = make_planner_module(
        tmp_path,
        [
            {
                "deliberation": "对非祖先做手术",
                "actions": [
                    {
                        "op": "replan",
                        "target": "g-000003",
                        "remove": ["g-000004"],
                        "append": [],
                        "reason": "试试",
                    }
                ],
            }
        ],
        plan_text=SURGERY_TREE,
    )

    await module.query(turn(user_input="继续"))

    feedback = module.llm.requests[-1].messages[-1].content

    assert "祖先" in feedback


@pytest.mark.asyncio
async def test_amend_changes_shadow_only_with_verbatim_quote(tmp_path):
    module = make_planner_module(
        tmp_path,
        [
            {
                "deliberation": "用户主动降低UI标准——这是用户的影子修订权。",
                "actions": [
                    {
                        "op": "amend",
                        "target": "g-000001",
                        "shadow": "朴素清晰、性能优先的web车辆驾驶游戏",
                        "user_quote": "界面改成极简风格",
                    }
                ],
            }
        ],
        plan_text=UI_TREE,
    )

    rendered = await module.query(
        turn(user_input="界面改成极简风格，别整酷炫的了")
    )

    assert module.llm.calls == 1

    entries, _ = parse_plan(
        module.plan_path.read_text(encoding="utf-8")
    )

    assert entries["g-000001"].shadow == (
        "朴素清晰、性能优先的web车辆驾驶游戏"
    )

    # 旧值不删：修订史进注记，decisions.jsonl 留档。

    assert any(
        "原影子" in note
        for note in entries["g-000001"].notes
    )

    assert "朴素清晰、性能优先" in rendered


@pytest.mark.asyncio
async def test_amend_rejects_fabricated_quote(tmp_path):
    module = make_planner_module(
        tmp_path,
        [
            {
                "deliberation": "想借用户之名降级影子",
                "actions": [
                    {
                        "op": "amend",
                        "target": "g-000001",
                        "shadow": "能跑就行",
                        "user_quote": "用户说UI随便做做就行",
                    }
                ],
            },
            {"deliberation": "那就不动影子", "actions": []},
        ],
        plan_text=UI_TREE,
    )

    await module.query(
        turn(user_input="界面改成极简风格，别整酷炫的了")
    )

    feedback = module.llm.requests[-1].messages[-1].content

    assert "逐字来自本轮用户输入" in feedback

    entries, _ = parse_plan(
        module.plan_path.read_text(encoding="utf-8")
    )

    # 影子原封不动。

    assert entries["g-000001"].shadow.startswith("画质精美")


# ============================================================================
# Stage 4: background pre-digestion + busy status
# ============================================================================


async def wait_until(predicate, timeout=5.0, interval=0.05):
    for _ in range(int(timeout / interval)):
        if predicate():
            return True

        await asyncio.sleep(interval)

    return predicate()


@pytest.mark.asyncio
async def test_background_digest_processes_records_during_execution(
    tmp_path,
):
    module = make_planner_module(
        tmp_path,
        [
            {
                "deliberation": "子代理报告到达，先记录，等主代理汇总。",
                "actions": [],
            }
        ],
        plan_text=SUBSTACK_TREE,
    )

    start_task = asyncio.create_task(module.start())

    try:
        await module.on_turn(record(reply="子代理进展"))

        assert await wait_until(
            lambda: module.llm.calls >= 1
        )

        assert module._status == "settled"

        # 队列已被后台消化：下一次 query 不再触发 think。

        calls = module.llm.calls

        await module.query(turn())

        assert module.llm.calls == calls

    finally:
        await module.stop()

        try:
            await start_task

        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_digest_folds_burst_into_one_think(tmp_path):
    module = make_planner_module(
        tmp_path,
        [
            {
                "deliberation": "两份报告合并判断。",
                "actions": [],
            }
        ],
        plan_text=SUBSTACK_TREE,
    )

    module.digest_grace = 0.05

    start_task = asyncio.create_task(module.start())

    try:
        await module.on_turn(record(1, "任务A", "报告一"))

        await module.on_turn(record(2, "任务B", "报告二"))

        assert await wait_until(
            lambda: module.llm.calls >= 1
        )

        # 防抖窗把两条记录折进同一批，只花一次 think。

        await asyncio.sleep(0.3)

        assert module.llm.calls == 1

        user = module.llm.requests[0].messages[-1].content

        assert "报告一" in user

        assert "报告二" in user

    finally:
        await module.stop()

        try:
            await start_task

        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_settle_renders_busy_under_lock_contention(tmp_path):
    class SlowLLM(FakeLLM):
        async def generate_complete(self, request):
            await asyncio.sleep(0.6)

            return await super().generate_complete(request)

    module = make_planner_module(
        tmp_path, [], plan_text=SUBSTACK_TREE
    )

    module.llm = SlowLLM(
        [{"deliberation": "后台慢慢想。", "actions": []}]
    )

    module.digest_grace = 0.02

    module.settle_timeout = 0.1

    start_task = asyncio.create_task(module.start())

    try:
        await module.on_turn(record(reply="进展"))

        # 等后台持锁进入思考（0.6s）。

        assert await wait_until(
            lambda: module._status == "busy"
        )

        # agent 的 settle 撞上锁，预算内拿不到 → busy 渲染，
        # 树仍是上次定案，不致盲。

        rendered = await module.query(turn())

        assert rendered.startswith("[Plan|busy")

        assert "根 g-000001" in rendered

        # 后台消化完成后，下一次 query 正常定案。

        assert await wait_until(
            lambda: module._status == "settled"
        )

        rendered = await module.query(turn())

        assert rendered.startswith("[Plan|settled")

    finally:
        await module.stop()

        try:
            await start_task

        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_stop_cancels_digest_task_cleanly(tmp_path):
    module = make_planner_module(
        tmp_path, [], plan_text=SUBSTACK_TREE
    )

    start_task = asyncio.create_task(module.start())

    await asyncio.sleep(0.05)

    assert module._digest_task is not None

    assert not module._digest_task.done()

    await module.stop()

    try:
        await start_task

    except asyncio.CancelledError:
        pass

    assert module._digest_task is None

    assert module._stop_event.is_set()


# ============================================================================
# rework: the complaint→push-reality-toward-shadow action
# ============================================================================


@pytest.mark.asyncio
async def test_rework_appends_under_focus_parent(tmp_path):
    module = make_planner_module(
        tmp_path,
        [
            {
                "deliberation": "用户说太丑——影子没变，追加返工条目。",
                "actions": [
                    {
                        "op": "rework",
                        "title": "重做UI直到酷炫",
                        "shadow": "酷炫且清晰的界面，对标顶级审美",
                        "reason": "用户反馈界面丑，现实离影子远",
                    }
                ],
            }
        ],
        plan_text=UI_TREE,
    )

    rendered = await module.query(
        turn(user_input="这界面也太丑了吧")
    )

    entries, _ = parse_plan(
        module.plan_path.read_text(encoding="utf-8")
    )

    # 挂在焦点（g-000003）的父层 = 根下。

    assert entries["g-000004"].parent == "g-000001"

    assert entries["g-000004"].title == "重做UI直到酷炫"

    # 父条目带返工注记，agent 在 feed 里看得到为什么。

    assert any(
        "返工条目" in note
        for note in entries["g-000001"].notes
    )

    assert "待办: g-000004 重做UI直到酷炫" in rendered

    # 焦点路径未动。

    assert module.data.snapshot()["focus_id"] == "g-000003"

    # 影子一个字没动。

    assert entries["g-000001"].shadow.startswith("画质精美")


DEEP_TREE = """<!-- plan-next-id: 4 -->

## [g-000001] 根目标
parent: null
order: 0
leaf: false
shadow: 根影子
missing: null
closed: null
closed_at: null
evidence:

## [g-000002] 中间层
parent: g-000001
order: 0
leaf: false
shadow: 中间影子
missing: null
closed: null
closed_at: null
evidence:

## [g-000003] 焦点叶子
parent: g-000002
order: 0
leaf: true
shadow: 正在做的活
missing: null
closed: null
closed_at: null
evidence:
"""


@pytest.mark.asyncio
async def test_rework_with_explicit_ancestor_target(tmp_path):
    module = make_planner_module(
        tmp_path,
        [
            {
                "deliberation": "整个上层方向都要返工。",
                "actions": [
                    {
                        "op": "rework",
                        "target": "g-000001",
                        "title": "重审整体方向",
                        "shadow": "对照根影子重定标准",
                        "reason": "用户整体不满意",
                    }
                ],
            }
        ],
        plan_text=DEEP_TREE,
    )

    await module.query(turn(user_input="全部不行，重来"))

    entries, _ = parse_plan(
        module.plan_path.read_text(encoding="utf-8")
    )

    # target 是更高层祖先 → 挂到 target 自己名下（该层末尾），
    # 而不是默认的焦点父层。

    assert entries["g-000004"].parent == "g-000001"


@pytest.mark.asyncio
async def test_rework_target_focus_itself_lands_on_parent_level(tmp_path):
    module = make_planner_module(
        tmp_path,
        [
            {
                "deliberation": "对当前焦点返工",
                "actions": [
                    {
                        "op": "rework",
                        "target": "g-000002",
                        "title": "返工当前工作",
                        "shadow": "对照父影子返工",
                        "reason": "用户不满意当前产出",
                    }
                ],
            }
        ],
        plan_text=SURGERY_TREE,
    )

    await module.query(turn(user_input="重做"))

    entries, _ = parse_plan(
        module.plan_path.read_text(encoding="utf-8")
    )

    # target=焦点本身 → 挂到焦点的父层，作为焦点的兄弟。

    assert entries["g-000005"].parent == "g-000001"


@pytest.mark.asyncio
async def test_rework_rejects_non_ancestor_target(tmp_path):
    module = make_planner_module(
        tmp_path,
        [
            {
                "deliberation": "挂到非祖先条目下",
                "actions": [
                    {
                        "op": "rework",
                        "target": "g-000002",
                        "title": "x",
                        "shadow": "y",
                        "reason": "z",
                    }
                ],
            },
            {"deliberation": "默认挂载", "actions": []},
        ],
        plan_text=UI_TREE,
    )

    await module.query(turn(user_input="重做"))

    # UI_TREE 里焦点是 g-000003，g-000002 是它的已闭合兄弟，
    # 不在焦点祖先链上 → 拒绝。

    feedback = module.llm.requests[-1].messages[-1].content

    assert "必须是焦点或其祖先" in feedback


@pytest.mark.asyncio
async def test_rework_requires_all_fields(tmp_path):
    module = make_planner_module(
        tmp_path,
        [
            {
                "deliberation": "缺 reason",
                "actions": [
                    {
                        "op": "rework",
                        "title": "x",
                        "shadow": "y",
                    }
                ],
            },
            {"deliberation": "放弃", "actions": []},
        ],
        plan_text=UI_TREE,
    )

    await module.query(turn(user_input="重做"))

    feedback = module.llm.requests[-1].messages[-1].content

    assert "title/shadow/reason 必填" in feedback

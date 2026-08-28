# @builtin

"""
Plan: the goal-nesting stack as an autonomous builtin Module.

One ordered tree. Every entry carries a shadow state — the ideal
world state its subtree is meant to reach. Execution walks the
tree depth-first, one focus at a time; parallelism is an
execution-layer tactic inside a single entry, never a plan
structure.

Implemented stages (see docs/plan-design.md):

    storage + rendering + hot reload + degradation   (stage 1)
    planner think-cycle: plant_root / expand / open_substack /
    close / note + settle loop + decisions.jsonl     (stage 2)
    replan (single-level surgery) + amend
    (verbatim user quote guard)                      (stage 3)

The planner is an internal LLM loop (llm handle provisioned by
the Facade). It has no control-flow power: its only output
channels are the rendered query section and the DataSpace.

query() renders the stack for the main agent only (depth == 0).
This module is the sanctioned exception to the "query() must not
do work" rule: the plan is the instruction, so a query that waits
for planning is correct. A corrupt PLAN.md never blinds the
agent — the last good parse is kept in memory and in private
state, and the section degrades to a status marker instead of
disappearing (query exceptions are swallowed into None by the
Facade, so degradation must be a rendered marker, never a raise).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from ..model import (
    Module,
    TurnRecord,
)

from ...utils.llm import (
    LLMRequest,
    Message,
)

PLAN_HEADER = re.compile(r"^## \[(g-\d{6})\]\s*(.+)$")

NEXT_ID_LINE = re.compile(r"^<!--\s*plan-next-id:\s*(\d+)\s*-->$")

VERDICTS = ("achieved", "superseded", "moot", "cancelled")

_TODO_LIMIT = 6


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds")
    )


# ============================================================================
# Entry model
# ============================================================================


@dataclass
class PlanEntry:
    """
    One node of the goal-nesting stack.

    shadow       the ideal world state this subtree serves
    leaf         planner verdict: directly executable, no
                 decomposition needed (expand with empty children)
    missing      for planning substacks: the gap that opened them
    closed       None while open; otherwise a verdict
    evidence     observation references backing the closure
    gap_report   root-only, multi-line: unmet items vs the root
                 shadow; stored as "- gap:" lines like notes
    notes        planner notes, latest last (render: latest only)
    """

    id: str

    parent: str | None

    title: str

    shadow: str

    order: int = 0

    leaf: bool = False

    missing: str | None = None

    closed: str | None = None

    closed_at: str | None = None

    evidence: list[str] = field(default_factory=list)

    gap_report: list[str] = field(default_factory=list)

    notes: list[str] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.closed is None


# ============================================================================
# Storage: PLAN.md  (disk is the state)
# ============================================================================


def _optional(value: str) -> str | None:
    return None if value in ("", "null") else value


def parse_plan(text: str) -> tuple[dict[str, PlanEntry], int]:
    """
    Parse PLAN.md into {id: entry} + next_id.

    Raises ValueError on any malformed content — the caller never
    truncates or overwrites a file it cannot parse.
    """
    next_id = 1

    entries: dict[str, PlanEntry] = {}

    current: PlanEntry | None = None

    for raw in text.splitlines():
        line = raw.strip()

        if not line:
            continue

        if line.startswith("<!--"):
            matched = NEXT_ID_LINE.match(line)

            if matched:
                next_id = max(next_id, int(matched.group(1)))

            continue

        header = PLAN_HEADER.match(line)

        if header:
            entry_id, title = header.group(1), header.group(2).strip()

            if entry_id in entries:
                raise ValueError(
                    f"duplicate plan entry id: {entry_id}"
                )

            current = PlanEntry(
                id=entry_id,
                parent=None,
                title=title,
                shadow="",
            )

            entries[entry_id] = current

            continue

        if current is None:
            raise ValueError(
                f"content outside any entry block: {line!r}"
            )

        if line.startswith("- note:"):
            current.notes.append(
                line[len("- note:"):].strip()
            )

            continue

        if line.startswith("- gap:"):
            current.gap_report.append(
                line[len("- gap:"):].strip()
            )

            continue

        key, sep, value = line.partition(":")

        if not sep:
            raise ValueError(
                f"unrecognized plan line: {line!r}"
            )

        key = key.strip()

        value = value.strip()

        if key == "parent":
            current.parent = _optional(value)

        elif key == "order":
            current.order = int(value)

        elif key == "leaf":
            current.leaf = value == "true"

        elif key == "shadow":
            current.shadow = value

        elif key == "missing":
            current.missing = _optional(value)

        elif key == "closed":
            verdict = _optional(value)

            if verdict is not None and verdict not in VERDICTS:
                raise ValueError(
                    f"unknown close verdict: {verdict!r}"
                )

            current.closed = verdict

        elif key == "closed_at":
            current.closed_at = _optional(value)

        elif key == "evidence":
            current.evidence = value.split()

        else:
            raise ValueError(f"unknown plan field: {key!r}")

    # An empty PLAN.md is a valid blank slate. The disk loader may
    # normalize a physically empty file to the canonical empty
    # representation, but parsing must accept both forms so an empty
    # plan can never enter degraded mode.
    if not entries:
        return {}, next_id

    # Referential integrity: parents exist, no cycles.

    for entry in entries.values():
        if entry.parent is not None and (
            entry.parent not in entries
        ):
            raise ValueError(
                f"{entry.id}: unknown parent {entry.parent}"
            )

    for entry in entries.values():
        seen: set[str] = set()

        node: PlanEntry | None = entry

        while node is not None and node.parent is not None:
            if node.id in seen:
                raise ValueError(
                    f"cycle in plan tree at {entry.id}"
                )

            seen.add(node.id)

            node = entries[node.parent]

    # next_id must never collide with existing ids.

    highest = max(
        int(entry.id.split("-")[1]) for entry in entries.values()
    )

    next_id = max(next_id, highest + 1)

    return entries, next_id


def serialize_plan(
    entries: dict[str, PlanEntry],
    next_id: int,
) -> str:
    """
    Render the tree back to PLAN.md, depth-first so the file
    reads top-to-bottom like the plan it is.
    """
    children: dict[str | None, list[PlanEntry]] = {}

    for entry in entries.values():
        children.setdefault(entry.parent, []).append(entry)

    for group in children.values():
        group.sort(key=lambda e: (e.order, e.id))

    parts = [
        f"<!-- plan-next-id: {next_id} -->",
        "",
    ]

    def emit(entry: PlanEntry) -> None:
        parts.append(f"## [{entry.id}] {entry.title}")

        parts.append(f"parent: {entry.parent or 'null'}")

        parts.append(f"order: {entry.order}")

        parts.append(f"leaf: {'true' if entry.leaf else 'false'}")

        parts.append(f"shadow: {entry.shadow}")

        parts.append(f"missing: {entry.missing or 'null'}")

        parts.append(f"closed: {entry.closed or 'null'}")

        parts.append(f"closed_at: {entry.closed_at or 'null'}")

        parts.append(f"evidence: {' '.join(entry.evidence)}")

        for gap in entry.gap_report:
            parts.append("")

            parts.append(f"- gap: {gap}")

        for note in entry.notes:
            parts.append("")

            parts.append(f"- note: {note}")

        parts.append("")

        for child in children.get(entry.id, []):
            emit(child)

    for root in children.get(None, []):
        emit(root)

    return "\n".join(parts).rstrip() + "\n"


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")

    tmp.write_text(text, encoding="utf-8")

    os.replace(tmp, path)


# ============================================================================
# Planner prompt (the behavioural laws; schema constrains artifacts,
# never the thinking)
# ============================================================================

PLANNER_SYSTEM = """你是 NAN 的规划器（planner），负责维护一棵目标嵌套栈：每个条目有标题和影子状态（该子树要推向的理想世界状态），执行沿第一个开放子条目深度优先推进，任一时刻只有一个焦点。

行为律：
1. 每层计划服务于其父条目的影子；根影子由用户执笔，至高无上。
2. 发现理想状态按当前手段不可达，第一反应是重规划手段（换路径/升能力/开子栈/replan），永远不是降低影子。改影子只有一个通道：amend，且必须逐字引用本轮用户原话。
3. 只规划当前焦点一层（JIT）：其他条目保持标题+影子，等它成为焦点再分解。条目是里程碑不是工单：粒度由"这个子块值不值得有自己的影子"决定，琐碎步骤合并进父条目的注记，不要为并行或琐碎操作造条目。
4. 栈里已经有根时，绝不再输出 plant_root——plant_root 只用于从零创建全新任务；已有栈的推进一律对 [焦点] 使用 expand / open_substack / close。
5. 规划中缺信息/缺能力才开子栈（挂在焦点名下，写明 missing）；能用假设推进的先推进，不开栈。
6. 闭合必须引用执行回流里的观测编号（r-xxxxxx）；根条目闭合必须附 gap_report，对照根影子逐条列出未满足项——不许自我宣布完成。
7. 子条目全部闭合的条目：若是规划子栈（missing 在场），继续 expand 完成该层规划；若子树工作已完成，对照影子裁决（close，或 note 说明差距后等待）。
8. 用户对结果的抱怨（如"太丑了"）不是降级信号：影子没变，是现实离影子太远——输出 rework 动作，它自动挂在焦点的父层，当前工作完成后自然接上。remove 只用于推翻尚未开始的待办方向；正在执行中的条目及其子树永远不可移除。
9. deliberation 里给出多个草案、逐一推演其对影子的逼近、写明筛除理由。

输出 JSON。一次输出一组有序动作，级联推进直到焦点可直接执行或必须等待；无事可做输出空 actions 数组：

{"deliberation": "草案/推演/筛除",
 "actions": [
   {"op": "plant_root", "title": "...", "shadow": "..."},
   {"op": "expand", "target": "g-000001", "children": [{"title": "...", "shadow": "..."}]},
   {"op": "open_substack", "target": "g-000001", "title": "...", "shadow": "...", "missing": "..."},
   {"op": "close", "target": "g-000004", "verdict": "achieved", "evidence": ["r-000012"], "gap_report": ["..."]},
   {"op": "note", "target": "g-000002", "text": "..."},
   {"op": "replan", "target": "g-000001", "remove": ["g-000005"],
    "append": [{"title": "...", "shadow": "..."}],
    "reason": "对照该层影子说明为什么要重规划"},
   {"op": "rework", "title": "按用户反馈重做并提升该项质量",
    "shadow": "对照父影子写清什么才算好",
    "reason": "用户反馈现实离影子远"},
   {"op": "amend", "target": "g-000001", "shadow": "修订后的影子",
    "user_quote": "逐字来自本轮用户输入的原话"}
 ]}

expand 的 children 为空数组 = 判定该条目可直接执行（叶子）。close 的 verdict ∈ achieved|superseded|moot|cancelled，只有根需要 gap_report。
replan 是单层手术：只动 target 下未闭合的待办兄弟（remove 连同子树移除，append 追加到该层末尾）；焦点路径（当前焦点所在的链）不可触；target 必须是焦点的祖先。更深层的重构不用你操心：条目成为焦点时再规划。
rework 是返工专用动作：自动追加到焦点的父层（当前工作完成后接上），不需要你计算挂载位置；target 可选，默认焦点父层，也可指定焦点或其祖先的 id。用户抱怨质量时一律用它。"""


# ============================================================================
# Module
# ============================================================================


class PlanModule(Module):
    id = "plan"

    # ------------------------------------------------------------------
    # Configuration (instance attributes; tests may override)
    # ------------------------------------------------------------------

    def __init__(self):
        base = os.getenv("NAN_PLAN_DIR", "data/databases/plan")

        self.base = Path(base)

        self.plan_path = self.base / "PLAN.md"

        self.decisions_path = self.base / "decisions.jsonl"

        self._file_sig: tuple[int, int] | None = None

        self._good_entries: dict[str, PlanEntry] = {}

        self._good_next_id = 1

        self._loaded = False

        self._degraded = False

        # manual → settled <ts>；degraded = planner 或文件不可用。
        self._status = "manual"

        self._status_at: str | None = None

        # ---- planner (stage 2) ----

        self.planner_think_cap = 4

        self.planner_llm_timeout = 120.0

        self.max_batch_records = 12

        self.reply_cap = 400

        # ---- background digestion (stage 4) ----

        # Debounce window: records arriving in a burst are folded
        # into one think.
        self.digest_grace: float = 3.0

        # How long the agent may block inside settle (lock wait +
        # continuation thinks). On overrun the section renders
        # busy with the last settled plan; pending input waits for
        # the next turn. The first think always runs (mandatory).
        self.settle_timeout: float = 120.0

        self._pending_records: list[
            tuple[int, TurnRecord]
        ] = []

        self._pending_user: list[str] = []

        self._record_seq = 0

        self._decision_seq = 0

        self._recent_deliberations: list[str] = []

        self._plan_lock = asyncio.Lock()

        self._wake_event = asyncio.Event()

        self._digest_task: asyncio.Task | None = None

        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Service lifetime
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._ensure_dirs()

        await asyncio.to_thread(self._refresh_from_disk)

        await asyncio.to_thread(self._load_decision_tail)

        self._publish()

        # Background pre-digestion: consume execution records
        # while the agent runs, so the next settle blocks briefly.
        self._digest_task = asyncio.create_task(
            self._digest_loop(), name="plan-digest"
        )

        # Park: Facade retries an immediately-returning start()
        # forever.
        await self._stop_event.wait()

    async def stop(self) -> None:
        self._stop_event.set()

        task = self._digest_task

        if task is not None:
            task.cancel()

            try:
                await task

            except asyncio.CancelledError:
                pass

            self._digest_task = None

    async def on_turn(self, record: TurnRecord) -> None:
        """
        Execution backflow: enqueue for the planner's next settle.

        Subagent records (depth > 0) are queued too — they carry
        work evidence — but the batch renderer labels them so the
        planner can never mistake a subagent brief for a goal.
        """
        self._record_seq += 1

        self._pending_records.append(
            (self._record_seq, record)
        )

        if len(self._pending_records) > 256:
            dropped = self._pending_records.pop(0)

            logger.warning(
                "plan record queue overflow; dropped r-{:06d}",
                dropped[0],
            )

        self._wake_event.set()

    async def query(self, turn) -> str | None:
        depth = getattr(turn, "depth", 0) or 0

        if depth > 0:
            # Subagents get self-contained briefs; the plan is the
            # main agent's intent structure. Also: a subagent's
            # task must never be consumed as a goal.
            return None

        changed = await asyncio.to_thread(
            self._refresh_from_disk
        )

        if changed:
            self._publish()

        user_input = getattr(turn, "user_input", "") or ""

        if user_input:
            self._pending_user.append(user_input)

        if self._pending_user or self._pending_records:
            # The sanctioned blocking point: settle = plan until
            # the stack names an executable focus or an explicit
            # wait. Planning latency IS planning.
            await self._settle()

        return self._render()

    # ------------------------------------------------------------------
    # Disk (parse on demand; last-good cache for degradation)
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        self.base.mkdir(parents=True, exist_ok=True)

    def _refresh_from_disk(self) -> bool:
        """
        Re-read PLAN.md if it changed. Returns True when the
        module's state view changed (including a transition into
        degradation) so the caller republishes.
        """
        try:
            stat = self.plan_path.stat()

            sig = (stat.st_mtime_ns, stat.st_size)

        except FileNotFoundError:
            sig = None

        except OSError as exc:
            logger.warning("plan stat failed: {}", exc)

            return False

        if sig == self._file_sig:
            return False

        self._file_sig = sig

        if sig is None:
            # Respect the deletion: blank slate.
            if self._good_entries:
                logger.info("PLAN.md removed; plan is now empty")

            self._good_entries = {}

            self._good_next_id = 1

            self._degraded = False

            self._loaded = True

            return True

        try:
            text = self.plan_path.read_text(encoding="utf-8")

        except OSError as exc:
            logger.warning("plan read failed: {}", exc)

            return False

        # A physically empty/whitespace-only PLAN.md is a valid blank
        # slate. Normalize it to the canonical empty representation so
        # the file is immediately self-describing and ready for the
        # first planner cycle. Importantly, we do NOT create PLAN.md
        # when it is absent: explicit deletion remains a blank slate.
        if not text.strip():
            try:
                canonical_empty = serialize_plan({}, 1)
                _atomic_write(self.plan_path, canonical_empty)
                stat = self.plan_path.stat()
                self._file_sig = (stat.st_mtime_ns, stat.st_size)
                text = canonical_empty
            except OSError as exc:
                logger.warning(
                    "empty PLAN.md initialization failed: {}", exc
                )
                return False

        try:
            entries, next_id = parse_plan(text)

        except ValueError as exc:
            # Never truncate, never overwrite, never raise: the
            # Facade swallows query exceptions into None, which
            # would blind the agent. Degrade to the last good
            # projection and wait for a human.
            if not self._degraded:
                logger.error(
                    "PLAN.md unparseable ({}); "
                    "serving last good projection",
                    exc,
                )

            self._degraded = True

            return True

        self._good_entries = entries

        self._good_next_id = next_id

        self._degraded = False

        self._loaded = True

        return True

    def _save(self) -> None:
        """Planner-side write path (stage 2+). Atomic, never over
        an unparseable file: refresh first, bail when degraded."""
        self._refresh_from_disk()

        if self._degraded:
            raise RuntimeError(
                "PLAN.md is unparseable; "
                "write path is suspended pending human repair"
            )

        _atomic_write(
            self.plan_path,
            serialize_plan(
                self._good_entries,
                self._good_next_id,
            ),
        )

        self._refresh_from_disk()

    # ------------------------------------------------------------------
    # Focus derivation
    # ------------------------------------------------------------------

    def _children(
        self,
        entries: dict[str, PlanEntry],
        node_id: str,
    ) -> list[PlanEntry]:
        kids = [
            entry
            for entry in entries.values()
            if entry.parent == node_id
        ]

        return sorted(kids, key=lambda e: (e.order, e.id))

    def _has_open_descendant(
        self,
        entries: dict[str, PlanEntry],
        node_id: str,
    ) -> bool:
        for child in self._children(entries, node_id):
            if child.is_open:
                return True

            if self._has_open_descendant(entries, child.id):
                return True

        return False

    def _focus_of(
        self,
        entries: dict[str, PlanEntry],
        node: PlanEntry,
    ) -> PlanEntry:
        open_kids = [
            kid
            for kid in self._children(entries, node.id)
            if kid.is_open
        ]

        if not open_kids:
            return node

        return self._focus_of(entries, open_kids[0])

    def _dfs_open_order(
        self,
        entries: dict[str, PlanEntry],
    ) -> list[PlanEntry]:
        ordered: list[PlanEntry] = []

        def visit(node_id: str | None) -> None:
            for child in self._children(entries, node_id):
                if child.is_open:
                    ordered.append(child)

                visit(child.id)

        visit_on_roots = [
            entry
            for entry in entries.values()
            if entry.parent is None
        ]

        visit_on_roots.sort(key=lambda e: (e.order, e.id))

        for root in visit_on_roots:
            if root.is_open:
                ordered.append(root)

            visit(root.id)

        return ordered

    # ==================================================================
    # Planner: settle loop, think cycle, action application
    # ==================================================================

    async def _settle(self) -> None:
        """
        Plan until resting: either the focus is an executable
        leaf, the planner said wait, or everything is closed.
        The first think requires pending input; continuation
        thinks run on tree state alone (the cascade), so a fresh
        batch is never a precondition for finishing a plan.

        Serialized against the background digest loop. The settle
        budget bounds how long the agent blocks: the first think
        always runs (planning is mandatory), but on overrun the
        remaining work is left queued and the section renders
        busy with the last settled plan.
        """
        deadline = time.monotonic() + self.settle_timeout

        try:
            await asyncio.wait_for(
                self._plan_lock.acquire(),
                timeout=max(0.05, self.settle_timeout),
            )

        except asyncio.TimeoutError:
            self._status = "busy"

            self._publish()

            return

        try:
            batch = self._take_batch()

            focus = self._current_focus()

            previous_focus = focus.id if focus else None

            for attempt in range(self.planner_think_cap):
                if attempt > 0:
                    batch = self._take_batch()

                    if time.monotonic() > deadline:
                        # Continuation would overrun the agent's
                        # blocking budget; leave the rest queued
                        # for the next turn.
                        self._status = "busy"

                        self._publish()

                        break

                if attempt == 0 and not (
                    batch["user_inputs"] or batch["records"]
                ):
                    break

                outcome = await self._planner_think(batch)

                if outcome is None:
                    self._status = "degraded"

                    self._status_at = _now_iso()

                    self._publish()

                    break

                await asyncio.to_thread(
                    self._append_decision, outcome
                )

                try:
                    await asyncio.to_thread(self._save)

                except RuntimeError:
                    self._status = "degraded"

                    self._status_at = _now_iso()

                    self._publish()

                    break

                self._status = "settled"

                self._status_at = outcome["ts"]

                self._publish()

                focus = self._current_focus()

                if focus is None:
                    break

                if not outcome["actions"]:
                    break

                if focus.leaf:
                    break

                if (
                    previous_focus is not None
                    and focus.id == previous_focus
                ):
                    # 焦点没有前进：本轮反应已完成（rework/note
                    # 等），继续级联只会重复——留给下一个事件。
                    break

                previous_focus = focus.id

        finally:
            self._plan_lock.release()

    async def _digest_loop(self) -> None:
        """
        Background pre-digestion: while the agent executes a long
        turn, records trickle in via on_turn; this loop debounces
        the burst into batches and thinks ahead, so the next
        query's settle finds the planner mostly resting and
        blocks only for the final incremental think.
        """
        while not self._stop_event.is_set():
            wake_task = asyncio.create_task(
                self._wake_event.wait()
            )

            stop_task = asyncio.create_task(
                self._stop_event.wait()
            )

            try:
                done, pending = await asyncio.wait(
                    {wake_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

            finally:
                # Never leave orphaned waiters behind — the digest
                # task itself may be cancelled mid-wait on stop.
                for task in (wake_task, stop_task):
                    task.cancel()

                await asyncio.gather(
                    wake_task,
                    stop_task,
                    return_exceptions=True,
                )

            if self._stop_event.is_set():
                break

            self._wake_event.clear()

            # Debounce: let a burst of records accumulate.
            await asyncio.sleep(self.digest_grace)

            if self._stop_event.is_set():
                break

            async with self._plan_lock:
                batch = self._take_batch()

                if not batch["records"] and not batch[
                    "user_inputs"
                ]:
                    continue

                self._status = "busy"

                self._publish()

                outcome = await self._planner_think(batch)

                if outcome is not None:
                    await asyncio.to_thread(
                        self._append_decision, outcome
                    )

                    try:
                        await asyncio.to_thread(self._save)

                    except RuntimeError:
                        self._status = "degraded"

                        self._status_at = _now_iso()

                        self._publish()

                        continue

                    self._status = "settled"

                    self._status_at = outcome["ts"]

                else:
                    self._status = "degraded"

                    self._status_at = _now_iso()

                self._publish()

    # ---- batching ----

    def _take_batch(self) -> dict:
        user_inputs = self._pending_user[:]

        self._pending_user.clear()

        records: list[str] = []

        while (
            self._pending_records
            and len(records) < self.max_batch_records
        ):
            seq, record = self._pending_records.pop(0)

            records.append(
                self._render_record(seq, record)
            )

        return {
            "user_inputs": user_inputs,
            "records": records,
        }

    def _render_record(
        self,
        seq: int,
        record: TurnRecord,
    ) -> str:
        lines = [f"--- r-{seq:06d}"]

        if record.depth > 0:
            lines.append(f"(子代理 depth={record.depth})")

        if record.task and record.task != record.user_input:
            lines.append(f"任务: {record.task[:200]}")

        lines.append(f"用户: {record.user_input[:300]}")

        if record.reply:
            lines.append(
                f"NAN: {record.reply[: self.reply_cap]}"
            )

        if record.error:
            lines.append(f"错误: {record.error[:300]}")

        return "\n".join(lines)

    # ---- planner input ----

    def _entry_state(
        self,
        entries: dict[str, PlanEntry],
        entry: PlanEntry,
    ) -> str:
        if not entry.is_open:
            return f"已闭合→{entry.closed}"

        if entry.leaf:
            return "可执行"

        kids = self._children(entries, entry.id)

        if not kids:
            return "待规划"

        if any(kid.is_open for kid in kids):
            return "推进中"

        return "待裁决"

    def _render_planner_view(
        self,
        entries: dict[str, PlanEntry],
    ) -> str:
        lines: list[str] = []

        def emit(entry: PlanEntry, depth: int) -> None:
            prefix = "  " * depth

            lines.append(
                f"{prefix}{entry.id} {entry.title} "
                f"[{self._entry_state(entries, entry)}] "
                f"影子: {entry.shadow}"
            )

            if entry.missing:
                lines.append(
                    f"{prefix}  missing: {entry.missing}"
                )

            if entry.notes:
                lines.append(
                    f"{prefix}  note: {entry.notes[-1]}"
                )

            for kid in self._children(entries, entry.id):
                emit(kid, depth + 1)

        roots = [
            entry
            for entry in entries.values()
            if entry.parent is None
        ]

        for root in sorted(roots, key=lambda e: (e.order, e.id)):
            emit(root, 0)

        return "\n".join(lines)

    def _render_planner_input(self, batch: dict) -> str:
        entries = self._good_entries

        parts = ["[目标栈]"]

        view = self._render_planner_view(entries)

        parts.append(view if view else "(空)")

        focus = self._current_focus()

        if focus is not None:
            parts.append(
                f"[焦点] {focus.id} {focus.title} "
                f"[{self._entry_state(entries, focus)}]"
            )

        if not batch["user_inputs"] and not batch["records"]:
            parts.append(
                "[本批无新输入] 这是上一次规划的续推："
                "对 [焦点] 继续使用 expand / open_substack，"
                "或输出空 actions 表示等待。"
            )

        if self._recent_deliberations:
            parts.append("[最近推演]")

            for text in self._recent_deliberations[-2:]:
                parts.append(f"- {text[:300]}")

        if batch["user_inputs"]:
            parts.append("[本轮用户输入]")

            for index, text in enumerate(
                batch["user_inputs"], 1
            ):
                parts.append(f"{index}. {text[:600]}")

        if batch["records"]:
            parts.append("[执行回流]")

            parts.extend(batch["records"])

        parts.append("请推进目标栈。只输出 JSON。")

        return "\n".join(parts)

    # ---- LLM ----

    async def _llm_json(
        self,
        system: str,
        user: str,
    ) -> dict | None:
        """
        One constrained-decoding call. Never raises: failures are
        logged and returned as None so settle degrades instead of
        blinding the agent.
        """
        if self.llm is None:
            logger.warning("plan planner has no LLM handle")

            return None

        request = LLMRequest(
            messages=[
                Message(role="system", content=system),
                Message(role="user", content=user),
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        try:
            response = await asyncio.wait_for(
                self.llm.generate_complete(request),
                timeout=self.planner_llm_timeout,
            )

        except asyncio.TimeoutError:
            logger.warning(
                "planner LLM call timed out after {}s",
                self.planner_llm_timeout,
            )

            return None

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            logger.warning(
                "planner LLM call failed: {}", exc
            )

            return None

        try:
            payload = json.loads(response.content)

        except (TypeError, ValueError):
            logger.warning(
                "planner LLM returned non-JSON output"
            )

            return None

        return payload if isinstance(payload, dict) else None

    async def _planner_think(
        self,
        batch: dict,
    ) -> dict | None:
        """
        One cognitive cycle: free deliberation, constrained
        artifacts. Actions are pre-validated against a simulated
        tree so a rejected cycle never half-applies; one retry
        with the error feedback, then give up (degraded).
        """
        user = self._render_planner_input(batch)

        feedback = ""

        error = "no attempt"

        for _ in (1, 2):
            payload = await self._llm_json(
                PLANNER_SYSTEM, user + feedback
            )

            if payload is None:
                return None

            actions = payload.get("actions")

            if not isinstance(actions, list):
                feedback = (
                    "\n\n上一次输出缺少 actions 数组，请修正。"
                )

                continue

            error = self._apply_actions(
                actions, batch["user_inputs"]
            )

            if error is None:
                return {
                    "ts": _now_iso(),
                    "deliberation": str(
                        payload.get("deliberation", "")
                    )[:2000],
                    "actions": actions,
                }

            feedback = (
                "\n\n上一次输出存在以下问题，请修正：\n"
                f"{error}"
            )

        logger.warning(
            "planner output invalid twice: {}", error
        )

        return None

    # ---- action application (simulate, then commit) ----

    @staticmethod
    def _clone_entry(entry: PlanEntry) -> PlanEntry:
        return PlanEntry(
            id=entry.id,
            parent=entry.parent,
            title=entry.title,
            shadow=entry.shadow,
            order=entry.order,
            leaf=entry.leaf,
            missing=entry.missing,
            closed=entry.closed,
            closed_at=entry.closed_at,
            evidence=list(entry.evidence),
            gap_report=list(entry.gap_report),
            notes=list(entry.notes),
        )

    @staticmethod
    def _sim_target(
        sim: dict[str, PlanEntry],
        action: dict,
    ) -> PlanEntry | None:
        target_id = action.get("target")

        if not isinstance(target_id, str):
            return None

        target = sim.get(target_id)

        if target is None or not target.is_open:
            return None

        return target

    def _sim_focus(
        self,
        sim: dict[str, PlanEntry],
    ) -> PlanEntry | None:
        roots = [
            entry
            for entry in sim.values()
            if entry.parent is None
        ]

        for root in sorted(roots, key=lambda e: (e.order, e.id)):
            if root.is_open:
                return self._focus_of(sim, root)

        return None

    def _current_focus(self) -> PlanEntry | None:
        return self._sim_focus(self._good_entries)

    @staticmethod
    def _ancestor_chain(
        sim: dict[str, PlanEntry],
        node: PlanEntry,
    ) -> set[str]:
        """
        The focus path: node itself plus every ancestor up to the
        root. Replan surgery must never touch these.
        """
        chain: set[str] = set()

        current: PlanEntry | None = node

        while current is not None:
            chain.add(current.id)

            current = (
                sim.get(current.parent)
                if current.parent
                else None
            )

        return chain

    def _apply_actions(
        self,
        actions: list,
        batch_user_inputs: list[str],
    ) -> str | None:
        """
        Validate + apply in order against a simulated tree; commit
        only if the whole list is legal. Returns an error string
        for the retry feedback, or None on success.
        """
        sim = {
            entry_id: self._clone_entry(entry)
            for entry_id, entry in self._good_entries.items()
        }

        state = {"next_id": self._good_next_id}

        def fail(index: int, op, message: str) -> str:
            return f"动作 #{index}（{op}）：{message}"

        def new_entry(
            parent: str | None,
            title: str,
            shadow: str,
            order: int,
            **extra,
        ) -> PlanEntry:
            entry = PlanEntry(
                id=f"g-{state['next_id']:06d}",
                parent=parent,
                title=title,
                shadow=shadow,
                order=order,
                **extra,
            )

            sim[entry.id] = entry

            state["next_id"] += 1

            return entry

        def base_order(target: PlanEntry) -> int:
            siblings = [
                e
                for e in sim.values()
                if e.parent == target.id
            ]

            return max(
                (s.order for s in siblings), default=-1
            ) + 1

        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                return f"动作 #{index}：不是对象"

            op = action.get("op")

            if op == "plant_root":
                if not batch_user_inputs:
                    return fail(
                        index, op, "需要本批含用户输入"
                    )

                # plant_root is exclusively the transition from a blank
                # slate to the first task. Once a root exists, planning
                # must continue with expand/open_substack/close/etc.
                if any(
                    entry.parent is None
                    for entry in sim.values()
                ):
                    return fail(
                        index,
                        op,
                        "PLAN 已有根条目；plant_root 只允许从空计划创建首个根",
                    )

                title = str(action.get("title", "")).strip()

                shadow = str(action.get("shadow", "")).strip()

                if not title or not shadow:
                    return fail(index, op, "title/shadow 必填")

                root_order = sum(
                    1
                    for e in sim.values()
                    if e.parent is None
                )

                new_entry(
                    None, title, shadow, root_order
                )

            elif op == "expand":
                target = self._sim_target(sim, action)

                if target is None:
                    return fail(index, op, "target 不存在或未开放")

                if target != self._sim_focus(sim):
                    return fail(
                        index,
                        op,
                        "target 不是当前焦点（JIT：只规划焦点一层）",
                    )

                if target.leaf:
                    return fail(
                        index, op, "可执行条目不重新规划"
                    )

                children = action.get("children")

                if not isinstance(children, list):
                    return fail(index, op, "children 必须是数组")

                if not children:
                    target.leaf = True

                    continue

                start = base_order(target)

                for offset, child in enumerate(children):
                    if not isinstance(child, dict):
                        return fail(
                            index,
                            op,
                            f"children[{offset}] 不是对象",
                        )

                    ctitle = str(
                        child.get("title", "")
                    ).strip()

                    cshadow = str(
                        child.get("shadow", "")
                    ).strip()

                    if not ctitle or not cshadow:
                        return fail(
                            index,
                            op,
                            f"children[{offset}] title/shadow 必填",
                        )

                    new_entry(
                        target.id,
                        ctitle,
                        cshadow,
                        start + offset,
                    )

            elif op == "open_substack":
                target = self._sim_target(sim, action)

                if target is None:
                    return fail(index, op, "target 不存在或未开放")

                if target != self._sim_focus(sim):
                    return fail(index, op, "target 不是当前焦点")

                title = str(action.get("title", "")).strip()

                shadow = str(action.get("shadow", "")).strip()

                missing = str(
                    action.get("missing", "")
                ).strip()

                if not title or not shadow or not missing:
                    return fail(
                        index, op, "title/shadow/missing 必填"
                    )

                new_entry(
                    target.id,
                    title,
                    shadow,
                    base_order(target),
                    leaf=True,
                    missing=missing,
                )

            elif op == "close":
                target = self._sim_target(sim, action)

                if target is None:
                    return fail(index, op, "target 不存在或未开放")

                if target != self._sim_focus(sim):
                    return fail(index, op, "只能闭合当前焦点")

                verdict = action.get("verdict")

                if verdict not in VERDICTS:
                    return fail(
                        index, op, f"verdict 非法: {verdict!r}"
                    )

                evidence = action.get("evidence")

                if (
                    not isinstance(evidence, list)
                    or not evidence
                    or not all(
                        isinstance(x, str) and x
                        for x in evidence
                    )
                ):
                    return fail(
                        index,
                        op,
                        "evidence 必须是非空字符串数组（引用 r-xxxxxx）",
                    )

                gap = action.get("gap_report")

                if target.parent is None and (
                    not isinstance(gap, list)
                    or not gap
                    or not all(
                        isinstance(x, str) and x.strip()
                        for x in gap
                    )
                ):
                    return fail(
                        index,
                        op,
                        "根条目闭合必须附非空 gap_report"
                        "（对照根影子列未满足项）",
                    )

                target.closed = verdict

                target.closed_at = _now_iso()

                target.evidence = [str(x) for x in evidence]

                if target.parent is None:
                    target.gap_report = [
                        str(x).strip() for x in gap
                    ]

            elif op == "note":
                target = self._sim_target(sim, action)

                if target is None:
                    return fail(index, op, "target 不存在或未开放")

                text = str(action.get("text", "")).strip()

                if not text:
                    return fail(index, op, "text 必填")

                target.notes.append(text)

            elif op == "replan":
                target = self._sim_target(sim, action)

                before = self._sim_focus(sim)

                recipe = ""

                if before is not None and before.parent:
                    recipe = (
                        f"提示：当前焦点是 {before.id} "
                        f"{before.title}，其父条目是 "
                        f"{before.parent}；用户抱怨质量请输出 "
                        "rework 动作（自动挂到焦点父层），"
                        "remove 只能用于未开始的待办兄弟。"
                    )

                def rfail(message: str) -> str:
                    if recipe:
                        return fail(
                            index, op, f"{message}。{recipe}"
                        )

                    return fail(index, op, message)

                if target is None:
                    return rfail("target 不存在或未开放")

                if before is None or target.id == before.id:
                    return rfail(
                        "target 不能是焦点本身"
                        "（给焦点追加子条目属于 expand 的职责）"
                    )

                if target.id not in self._ancestor_chain(
                    sim, before
                ):
                    return rfail(
                        "target 必须是当前焦点的祖先"
                        "（单层手术只在该层待办兄弟上进行）"
                    )

                # The focus-path child: the ancestor of the focus
                # that is a direct child of target. Untouchable.
                path_child = before

                while (
                    path_child.parent is not None
                    and path_child.parent != target.id
                ):
                    path_child = sim[path_child.parent]

                remove = action.get("remove", [])

                append = action.get("append", [])

                reason = str(
                    action.get("reason", "")
                ).strip()

                if not reason:
                    return rfail(
                        "reason 必填（对照该层影子论证重规划）"
                    )

                if not isinstance(remove, list) or not isinstance(
                    append, list
                ):
                    return rfail("remove/append 必须是数组")

                if not remove and not append:
                    return rfail("remove 与 append 至少其一")

                for rid in remove:
                    victim = sim.get(rid)

                    if victim is None or (
                        victim.parent != target.id
                    ):
                        return rfail(
                            f"remove 的 {rid} "
                            "不是 target 的直接子条目"
                        )

                    if not victim.is_open:
                        return rfail(
                            f"remove 的 {rid} 已闭合"
                            "（手术只动待办兄弟）"
                        )

                    if rid == path_child.id:
                        return rfail(
                            f"{rid} 在焦点路径上，不可触"
                        )

                # Subtrees go with their roots.
                doomed: set[str] = set()

                stack = list(remove)

                while stack:
                    current = stack.pop()

                    if current in doomed:
                        continue

                    doomed.add(current)

                    stack.extend(
                        kid.id
                        for kid in sim.values()
                        if kid.parent == current
                    )

                if before.id in doomed:
                    return fail(
                        index,
                        op,
                        "焦点在移除子树内（防御性检查，不应触发）",
                    )

                for entry_id in doomed:
                    del sim[entry_id]

                start = max(
                    (
                        e.order
                        for e in sim.values()
                        if e.parent == target.id
                    ),
                    default=-1,
                ) + 1

                for offset, child in enumerate(append):
                    if not isinstance(child, dict):
                        return fail(
                            index,
                            op,
                            f"append[{offset}] 不是对象",
                        )

                    ctitle = str(
                        child.get("title", "")
                    ).strip()

                    cshadow = str(
                        child.get("shadow", "")
                    ).strip()

                    if not ctitle or not cshadow:
                        return fail(
                            index,
                            op,
                            f"append[{offset}] title/shadow 必填",
                        )

                    new_entry(
                        target.id,
                        ctitle,
                        cshadow,
                        start + offset,
                    )

                # Defensive: the focus path must be untouched.
                after = self._sim_focus(sim)

                if after is None or after.id != before.id:
                    return fail(
                        index,
                        op,
                        "手术后焦点路径改变（不应触发）",
                    )

            elif op == "amend":
                target = self._sim_target(sim, action)

                if target is None:
                    return fail(index, op, "target 不存在或未开放")

                shadow = str(action.get("shadow", "")).strip()

                quote = str(
                    action.get("user_quote", "")
                ).strip()

                if not shadow:
                    return fail(index, op, "shadow 必填")

                if not quote:
                    return fail(
                        index,
                        op,
                        "user_quote 必填"
                        "（影子的唯一写通道是用户）",
                    )

                if not any(
                    quote in text
                    for text in batch_user_inputs
                ):
                    return fail(
                        index,
                        op,
                        "user_quote 必须逐字来自本轮用户输入",
                    )

                old_shadow = target.shadow

                target.shadow = shadow

                target.notes.append(
                    f"影子修订，原影子: {old_shadow}"
                )

            elif op == "rework":
                # 返工专用：挂载几何由系统解析（焦点父层或指定
                # 祖先层），模型只负责内容——语义与机械分离。
                before = self._sim_focus(sim)

                if before is None:
                    return fail(
                        index,
                        op,
                        "当前无焦点（栈为空或已全部闭合），无处返工",
                    )

                title = str(action.get("title", "")).strip()

                shadow = str(action.get("shadow", "")).strip()

                reason = str(
                    action.get("reason", "")
                ).strip()

                if not title or not shadow or not reason:
                    return fail(
                        index,
                        op,
                        "title/shadow/reason 必填"
                        "（reason 对照目标影子说明返工依据）",
                    )

                target_id = action.get("target")

                if target_id is None:
                    parent_id = before.parent

                else:
                    if not isinstance(target_id, str):
                        return fail(
                            index, op, "target 必须是条目 id"
                        )

                    if target_id not in self._ancestor_chain(
                        sim, before
                    ):
                        return fail(
                            index,
                            op,
                            f"target {target_id} "
                            "必须是焦点或其祖先",
                        )

                    # target == focus → 挂到焦点的父层（作为
                    # 焦点的兄弟，当前工作完成后接上）。

                    parent_id = (
                        before.parent
                        if target_id == before.id
                        else target_id
                    )

                parent = sim.get(parent_id) if parent_id else None

                if parent is None or not parent.is_open:
                    return fail(
                        index, op, "返工挂载点不存在或未开放"
                    )

                new_entry(
                    parent_id,
                    title,
                    shadow,
                    base_order(parent),
                )

                parent.notes.append(
                    f"返工条目「{title}」：{reason}"
                )

            else:
                return fail(
                    index, op, f"未知动作 op: {op!r}"
                )

        # Commit.
        self._good_entries = sim

        self._good_next_id = state["next_id"]

        return None

    # ---- decisions log (audit + resume memory) ----

    def _append_decision(self, outcome: dict) -> None:
        self._decision_seq += 1

        self._ensure_dirs()

        line = json.dumps(
            {
                "seq": self._decision_seq,
                "ts": outcome["ts"],
                "deliberation": outcome["deliberation"],
                "actions": outcome["actions"],
            },
            ensure_ascii=False,
        )

        with self.decisions_path.open(
            "a", encoding="utf-8"
        ) as fh:
            fh.write(line + "\n")

        self._recent_deliberations.append(
            outcome["deliberation"]
        )

        del self._recent_deliberations[:-5]

    def _load_decision_tail(self) -> None:
        try:
            text = self.decisions_path.read_text(
                encoding="utf-8"
            )

        except (FileNotFoundError, OSError):
            return

        deliberations: list[str] = []

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            try:
                payload = json.loads(line)

            except ValueError:
                continue

            if isinstance(payload, dict):
                self._decision_seq = max(
                    self._decision_seq,
                    int(payload.get("seq", 0)),
                )

                deliberation = payload.get("deliberation")

                if isinstance(deliberation, str):
                    deliberations.append(deliberation)

        self._recent_deliberations = deliberations[-5:]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _emit_note(
        self,
        lines: list[str],
        entry: PlanEntry,
        prefix: str,
    ) -> None:
        if entry.notes:
            lines.append(f"{prefix}  注: {entry.notes[-1]}")

    def _walk(
        self,
        entries: dict[str, PlanEntry],
        node: PlanEntry,
        lines: list[str],
        depth: int,
        path_ids: set[str],
    ) -> None:
        """
        Emit ✓ / · / ▶ lines down the active path; the deepest
        active entry gets the ▶ marker with its shadow.
        """
        prefix = "  " * depth

        kids = self._children(entries, node.id)

        for kid in kids:
            if not kid.is_open:
                lines.append(
                    f"{prefix}✓ {kid.id} {kid.title} → {kid.closed}"
                )

        open_kids = [kid for kid in kids if kid.is_open]

        if not open_kids:
            return

        nxt = open_kids[0]

        path_ids.add(nxt.id)

        if self._has_open_descendant(entries, nxt.id):
            lines.append(f"{prefix}· {nxt.id} {nxt.title}")

            self._emit_note(lines, nxt, prefix)

            self._walk(
                entries, nxt, lines, depth + 1, path_ids
            )

            return

        # Completed work inside the focus entry is evidence the
        # agent should see (e.g. a substack's conclusion).
        for kid in self._children(entries, nxt.id):
            if not kid.is_open:
                lines.append(
                    f"{prefix}  ✓ {kid.id} {kid.title} → {kid.closed}"
                )

        # ▶ = executable (the agent may act); ◌ = planner has not
        # finished with this entry (pending planning / awaiting
        # judgment) — the agent must not mistake it for work order.
        marker = "▶" if nxt.leaf else "◌"

        suffix = (
            ""
            if nxt.leaf
            else f"（{self._entry_state(entries, nxt)}）"
        )

        lines.append(
            f"{prefix}{marker} {nxt.id} {nxt.title} 影子: {nxt.shadow}{suffix}"
        )

        self._emit_note(lines, nxt, prefix)

        if nxt.missing:
            lines.append(f"{prefix}  缺: {nxt.missing}")

    def _render(self) -> str | None:
        entries = self._good_entries

        roots = [
            entry
            for entry in entries.values()
            if entry.parent is None
        ]

        if not roots:
            return None

        if self._degraded:
            status = "degraded"

        elif all(not entry.is_open for entry in entries.values()):
            status = "closed"

        else:
            status = self._status

        if status in ("settled", "busy") and self._status_at:
            lines = [
                f"[Plan|{status} {self._status_at[11:16]}]"
            ]

        else:
            lines = [f"[Plan|{status}]"]

        path_ids: set[str] = set()

        for root in sorted(roots, key=lambda e: (e.order, e.id)):
            path_ids.add(root.id)

            if not root.is_open:
                lines.append(
                    f"根 {root.id}「{root.title}」已闭合 → {root.closed}"
                )

                for gap_line in root.gap_report:
                    if gap_line.strip():
                        lines.append(f"  差距: {gap_line.strip()}")

                continue

            lines.append(
                f"根 {root.id}「{root.title}」影子: {root.shadow}"
            )

            self._emit_note(lines, root, "")

            self._walk(
                entries, root, lines, 1, path_ids
            )

        todo = [
            entry
            for entry in self._dfs_open_order(entries)
            if entry.id not in path_ids
        ]

        if todo:
            items = [
                f"{entry.id} {entry.title}" for entry in todo
            ]

            if len(items) > _TODO_LIMIT:
                shown = " · ".join(items[:_TODO_LIMIT])

                lines.append(
                    f"  待办: {shown} … 等{len(items)}项"
                )

            else:
                lines.append(f"  待办: {' · '.join(items)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # DataSpace / persistence
    # ------------------------------------------------------------------

    def _publish(self) -> None:
        entries = self._good_entries

        roots = [
            entry
            for entry in entries.values()
            if entry.parent is None
        ]

        focus: PlanEntry | None = None

        for root in sorted(roots, key=lambda e: (e.order, e.id)):
            candidate = self._focus_of(entries, root)

            if candidate.is_open:
                focus = candidate

                break

        if self._degraded:
            status = "degraded"

        elif entries and all(
            not entry.is_open for entry in entries.values()
        ):
            status = "closed"

        else:
            status = self._status

        self.data.publish(
            {
                "status": status,
                "status_at": self._status_at,
                "root_count": len(roots),
                "entry_count": len(entries),
                "open_count": sum(
                    1 for e in entries.values() if e.is_open
                ),
                "focus_id": focus.id if focus else None,
                "focus_title": focus.title if focus else None,
                "next_id": self._good_next_id,
            }
        )

    def serialize_state(self) -> dict:
        return {
            "entries": [
                asdict(entry)
                for entry in self._good_entries.values()
            ],
            "next_id": self._good_next_id,
            "status": self._status,
            "status_at": self._status_at,
            "record_seq": self._record_seq,
            "decision_seq": self._decision_seq,
        }

    def restore_state(self, state) -> None:
        if not isinstance(state, dict):
            raise TypeError("plan private state must be an object")

        raw_entries = state.get("entries", [])

        if not isinstance(raw_entries, list):
            raise TypeError(
                "plan private state 'entries' must be a list"
            )

        entries: dict[str, PlanEntry] = {}

        for raw in raw_entries:
            if not isinstance(raw, dict):
                raise TypeError(
                    "plan private state entry must be an object"
                )

            entry_id = raw.get("id")

            title = raw.get("title")

            if not isinstance(entry_id, str) or not isinstance(
                title, str
            ):
                raise TypeError(
                    "plan entry requires string id and title"
                )

            entries[entry_id] = PlanEntry(
                id=entry_id,
                parent=raw.get("parent"),
                title=title,
                shadow=raw.get("shadow", ""),
                order=int(raw.get("order", 0)),
                leaf=bool(raw.get("leaf", False)),
                missing=raw.get("missing"),
                closed=raw.get("closed"),
                closed_at=raw.get("closed_at"),
                evidence=list(raw.get("evidence", [])),
                gap_report=list(raw.get("gap_report", [])),
                notes=list(raw.get("notes", [])),
            )

        next_id = state.get("next_id", 1)

        if not isinstance(next_id, int):
            raise TypeError(
                "plan private state 'next_id' must be an integer"
            )

        status = state.get("status", "manual")

        self._good_entries = entries

        self._good_next_id = next_id

        self._status = (
            status if isinstance(status, str) else "manual"
        )

        status_at = state.get("status_at")

        self._status_at = (
            status_at
            if isinstance(status_at, str)
            else None
        )

        for key, attr in (
            ("record_seq", "_record_seq"),
            ("decision_seq", "_decision_seq"),
        ):
            value = state.get(key, 0)

            if isinstance(value, int) and value >= 0:
                setattr(self, attr, value)

        self._loaded = True

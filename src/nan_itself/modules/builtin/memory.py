# @builtin

"""
Memory: semantic memory as an autonomous builtin Module.

Three layers (see docs/memory-design.md):

    journal.jsonl   loss-free capture of every TurnRecord
    WORKSTATE.md    hot bounded set of open threads
    MEMORY.md       long-term atomic facts (authoritative,
                    human-editable, hot-reloaded)

Five LLM operations, all constrained-decoding JSON mode, all in
the background consumer loop:

    REVIEW   turn batch + open items -> workstate ops + fact
             candidates
    MERGE    candidates vs neighborhood -> ADD / UPDATE /
             DELETE / NOOP
    PROMOTE  closed work item -> episode summary candidate
    REFLECT  accumulated importance -> insight entry
    DISTILL  high-importance entries -> core block text

query() is a cheap projection: no LLM calls, disk is the state.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections import Counter
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(
        timespec="seconds"
    )


def _parse_json(text: str | None) -> dict | None:
    """Tolerant JSON extraction: fences, prose, raw object."""
    if not text:
        return None

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None

    return None


def _tokens(text: str) -> list[str]:
    """
    CJK character bigrams + ASCII words. Grounded enough for
    BM25 over short Chinese entries.
    """
    tokens: list[str] = []

    current_ascii: list[str] = []
    cjk_run: list[str] = []

    def flush_ascii():
        if current_ascii:
            tokens.append("".join(current_ascii).lower())
            current_ascii.clear()

    def flush_cjk():
        if len(cjk_run) == 1:
            tokens.append(cjk_run[0])
        elif len(cjk_run) > 1:
            for i in range(len(cjk_run) - 1):
                tokens.append(cjk_run[i] + cjk_run[i + 1])
        cjk_run.clear()

    for ch in text:
        if ch.isascii() and (ch.isalnum() or ch == "_"):
            flush_cjk()
            current_ascii.append(ch)
        elif ord(ch) > 0x2E00:
            flush_ascii()
            cjk_run.append(ch)
        else:
            flush_ascii()
            flush_cjk()

    flush_ascii()
    flush_cjk()

    return tokens


class _Entry:
    __slots__ = (
        "id",
        "kind",
        "importance",
        "created",
        "valid_from",
        "valid_to",
        "source",
        "links",
        "content",
    )

    def __init__(self, **fields):
        for key in self.__slots__:
            setattr(self, key, fields.get(key))

    def valid(self) -> bool:
        return self.valid_to in (None, "", "null")

    def brief(self, limit: int = 120) -> str:
        kind = self.kind or "fact"
        content = (self.content or "")[:limit]
        return f"- ({kind}) {content}"


class _WorkItem:
    __slots__ = (
        "id",
        "kind",
        "content",
        "opened",
        "last_active",
        "opened_seq",
        "notes",
    )

    def __init__(self, **fields):
        for key in self.__slots__:
            setattr(self, key, fields.get(key))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    try:
        tmp.write_text(text, encoding="utf-8")

        os.replace(tmp, path)

    finally:
        if tmp.exists():
            tmp.unlink()


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")

    except FileNotFoundError:
        return None


class MemoryModule(Module):
    id = "memory"

    # ------------------------------------------------------------------
    # Configuration (instance attributes; tests may override)
    # ------------------------------------------------------------------

    poll_interval: float = 15.0

    batch_limit: int = 8

    llm_timeout: float = 90.0

    workstate_max_open: int = 10

    full_inject_limit: int = 60

    reflect_threshold: int = 40

    distill_tolerance: int = 1

    score_top_k: int = 15

    tau_days: float = 30.0

    weight_recency: float = 0.3

    weight_importance: float = 0.4

    weight_similarity: float = 0.3

    render_cap: int = 800

    preview_cap: int = 80

    entry_cap: int = 120

    def __init__(self):
        base = os.getenv("NAN_MEMORY_DIR", "data/databases/memory")

        self.base = Path(base)

        self.journal_path = self.base / "journal.jsonl"

        self.cursor_path = self.base / "journal.cursor"

        self.workstate_path = self.base / "WORKSTATE.md"

        self.memory_path = self.base / "MEMORY.md"

        self.core_path = self.base / "CORE.md"

        self._cursor_seq = 0

        self._workstate_next_id = 1

        self._reflect_pending = 0

        self._reflect_candidate_ids: list[str] = []

        self._core_contributors: list[str] = []

    # ==================================================================
    # Storage primitives (disk is the state; parse on demand)
    # ==================================================================

    def _ensure_dirs(self) -> None:
        self.base.mkdir(parents=True, exist_ok=True)

    # ---- journal ----

    def _append_journal(self, record: TurnRecord) -> int:
        self._ensure_dirs()

        seq = self._next_seq()

        line = json.dumps(
            {
                "seq": seq,
                "ts": _now_iso(),
                "agent_hash": record.agent_hash,
                "depth": record.depth,
                "task": record.task,
                "user_input": record.user_input,
                "reply": record.reply,
                "error": record.error,
            },
            ensure_ascii=False,
        )

        with self.journal_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

        return seq

    def _next_seq(self) -> int:
        last = self._last_journal_seq()

        return last + 1

    def _last_journal_seq(self) -> int:
        try:
            with self.journal_path.open("rb") as fh:
                fh.seek(0, os.SEEK_END)

                size = fh.tell()

                fh.seek(max(0, size - 4096))

                tail = fh.read().decode(
                    errors="replace"
                ).strip().splitlines()

            for line in reversed(tail):
                try:
                    return json.loads(line)["seq"]

                except Exception:
                    continue

        except FileNotFoundError:
            pass

        return 0

    def _load_cursor(self) -> None:
        text = _read_text(self.cursor_path)

        if text:
            try:
                self._cursor_seq = int(text.strip())

            except ValueError:
                self._cursor_seq = 0

    def _save_cursor(self, seq: int) -> None:
        self._cursor_seq = seq

        _atomic_write(self.cursor_path, str(seq))

    def _read_pending_batch(self) -> tuple[list[dict], int]:
        """
        Returns (main-agent records up to batch_limit, max seq seen).
        Sub-agent records are skipped but still advance the batch.
        """
        text = _read_text(self.journal_path)

        if not text:
            return [], self._cursor_seq

        batch: list[dict] = []

        max_seq = self._cursor_seq

        for line in text.splitlines():

            try:
                row = json.loads(line)

            except json.JSONDecodeError:
                continue

            seq = row.get("seq", 0)

            if seq <= self._cursor_seq:
                continue

            max_seq = max(max_seq, seq)

            if row.get("depth") == 0:
                batch.append(row)

                if len(batch) >= self.batch_limit:
                    break

        return batch, max_seq

    # ---- WORKSTATE.md ----

    def _load_workitems(self) -> tuple[list[_WorkItem], int]:
        """
        Returns (open items, next workstate id).
        """
        text = _read_text(self.workstate_path)

        next_id = 1

        if not text:
            return [], next_id

        items: list[_WorkItem] = []

        current: dict | None = None

        for line in text.splitlines():

            if line.startswith("<!-- workstate-next-id:"):

                raw = line.split(":", 1)[1]

                raw = raw.replace("-->", "").strip()

                try:
                    next_id = int(raw)

                except ValueError:
                    pass

                continue

            if line.startswith("## ["):

                if current is not None:
                    items.append(_WorkItem(**current))

                inner = line[len("## ["):]

                item_id, _, title = inner.partition("]")

                current = {
                    "id": item_id.strip(),
                    "content": title.strip(),
                    "notes": [],
                }

                continue

            if current is None:
                continue

            stripped = line.strip()

            if (
                stripped
                and not stripped.startswith("- ")
                and ": " in stripped
            ):

                key, _, value = stripped.partition(": ")

                if key in (
                    "kind",
                    "opened",
                    "last_active",
                    "opened_seq",
                ):

                    current[key] = value

                    continue

            if stripped:
                current["notes"].append(line)

        if current is not None:
            items.append(_WorkItem(**current))

        return items, next_id

    def _save_workitems(
        self,
        items: list[_WorkItem],
        next_id: int,
    ) -> None:

        parts = [
            f"<!-- workstate-next-id: {next_id} -->",
            "",
            "# Open Work State",
            "",
        ]

        for item in items:
            parts.append(f"## [{item.id}] {item.content}")

            parts.append(f"kind: {item.kind}")

            parts.append(f"opened: {item.opened}")

            parts.append(f"last_active: {item.last_active}")

            parts.append(f"opened_seq: {item.opened_seq}")

            for note in item.notes:
                parts.append(note)

            parts.append("")

        _atomic_write(
            self.workstate_path, "\n".join(parts).rstrip() + "\n"
        )

    # ---- MEMORY.md ----

    def _load_entries(self) -> tuple[list[_Entry], int, int, list[str]]:
        """
        Returns (entries, next_id, reflect_pending, reflect_candidates).
        """
        text = _read_text(self.memory_path)

        next_id = 1

        pending = 0

        candidates: list[str] = []

        entries: list[_Entry] = []

        if not text:
            return entries, next_id, pending, candidates

        for line in text.splitlines():

            if ":" not in line:
                continue

            key, _, raw_value = line.partition(":")

            raw_value = raw_value.replace("-->", "").strip()

            if key.strip() == "<!-- memory-next-id":
                try:
                    next_id = int(raw_value)

                except ValueError:
                    pass

            elif key.strip() == "<!-- reflect-pending":
                try:
                    pending = int(raw_value)

                except ValueError:
                    pass

            elif key.strip() == "<!-- reflect-candidates":
                candidates = [
                    part
                    for part in raw_value.split(",")
                    if part
                ]

        # Split into per-entry sections first.
        sections: list[list[str]] = []

        current_lines: list[str] | None = None

        for raw_line in text.splitlines():

            if raw_line.startswith("## ["):

                if current_lines is not None:
                    sections.append(current_lines)

                current_lines = [raw_line]

            elif current_lines is not None:
                current_lines.append(raw_line)

        if current_lines is not None:
            sections.append(current_lines)

        for lines in sections:

            first = lines[0]

            inner = first[len("## ["):]

            entry_id, _, _title = inner.partition("]")

            entry_id = entry_id.strip()

            rest = lines[1:]

            fields: dict = {"id": entry_id}

            body_lines: list[str] = []

            in_body = False

            for line in rest:

                if not in_body and ": " in line:

                    key, _, value = line.partition(": ")

                    if key in (
                        "kind",
                        "importance",
                        "created",
                        "valid_from",
                        "valid_to",
                        "source",
                        "links",
                    ):

                        if key == "links":
                            value = value.strip().strip("[]").strip()

                            fields[key] = [
                                p
                                for p in value.split(",")
                                if p
                            ] if value else []

                        elif key == "importance":
                            try:
                                fields[key] = int(value)

                            except ValueError:
                                fields[key] = 5

                        else:
                            if value in ("null", ""):
                                value = None

                            fields[key] = value

                        continue

                in_body = True

                body_lines.append(line)

            fields["content"] = "\n".join(body_lines).strip()

            fields.setdefault("kind", "fact")

            fields.setdefault("importance", 5)

            entries.append(_Entry(**fields))

        return entries, next_id, pending, candidates

    def _save_entries(
        self,
        entries: list[_Entry],
        next_id: int,
        reflect_pending: int,
        reflect_candidates: list[str],
    ) -> None:
        parts = [
            f"<!-- memory-next-id: {next_id} -->",
            f"<!-- reflect-pending: {reflect_pending} -->",
            "<!-- reflect-candidates: "
            + ",".join(reflect_candidates)
            + " -->",
            "",
        ]

        for entry in entries:
            parts.append(
                f"## [{entry.id}] "
                f"{(entry.content or '').splitlines()[0][:60]}"
            )

            parts.append(f"kind: {entry.kind}")

            parts.append(f"importance: {entry.importance}")

            parts.append(f"created: {entry.created}")

            parts.append(f"valid_from: {entry.valid_from}")

            parts.append(f"valid_to: {entry.valid_to or 'null'}")

            parts.append(f"source: {entry.source or 'null'}")

            links = ",".join(entry.links or [])

            parts.append(f"links: [{links}]")

            parts.append("")

            parts.append(entry.content or "")

            parts.append("")

        _atomic_write(
            self.memory_path, "\n".join(parts).rstrip() + "\n"
        )

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def start(self) -> None:
        self._ensure_dirs()

        self._load_cursor()

        logger.info(
            "Memory module ready at {} (cursor={})",
            self.base,
            self._cursor_seq,
        )

        while True:
            try:
                await self._consume_once()

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception("memory consume cycle failed")

            await asyncio.sleep(self.poll_interval)

    async def on_turn(self, record: TurnRecord) -> None:
        seq = self._append_journal(record)

        logger.debug(
            "memory journaled seq={} depth={}",
            seq,
            record.depth,
        )

    async def stop(self) -> None:
        pass

    async def _llm_json(self, system: str, user: str) -> dict | None:
        """
        One constrained-decoding call. Never raises: failures are
        logged and returned as None so the consumer loop survives
        anything a backend can do.
        """
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
                timeout=self.llm_timeout,
            )

        except asyncio.TimeoutError:
            logger.warning(
                "memory LLM call timed out after {}s",
                self.llm_timeout,
            )

            return None

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            logger.warning("memory LLM call failed: {}", exc)

            return None

        return _parse_json(response.content)

    # ==================================================================
    # Consumer loop
    # ==================================================================

    def _sync_reflect_state(self) -> None:
        """
        Pull durable reflection counters from MEMORY.md into the
        instance so thresholds survive restarts.
        """
        _, _, self._reflect_pending, self._reflect_candidate_ids = (
            self._load_entries()
        )

    async def _consume_once(self) -> bool:
        self._sync_reflect_state()

        batch, max_seq = self._read_pending_batch()

        if not batch:
            return False

        work_items, ws_next_id = self._load_workitems()

        outcome = await self._review(batch, work_items)

        if outcome is None:
            return False

        work_ops, candidates = outcome

        closures, ws_next_id = self._apply_workstate_ops(
            work_items, work_ops, ws_next_id
        )

        if candidates:
            await self._merge_facts(candidates)

        for item in closures:
            await self._promote(item, batch)

        self._save_cursor(max_seq)

        await self._maybe_reflect()

        return True

    # ------------------------------------------------------------------
    # REVIEW
    # ------------------------------------------------------------------

    def _render_turn(self, row: dict) -> str:
        user = (row.get("user_input") or "")[: self.render_cap]

        reply = (row.get("reply") or "")[: self.render_cap]

        error = row.get("error")

        reply_part = reply or f"(error: {error})" if (reply or error) else "(empty)"

        stamp = row.get("ts", "")

        return f"--- {stamp}\n用户: {user}\nNAN: {reply_part}"

    def _render_open_items(self, items: list[_WorkItem]) -> str:
        if not items:
            return "（当前没有开放事务）"

        lines = []

        today = time.time()

        for item in items:

            try:
                opened_ts = datetime.fromisoformat(
                    item.opened
                ).timestamp()

                age_days = max(
                    0, int((today - opened_ts) / 86400)
                )

            except Exception:
                age_days = 0

            content = (item.content or "")[: self.preview_cap]

            lines.append(
                f"{item.id} ({item.kind}, {age_days}d) {content}"
            )

        return "\n".join(lines)

    _REVIEW_SCHEMA_NOTE = (
        '输出必须是 JSON 对象：\n'
        '{"workstate_ops":[{"op":"open|progress|close",'
        '"target":"w-编号或null","kind":"task|promise|question",'
        '"content":"条目内容","note":"进度或闭合结果"}],'
        '"facts":[{"content":"自包含原子事实","kind":'
        '"preference|agreement|project|other","importance":1}]}'
    )

    async def _review(self, batch, work_items):
        system = (
            "你是个人助理的记忆管理器。每批输入包含：当前开放事务列表、"
            "以及若干轮对话记录。你的职责：\n"
            "1) 维护开放事务：新任务/承诺/待答问题要开启(open)；"
            "有进展要 progress 并写一句注记；确认完成/得到回答/明确取消要 close。\n"
            f"开放条目上限 {self.workstate_max_open} 条；已满时优先合并同类或关闭陈旧项。\n"
            "2) 提取与事务无关的长期事实（用户偏好、约定、身份等）进 facts，"
            "每条自包含、指代已消解、一句话，重要性 1-10。\n"
            "负向规则（重要）：\n"
            "- 用户的偏好/身份/时区等个人属性不是事务，禁止进 workstate_ops，只能进 facts。\n"
            "- 同一批中指向同一事的多个表述：facts 合并为一条，事务才可能 open 一条。\n"
            "- 没有新信息就输出空数组，不要编造。\n"
            "- 琐碎寒暄、纯工具过程不要产出任何输出。\n"
            "示例：用户说 我的时区是 UTC+8 我喜欢极简 → work items ops=[]，\n"
            "facts=[{content:用户时区是 UTC+8,kind:preference,importance:5},"
            "{content:用户偏好极简风格的回答,kind:preference,importance:5}]\n"
            + self._REVIEW_SCHEMA_NOTE
        )

        user = (
            "[当前开放工作状态]\n"
            + self._render_open_items(work_items)
            + "\n\n[本批对话]\n"
            + "\n".join(
                self._render_turn(row) for row in batch
            )
        )

        data = await self._llm_json(system, user)

        if data is None:
            logger.warning("memory REVIEW returned unparseable output")

            return None

        work_ops = data.get("workstate_ops") or []

        facts = data.get("facts") or []

        return work_ops, facts

    def _apply_workstate_ops(
        self,
        items: list[_WorkItem],
        ops: list[dict],
        next_id: int,
    ) -> tuple[list[_WorkItem], int]:
        """
        Applies ops; returns (items, next workstate id); closed
        items are returned for PROMOTE.
        """
        by_id = {item.id: item for item in items}

        closures: list[_WorkItem] = []

        for op in ops:

            if not isinstance(op, dict):
                continue

            action = op.get("op")

            target = op.get("target")

            kind = op.get("kind") or "task"

            content = (op.get("content") or "").strip()

            note = (op.get("note") or "").strip()

            if action == "open":

                if not content:
                    continue

                new_id = f"w-{next_id:06d}"

                next_id += 1

                items.append(
                    _WorkItem(
                        id=new_id,
                        kind=kind,
                        content=content,
                        opened=_now_iso(),
                        last_active=_now_iso(),
                        opened_seq=str(
                            self._cursor_seq
                        ),
                        notes=[],
                    )
                )

            elif action == "progress":

                item = by_id.get(target or "")

                if item is None:
                    continue

                item.last_active = _now_iso()

                if note:
                    item.notes.append(f"- {_now_iso()[:10]} {note}")

            elif action == "close":

                item = by_id.get(target or "")

                if item is None:
                    continue

                if note:
                    item.notes.append(f"- 关闭: {note}")

                item.last_active = _now_iso()

                closures.append(item)

                items.remove(item)

        # Mechanical cap on count (not a content judgment).
        overflow = len(items) - self.workstate_max_open

        for _ in range(max(0, overflow)):
            oldest = min(
                items,
                key=lambda i: i.last_active or "",
            )

            logger.warning(
                "workstate overflow; force-closing {}",
                oldest.id,
            )

            closures.append(oldest)

            items.remove(oldest)

        self._save_workitems(items, next_id)

        return closures, next_id

    # ------------------------------------------------------------------
    # MERGE
    # ------------------------------------------------------------------

    async def _merge_facts(self, candidates: list[dict]) -> None:
        if not candidates:
            return

        entries, next_id, pending, cand_ids = (
            self._load_entries()
        )

        valid_entries = [e for e in entries if e.valid()]

        neighborhoods = {
            index: self._neighborhood(candidate, valid_entries)
            for index, candidate in enumerate(candidates)
        }

        decisions = await self._merge_call(
            candidates, valid_entries, neighborhoods
        )

        if decisions is None:
            return

        for decision in decisions.get("decisions", []):

            if not isinstance(decision, dict):
                continue

            op = decision.get("op")

            try:
                index = int(decision.get("candidate", -1))

            except (TypeError, ValueError):
                continue

            if not (0 <= index < len(candidates)):
                continue

            candidate = candidates[index]

            content = (
                decision.get("content")
                or candidate.get("content")
                or ""
            ).strip()

            if not content:
                continue

            if op == "ADD":

                entry = _Entry(
                    id=f"m-{next_id:06d}",
                    kind=candidate.get("kind") or "fact",
                    importance=int(
                        candidate.get("importance") or 5
                    ),
                    created=_now_iso(),
                    valid_from=_now_iso(),
                    valid_to=None,
                    source=None,
                    links=[],
                    content=content,
                )

                entries.append(entry)

                next_id += 1

                self._reflect_pending += entry.importance

                self._reflect_candidate_ids.append(entry.id)

            elif op == "UPDATE":

                target_id = decision.get("target")

                for entry in entries:

                    if entry.id != target_id:
                        continue

                    entry.valid_to = _now_iso()

                    new_entry = _Entry(
                        id=f"m-{next_id:06d}",
                        kind=entry.kind,
                        importance=int(
                            candidate.get("importance")
                            or entry.importance
                            or 5
                        ),
                        created=_now_iso(),
                        valid_from=_now_iso(),
                        valid_to=None,
                        source=entry.id,
                        links=[entry.id],
                        content=content,
                    )

                    entries.append(new_entry)

                    next_id += 1

                    self._reflect_pending += new_entry.importance

                    self._reflect_candidate_ids.append(new_entry.id)

                    break

            elif op == "DELETE":

                target_id = decision.get("target")

                for entry in entries:

                    if entry.id == target_id:
                        entry.valid_to = _now_iso()

                        break

            # NOOP: nothing to do.

        self._reflect_pending = sum(
            e.importance
            for e in entries[-len(candidates):]
            if False
        ) or self._reflect_pending

        self._save_entries(
            entries, next_id, self._reflect_pending,
            self._reflect_candidate_ids,
        )

    def _neighborhood(
        self,
        candidate: dict,
        entries: list[_Entry],
        k: int = 3,
    ) -> list[_Entry]:
        query_tokens = _tokens(
            candidate.get("content") or ""
        )

        if not query_tokens or not entries:
            return []

        doc_counts = []

        df: Counter = Counter()

        docs = []

        for entry in entries:
            tokens = _tokens(entry.content or "")

            docs.append(tokens)

            df.update(set(tokens))

        n_docs = len(docs)

        avg_len = (
            sum(len(d) for d in docs) / n_docs if n_docs else 1
        )

        scores = []

        k1, b = 1.5, 0.75

        for tokens in docs:

            tf = Counter(tokens)

            score = 0.0

            for token in query_tokens:

                if token not in tf:
                    continue

                idf = math.log(
                    1 + (n_docs - df[token] + 0.5)
                    / (df[token] + 0.5)
                )

                score += idf * (
                    tf[token] * (k1 + 1)
                    / (
                        tf[token]
                        + k1
                        * (1 - b + b * len(tokens) / avg_len)
                    )
                )

            scores.append(score)

        ranked = sorted(
            range(len(entries)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [
            entries[i]
            for i in ranked
            if scores[i] > 0
        ]

    async def _merge_call(self, candidates, entries, neighborhoods):
        candidate_blocks = []

        for index, candidate in enumerate(candidates):
            neighbors = neighborhoods.get(index) or []

            neighbor_text = "\n".join(
                f"{e.id}: {e.content}" for e in neighbors
            ) or "（无相近条目）"

            candidate_blocks.append(
                f"[候选{index}] "
                f"(importance={candidate.get('importance')}) "
                f"{candidate.get('content')}\n相近既有条目:\n{neighbor_text}"
            )

        system = (
            "你是记忆合并器。对每个候选事实，对照其相近既有条目做出决策：\n"
            "- ADD: 全新信息\n"
            "- UPDATE: 更新了某条既有信息（target=该条目 id），content 给出新表述\n"
            "- DELETE: 该候选表明某既有条目已失效错误（target=该条目 id）\n"
            "- NOOP: 候选与某条既有事实一致或仅措辞不同（相同内容绝不 UPDATE 或 DELETE）\n"
            "同时检查候选之间是否互相重复，重复的只保留一条 ADD，其余 NOOP。\n"
            '输出 JSON：{"decisions":[{"candidate":序号,"op":"...",'
            '"target":"m-编号或null","content":"最终入库文本"}]}'
        )

        user = "\n\n".join(candidate_blocks)

        return await self._llm_json(system, user)

    # ------------------------------------------------------------------
    # PROMOTE
    # ------------------------------------------------------------------

    async def _promote(self, item: _WorkItem, batch) -> None:

        try:
            opened_seq = int(item.opened_seq or 0)

        except (TypeError, ValueError):
            opened_seq = 0

        text = _read_text(self.journal_path) or ""

        slice_rows = []

        for line in text.splitlines():

            try:
                row = json.loads(line)

            except json.JSONDecodeError:
                continue

            if (
                row.get("seq", 0) >= opened_seq
                and row.get("depth") == 0
            ):
                slice_rows.append(row)

        chosen = slice_rows[:2] + slice_rows[-4:]

        transcript = "\n".join(
            self._render_turn(row) for row in chosen
        )

        outcome_notes = "\n".join(item.notes or [])

        system = (
            "把一段已完成事务的对话轨迹压缩为一条情节摘要。"
            "摘要必须自包含：目标、关键过程、结果。两到三句话，"
            "指代消解，第三人称陈述。同时给重要性 1-10。\n"
            '输出 JSON：{"summary":"…","importance":n}'
        )

        user = (
            f"[事务] {item.id} ({item.kind}) {item.content}\n"
            f"[开启时间] {item.opened}\n"
            f"[过程注记]\n{outcome_notes or '（无）'}\n"
            f"[对话节选]\n{transcript}"
        )

        data = await self._llm_json(system, user)

        if data is None or not data.get("summary"):
            return

        entries, next_id, pending, cand_ids = (
            self._load_entries()
        )

        entry = _Entry(
            id=f"m-{next_id:06d}",
            kind="episode",
            importance=int(data.get("importance") or 6),
            created=_now_iso(),
            valid_from=_now_iso(),
            valid_to=None,
            source=item.id,
            links=[item.id],
            content=data["summary"],
        )

        entries.append(entry)

        next_id += 1

        self._reflect_pending += entry.importance

        self._reflect_candidate_ids.append(entry.id)

        self._save_entries(
            entries, next_id, self._reflect_pending,
            self._reflect_candidate_ids,
        )

    # ------------------------------------------------------------------
    # REFLECT / DISTILL
    # ------------------------------------------------------------------

    async def _maybe_reflect(self) -> None:
        if self._reflect_pending < self.reflect_threshold:
            return

        entries, next_id, pending, cand_ids = (
            self._load_entries()
        )

        cluster = [
            entry
            for entry in entries
            if entry.id in cand_ids
        ]

        if len(cluster) < 3:
            self._reflect_pending = 0

            self._reflect_candidate_ids = []

            self._save_entries(
                entries, next_id, 0, []
            )

            return

        cluster_text = "\n".join(
            f"{e.id} (imp={e.importance}): {e.content}"
            for e in cluster
        )

        system = (
            "从下列近期记忆中归纳出更高层的洞见（模式、偏好倾向、"
            "长期趋势）。一到两条，每条自包含，并引用来源编号。\n"
            '输出 JSON：{"insights":[{"content":"…","sources":["m-000001"]}]}'
        )

        data = await self._llm_json(system, cluster_text)

        self._reflect_pending = 0

        self._reflect_candidate_ids = []

        if data is None:
            self._save_entries(entries, next_id, 0, [])

            return

        for insight in data.get("insights", []):

            content = (insight.get("content") or "").strip()

            if not content:
                continue

            entry = _Entry(
                id=f"m-{next_id:06d}",
                kind="insight",
                importance=8,
                created=_now_iso(),
                valid_from=_now_iso(),
                valid_to=None,
                source="reflection",
                links=list(insight.get("sources") or []),
                content=content,
            )

            entries.append(entry)

            next_id += 1

        self._save_entries(
            entries, next_id, 0, []
        )

        self._core_contributors = []  # force distill refresh

    async def _distill_core_block(self, contributors) -> str | None:
        entries_text = "\n".join(
            f"- {e.content}" for e in contributors
        )

        system = (
            "把下列关于用户的稳定事实压缩为不超过 12 行的常驻备忘，"
            "每行一条要点，直接陈述，不要开场白。\n"
            '输出 JSON：{"block":"多行文本"}'
        )

        data = await self._llm_json(system, entries_text)

        return (data or {}).get("block")

    def _refresh_core_block(self, entries: list[_Entry]) -> None:
        contributors = [
            entry
            for entry in entries
            if entry.valid()
            and (entry.importance or 0) >= 8
        ]

        contributors.sort(
            key=lambda e: (-(e.importance or 0), e.created or "")
        )

        contributors = contributors[:12]

        contributor_ids = [e.id for e in contributors]

        if contributor_ids == self._core_contributors:
            return

        self._core_contributors = contributor_ids

        if not contributor_ids:
            _atomic_write(self.core_path, "")

            return

        asyncio.get_running_loop().create_task(
            self._distill_and_store(contributors)
        )

    async def _distill_and_store(self, contributors) -> None:
        block = await self._distill_core_block(contributors)

        if block is None:
            block = "\n".join(
                f"- {e.content}" for e in contributors
            )

        header = (
            "<!-- contributors: "
            + ",".join(e.id for e in contributors)
            + " -->\n"
        )

        _atomic_write(self.core_path, header + block + "\n")

    # ==================================================================
    # Feeding (cheap projection)
    # ==================================================================

    def _score_entry(self, entry: _Entry, input_tokens) -> float:
        age_days = 0.0

        try:
            anchor = datetime.fromisoformat(
                entry.valid_from or entry.created
            )

            age_days = max(
                0.0,
                (
                    datetime.now(anchor.tzinfo or timezone.utc)
                    - anchor
                ).total_seconds()
                / 86400,
            )

        except Exception:
            pass

        recency = math.exp(-age_days / self.tau_days)

        importance = (entry.importance or 0) / 10

        entry_tokens = set(_tokens(entry.content or ""))

        overlap = len(entry_tokens & set(input_tokens))

        similarity = (
            overlap / max(1, len(set(input_tokens)))
        )

        return (
            self.weight_recency * recency
            + self.weight_importance * importance
            + self.weight_similarity * similarity
        )

    def _render_memory_body(self, user_input: str) -> str:
        entries, *_ = self._load_entries()

        valid = [e for e in entries if e.valid()]

        if not valid:
            return ""

        if len(valid) <= self.full_inject_limit:
            ordered = sorted(
                valid,
                key=lambda e: -(e.importance or 0),
            )

            return "\n".join(
                e.brief(self.entry_cap) for e in ordered
            )

        core_text = _read_text(self.core_path) or ""

        core_body = core_text.split("-->", 1)[-1].strip()

        contributor_ids = set(self._core_contributors)

        rest = [
            e
            for e in valid
            if e.id not in contributor_ids
        ]

        input_tokens = _tokens(user_input)

        scored = sorted(
            rest,
            key=lambda e: self._score_entry(e, input_tokens),
            reverse=True,
        )[: self.score_top_k]

        sections = []

        if core_body:
            # Core projection stays inside this module's single
            # territory: an indented sub-block, never a header.
            sections.append("- core:")

            for core_line in core_body.splitlines():
                stripped = core_line.strip()

                if stripped:
                    sections.append(f"  {stripped}")

        sections.extend(
            e.brief(self.entry_cap) for e in scored
        )

        return "\n".join(sections)

    async def query(self, turn) -> str | None:
        work_items, _ws_next = self._load_workitems()

        user_input = getattr(turn, "user_input", "") or ""

        lines = ["[Memory]"]

        body_lines = []

        for item in work_items:
            content = (item.content or "")[: self.preview_cap]

            body_lines.append(
                f"- workstate {item.id} ({item.kind}) {content}"
            )

        memory_body = self._render_memory_body(user_input)

        if memory_body:
            body_lines.append(memory_body)

        if not body_lines:
            return None

        return "\n".join(lines + body_lines)

"""What an epoch opens with, and nothing else.

The first turn of an epoch gets the last few raw turns, a pointer and a
health line. Every later turn of the same epoch gets an empty string:
continuous memory is pulled with `segv search`, not pushed. The plugin is
deliberately stateless apart from the per-agent epoch marker, so it can
never serve a stale summary.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from ..fact.layer import FactLayer
from ..store import Store
from .config import CacheConfig
from .prompts import GUIDE, status_block
from .store import CacheStore

EMPTY_TOOL = "exit code: 0"


def _clock(value: Any) -> str:
    try:
        seconds = float(value or 0.0)

    except (TypeError, ValueError):
        return "-"

    if seconds <= 0:
        return "-"

    return datetime.fromtimestamp(seconds, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _one_line(text: Any) -> str:
    return " ".join(str(text or "").split())


def _turn_line(record: dict[str, Any], budget: int) -> str:
    """One archived turn as a single bounded line: seq, time, intent, result."""
    meta = record.get("meta") or {}

    head = (
        f"[{record.get('seq')}] {_clock(record.get('started_at'))} "
        f"{meta.get('origin') or 'normal'}"
    )

    think = ""
    tool = ""

    for message in record.get("messages") or []:
        if not isinstance(message, dict):
            continue

        role = message.get("role")

        if role == "assistant":
            text = " ".join(
                str(block.get("text") or "")
                for block in message.get("thinking") or []
                if isinstance(block, dict)
            ).strip()

            intent = _one_line(text or message.get("content"))

            if intent:
                think = intent

        elif role == "tool":
            text = _one_line(message.get("content"))

            if text and text != EMPTY_TOOL:
                tool = text

    parts = [part for part in (f"think: {think}" if think else "", f"tool: {tool}" if tool else "") if part]

    line = f"{head} | {' | '.join(parts)}" if parts else head

    if budget > 0 and len(line) > budget:
        line = line[: max(0, budget - 1)] + "…"

    return line


class CacheLayer:
    """The epoch gate plus the opening block."""

    def __init__(
        self,
        l0: Store,
        fact: FactLayer,
        config: CacheConfig,
        *,
        store: CacheStore | None = None,
    ) -> None:
        self.l0 = l0
        self.fact = fact
        self.config = config

        self.enabled = bool(config.enabled)

        self.store = (
            store or CacheStore(Path(l0.path) / config.data_dir)
            if self.enabled
            else None
        )
        self._owns_store = store is None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Nothing warms here: no model, no queue, no rewrite."""

    def close(self) -> None:
        if self.store is not None and self._owns_store:
            self.store.close()

    # ------------------------------------------------------------------
    # Inject
    # ------------------------------------------------------------------

    def inject(self, agent_id: str, epoch: int) -> str:
        """The epoch's opening block, or "" when this epoch is already open."""
        if not self.enabled or self.store is None:
            return ""

        agent_id = agent_id.strip()

        if not agent_id:
            return ""

        with self._lock:
            try:
                last = self.store.last_epoch(agent_id)

            except Exception:
                logger.exception("segv-cache marker read failed for {}", agent_id)

                return ""

            if last is not None and epoch <= last:
                return ""

            try:
                self.store.set_last_epoch(agent_id, epoch)

            except Exception:
                logger.exception("segv-cache marker write failed for {}", agent_id)

                return ""

        try:
            return self._opening(agent_id)

        except Exception:
            logger.exception("segv-cache opening failed for {}", agent_id)

            return ""

    def _opening(self, agent_id: str) -> str:
        meta = self.l0.meta(agent_id) or {}

        latest = int(meta.get("next_seq") or 0) - 1

        blocks: list[str] = [GUIDE]

        page = self.l0.turns(agent_id, limit=self.config.recent_turns)
        recent = page.get("turns") or []

        if recent:
            lines = [
                _turn_line(record, self.config.recent_chars)
                for record in reversed(recent)
            ]

            blocks.append("<segv-recent>\n" + "\n".join(lines) + "\n</segv-recent>")

        if latest >= 0:
            blocks.append(
                f"<segv-pointer>latest archived turn is seq {latest}; "
                'search older history with segv search "text", or read one '
                "turn with segv turn <seq>.</segv-pointer>"
            )

        indexed = self._indexed_seq(agent_id)

        lag = max(0, latest - indexed) if indexed >= 0 else max(0, latest + 1)

        blocks.append(status_block(lag <= 0, lag, latest))

        return "\n\n".join(blocks)

    def _indexed_seq(self, agent_id: str) -> int:
        try:
            return int(self.fact.max_seq(agent_id))

        except Exception:
            logger.exception("segv-cache index probe failed for {}", agent_id)

            return -1

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def state(self, agent_id: str) -> dict[str, Any]:
        if not self.enabled or self.store is None:
            return {"enabled": False, "agent_id": agent_id}

        meta = self.l0.meta(agent_id) or {}

        latest = int(meta.get("next_seq") or 0) - 1

        indexed = self._indexed_seq(agent_id)

        lag = max(0, latest - indexed) if indexed >= 0 else max(0, latest + 1)

        page = self.l0.turns(agent_id, limit=self.config.recent_turns)
        recent = page.get("turns") or []

        return {
            "enabled": True,
            "agent_id": agent_id,
            "last_epoch": self.store.last_epoch(agent_id),
            "latest_seq": latest,
            "indexed_seq": indexed,
            "lag_turns": lag,
            "recent": [
                _turn_line(record, self.config.recent_chars)
                for record in reversed(recent)
            ],
            "path": str(self.store.path),
        }

    def stats(self) -> dict[str, Any]:
        if not self.enabled or self.store is None:
            return {"enabled": False}

        return {"enabled": True, **self.store.stats()}

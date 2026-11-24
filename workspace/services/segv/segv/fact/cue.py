"""Cue extraction: one plain sentence per chunk.

The prompt and cleanup are the frozen v2 versions, with the current line
adapted from a message to a chunk. There is no disk cache: fact itself is
the cache -- a turn already committed carries a marker, so a replay never
reaches the model twice.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from .config import CueConfig
from .slm import Slm

_CJK = r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
# CJK models often pad digits with spaces ("2023 年 5 月 8 日"), which breaks
# exact timestamp matching against the raw chunk text.
_SPACE_AROUND_CJK_DIGIT = re.compile(rf"(?<=[0-9])\s+(?={_CJK})|(?<={_CJK})\s+(?=[0-9])")


def normalize_event(text: str) -> str:
    return _SPACE_AROUND_CJK_DIGIT.sub("", text)


def _clean(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`").strip()

        if text.lower().startswith("text"):
            text = text[4:].strip()

    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()

    return normalize_event(text)


class CueExtractor:
    def __init__(self, config: CueConfig, slm: Slm) -> None:
        self.config = config
        self.slm = slm

    def extract(
        self,
        ts: str,
        roles: str,
        text: str,
        prev: tuple[str, str] | None = None,
        nxt: tuple[str, str] | None = None,
    ) -> str:
        prompt = self.config.prompt.format(
            ts=ts,
            spk=roles,
            text=text,
            p_spk=prev[0] if prev else "-",
            prev=prev[1][: self.config.ctx_chars] if prev else "(none)",
            n_spk=nxt[0] if nxt else "-",
            nxt=nxt[1][: self.config.ctx_chars] if nxt else "(none)",
        )

        try:
            event = _clean(self.slm.chat(self.config.system, prompt))

        except Exception:
            event = ""

        if event:
            return event

        if not self.config.fallback_on_error:
            return ""

        # Language-neutral: timestamp and roles first, same as the contract.
        return normalize_event(f"{ts}, {roles}: {text[:160]}")

    def extract_many(
        self, tasks: list[tuple], workers: int | None = None
    ) -> list[str]:
        """Run independent cue tasks concurrently; order follows tasks."""
        if not tasks:
            return []

        workers = max(1, workers or self.config.workers)

        if workers == 1 or len(tasks) == 1:
            return [self.extract(*task) for task in tasks]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda task: self.extract(*task), tasks))

"""Cache layer knobs.

The defaults follow the approved design: a small cue threshold, a fixed
fact-recall budget per step, and a tail bounded at an epoch boundary.
"""

from __future__ import annotations

from pydantic import Field

from ..fact.config import FactModel, SlmConfig


class CacheConfig(FactModel):
    enabled: bool = True
    # Relative to the L0 store directory: <data>/<cache.data_dir>.
    data_dir: str = "cache"

    # Pending cues that trigger a rewrite. 3-5 keeps each rewrite small.
    trigger: int = 5

    # Pending chunks per rewrite. The input is raw text, so this stays small
    # and a long backlog is folded in several bounded batches.
    batch_cues: int = 10

    # Verbatim source text included in one rewrite, in characters. Chunks are
    # taken whole; anything that does not fit waits for the next batch.
    raw_chars: int = 4000

    # What an epoch's first turn sees before the snapshot catches up.
    tail_cues: int = 20
    tail_chars: int = 2000

    # Per-step fact recall. Fixed pool: if the pool holds only seen chunks,
    # recall returns fewer rather than expanding without bound.
    recall_k: int = 3
    recall_pool: int = 200

    # A snapshot list never grows past this; the most recent entries win.
    max_items: int = 50

    # A rewrite that keeps less than this share of the previous items is
    # treated as a bad parse, not as a legitimate state change.
    min_retention: float = 0.5

    slm: SlmConfig = Field(
        default_factory=lambda: SlmConfig(max_tokens=2048, timeout=600.0)
    )

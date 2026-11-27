"""Cache layer knobs: the small block an epoch opens with.

The cache no longer compresses or pushes memory. It remembers which epoch
it has opened for each agent and, on that first turn, emits the last few
raw turns plus a pointer and a health line. Everything older is pulled on
demand with `segv search`.
"""

from __future__ import annotations

from ..fact.config import FactModel


class CacheConfig(FactModel):
    enabled: bool = True
    # Relative to the L0 store directory: <data>/<cache.data_dir>.
    data_dir: str = "cache"

    # Raw turns emitted when an epoch opens, and the cap per turn.
    recent_turns: int = 3
    recent_chars: int = 300

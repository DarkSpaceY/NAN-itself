"""Keys the core owns. Protocol-specific keys in the same YAML are extras."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .cache.config import CacheConfig
from .fact.config import FactConfig


class CoreConfig(BaseModel):
    """SEGV knobs: where the store lives and how much it may keep."""

    model_config = ConfigDict(extra="allow")

    host: str = "127.0.0.1"

    port: int = 9004

    data_dir: str = "data"

    fact: FactConfig = Field(default_factory=FactConfig)

    cache: CacheConfig = Field(default_factory=CacheConfig)

    # L0 retention. 0 disables pruning entirely (unbounded growth).
    max_bytes: int = 20 * 1024**3

    prune_to: float = 0.95

    ledger_limit: int = 300

    activity_buckets: int = 1200

    _home: Path = PrivateAttr(default=Path())

    @property
    def home(self) -> Path:
        """The service's own directory, bound by the process entry."""
        return self._home

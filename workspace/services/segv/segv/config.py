"""Keys this service owns; the shared ones come from the service kit."""

from __future__ import annotations

from pathlib import Path

from nan_service_kit import ServiceConfig, read_config
from pydantic import Field

from .cache.config import CacheConfig
from .fact.config import FactConfig


class Config(ServiceConfig):
    """SEGV knobs: where the store lives and how much it may keep."""

    data_dir: str = "data"

    # Two hub registrations served by one process: the archiver that
    # observes turns, and the injector that speaks.
    archive_id: str = "segv-archive"
    inject_id: str = "segv"

    fact: FactConfig = Field(default_factory=FactConfig)

    cache: CacheConfig = Field(default_factory=CacheConfig)

    # L0 retention. 0 disables pruning entirely (unbounded growth).
    max_bytes: int = 4 * 1024**3

    prune_to: float = 0.95

    ledger_limit: int = 300

    activity_buckets: int = 1200


def load_config(path: Path) -> Config:
    return read_config(Config, path)

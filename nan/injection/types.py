"""Shared injection protocol shapes (no server, no client)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ServiceMode = Literal["inject", "archive"]


@dataclass
class ServiceEntry:
    service_id: str
    endpoint: str
    mode: ServiceMode
    last_seen: float
    # Read budget for one delivery, in seconds. None means the
    # global injection.timeout; only inject services may declare it.
    timeout: float | None = None

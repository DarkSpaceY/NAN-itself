# @module

"""
Inbox: everything addressed to the agent arrives here.

User messages, external events, late subagent reports -- the
InboxModule drains them at query time so each turn's observation
carries the current backlog. Empty inbox means the section simply
disappears from the observation.

Subagents never see the inbox: their instruction is the task they
were spawned with, which the engine places into the observation
directly.
"""

from __future__ import annotations

import asyncio
from collections import deque

from loguru import logger


class InboxModule(Module):
    id = "inbox"

    max_size: int = 256

    def __init__(self) -> None:
        self._items: deque[str] = deque()

    # ------------------------------------------------------------------
    # Delivery API (used by the composition root and by the engine's
    # subagent-report parking)
    # ------------------------------------------------------------------

    def put(
        self,
        item: str,
    ) -> None:
        """
        Add one message; when full, the oldest item is dropped.
        """
        if not item:
            return

        while len(self._items) >= self.max_size:
            dropped = self._items.popleft()

            logger.warning(
                "Inbox overflow, dropped oldest item: {!r}",
                dropped[:80],
            )

        self._items.append(item)

    async def start(self) -> None:
        # Long-lived module: park forever. Returning would mark
        # the module DOWN and drop puts until the facade
        # restarts it.
        await asyncio.Event().wait()

    async def on_turn(
        self,
        record,
    ) -> None:
        pass

    # ------------------------------------------------------------------

    async def query(
        self,
        turn: Turn,
    ) -> str | None:
        # Subagents receive their task from the engine, never
        # the main inbox.
        if getattr(
            turn,
            "depth",
            0,
        ) > 0:
            return None

        if not self._items:
            return None

        items = list(
            self._items
        )

        self._items.clear()

        return "[Inbox]\n" + "\n\n".join(
            items
        )

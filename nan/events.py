"""
Event sink: lossy, fire-and-forget runtime activity feed.

The UI gateway subscribes; agent code only *emits* and never
blocks. Every emitted event carries a monotonic seq and a wall
timestamp. A bounded history ring supports reconnect snapshots.

The sink is the process-wide singleton (loguru-style):
producers call `sink.emit(...)`, consumers call
`sink.subscribe()` -- nobody ever receives the sink by reference,
the same way nobody passes loguru's `logger` around. The sink
carries no domain concepts at all: the three identity keys
(`agent_hash` / `parent_hash` / `depth`), when present, are
placed into `content` by the CALLER; an event that carries none
simply has no agent context (module lifecycle, gateway echo).

Tests isolate themselves via `sink.reset()`.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from typing import Any

HISTORY_LIMIT = 500

SUB_QUEUE_SIZE = 2000


class Sink:
    """
    Process-wide event bus: emission AND subscription in one
    module singleton (loguru-style).

    A pure transport layer: it owns the envelope (`t` / `id` /
    `boot_id` / `seq` / `ts`), the id generator, the fan-out and
    the loop-thread constraint. Identity is carried by the
    emitter inside `content`; without it the event simply has no
    agent context (module lifecycle, gateway echo).

    `reset()` clears subscribers and history -- the test seam.
    """

    def __init__(
        self, history_limit: int = HISTORY_LIMIT, subscriber_queue_size: int = SUB_QUEUE_SIZE
    ) -> None:
        self._subs: set[asyncio.Queue] = set()
        self._history: deque[dict] = deque(maxlen=history_limit)
        self._subscriber_queue_size = subscriber_queue_size
        self._seq = 0
        self._boot_id: str | None = None

        # ids must be unique across the WHOLE process lifetime, not
        # just one turn: the UI keys rows and matches updates by id,
        # and this sink is a process-wide singleton.
        self._ns = uuid.uuid4().hex[:6]
        self._counter = 0

    def reset(self, *, boot_id: str | None = None) -> None:
        """
        Detach everything (the test seam); optionally re-arm with
        a boot id.
        """
        self._subs.clear()
        self._history.clear()
        self._seq = 0
        self._boot_id = boot_id

    # -- ids ------------------------------------------------------------

    def _id(self) -> str:
        self._counter += 1
        return f"e{self._ns}-{self._counter}"

    # -- emission ---------------------------------------------------------

    def emit(
        self,
        t: str,
        *,
        content: dict[str, Any] | None = None,
        id: str | None = None,
    ) -> str:
        """
        Stamp and publish one event; return its id.

        `t` is the flat event-type discriminant; all business
        fields travel inside `content`, which is always an object
        (empty when omitted). `content` is passed through
        verbatim: the sink neither interprets nor adds any field
        (identity keys included).

        Must be called from the event loop thread: the fan-out
        below neither awaits nor locks, which is what makes
        checking and mutating each queue atomic, and the sequence
        counter is not synchronized. A publisher on any other
        thread has to hop through loop.call_soon_threadsafe().
        """
        self._seq += 1

        eid = id or self._id()

        payload: dict[str, Any] = dict(content) if content else {}

        event: dict[str, Any] = {
            "t": t,
            "id": eid,
            "seq": self._seq,
            "ts": time.time(),
            "content": payload,
        }

        if self._boot_id is not None:
            event["boot_id"] = self._boot_id

        self._history.append(event)

        for q in self._subs:
            if q.full():
                # Drop oldest: the UI is a live view, not a log.
                q.get_nowait()

            q.put_nowait(event)

        return eid

    # -- subscription -----------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._subscriber_queue_size)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    @property
    def seq(self) -> int:
        return self._seq

    def history(self) -> list[dict]:
        return list(self._history)


sink = Sink()

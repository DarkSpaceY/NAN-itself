"""
Event bus: lossy, fire-and-forget runtime activity feed.

The UI gateway subscribes; agent code only *emits* and never
blocks. Every emitted event carries a monotonic seq and a wall
timestamp. A bounded history ring supports reconnect snapshots.

Emission happens through the process-wide `sink` singleton
(loguru-style): no reference threading. The sink carries no
domain concepts at all: the three identity keys (`agent_hash` /
`parent_hash` / `depth`), when present, are placed into
`content` by the CALLER; an event that carries none simply has
no agent context (module lifecycle, gateway echo).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from typing import Any
from loguru import logger

HISTORY_LIMIT = 500

SUB_QUEUE_SIZE = 2000


class EventBus:
    def __init__(self, history_limit: int = HISTORY_LIMIT, subscriber_queue_size: int = SUB_QUEUE_SIZE) -> None:
        self._subs: set[asyncio.Queue] = set()
        self._history: deque[dict] = deque(maxlen=history_limit)
        self._subscriber_queue_size = subscriber_queue_size
        self._seq = 0

    def emit(self, event: dict[str, Any]) -> None:
        """
        Stamp one event and fan it out to every subscriber.

        Must be called from the event loop thread: the fan-out
        below neither awaits nor locks, which is what makes
        checking and mutating each queue atomic, and the sequence
        counter is not synchronized. A publisher on any other
        thread has to hop through loop.call_soon_threadsafe().
        """
        self._seq += 1

        stamped = {
            "seq": self._seq,
            "ts": time.time(),
            **event,
        }

        self._history.append(stamped)

        # logger.info(f"[Debug] Event Sent: {event}")

        for q in self._subs:
            if q.full():
                # Drop oldest: the UI is a live view, not a log.
                q.get_nowait()

            q.put_nowait(stamped)

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


class Sink:
    """
    Process-wide emission point for UI activity: a loguru-style
    module singleton, never passed by reference.

    The sink is a pure transport layer: it owns the envelope
    (`t` / `id` / `boot_id`), the id generator and the
    loop-thread constraint, and nothing else. Identity is carried
    by the emitter inside `content`; without it the event simply
    has no agent context (module lifecycle, gateway echo).

    `attach()` wires the bus once at composition root; without a
    bus `emit()` is a no-op returning "". `reset()` detaches the
    bus -- the test seam.
    """

    def __init__(self) -> None:
        self._bus: EventBus | None = None
        self._boot_id: str | None = None

        # ids must be unique across the WHOLE process lifetime, not
        # just one turn: the UI keys rows and matches updates by id,
        # and this sink is a process-wide singleton.
        self._ns = uuid.uuid4().hex[:6]
        self._counter = 0

    # -- lifecycle ------------------------------------------------------

    def attach(
        self,
        bus: EventBus,
        *,
        boot_id: str | None = None,
    ) -> None:
        self._bus = bus
        self._boot_id = boot_id

    @property
    def attached(self) -> bool:
        return self._bus is not None

    def reset(self) -> None:
        self._bus = None
        self._boot_id = None

    # -- ids ------------------------------------------------------------

    def _id(self) -> str:
        self._counter += 1
        return f"e{self._ns}-{self._counter}"

    def emit(
        self,
        t: str,
        *,
        content: dict[str, Any] | None = None,
        id: str | None = None,
    ) -> str:
        """
        Stamp and publish one event; return its id ("" without a bus).

        `t` is the flat event-type discriminant; all business
        fields travel inside `content`, which is always an object
        (empty when omitted). `content` is passed through
        verbatim: the sink neither interprets nor adds any field
        (identity keys included).

        Must be called from the event loop thread, exactly like
        `EventBus.emit`: the fan-out neither awaits nor locks and
        the seq counter is not synchronized. Publishing from any
        other thread has to hop through
        loop.call_soon_threadsafe().
        """
        if self._bus is None:
            return ""

        eid = id or self._id()

        payload: dict[str, Any] = (
            dict(content) if content else {}
        )

        event: dict[str, Any] = {
            "t": t,
            "id": eid,
            "content": payload,
        }

        if self._boot_id is not None:
            event["boot_id"] = self._boot_id

        self._bus.emit(event)

        return eid


sink = Sink()

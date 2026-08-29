"""
Event bus: lossy, fire-and-forget runtime activity feed.

The UI gateway subscribes; agent code only *emits* and never
blocks. Every emitted event carries a monotonic seq and a wall
timestamp. A bounded history ring supports reconnect snapshots.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from typing import Any
from loguru import logger

HISTORY_LIMIT = 500

_SUB_QUEUE_SIZE = 2000


class EventBus:
    def __init__(self, history_limit: int = HISTORY_LIMIT, subscriber_queue_size: int = _SUB_QUEUE_SIZE) -> None:
        self._subs: set[asyncio.Queue] = set()
        self._history: deque[dict] = deque(maxlen=history_limit)
        self._subscriber_queue_size = subscriber_queue_size
        self._seq = 0

    def emit(self, event: dict[str, Any]) -> None:
        self._seq += 1

        stamped = {
            "seq": self._seq,
            "ts": time.time(),
            **event,
        }

        self._history.append(stamped)

        logger.info(f"[Debug] Event Sent: {event}")

        for q in self._subs:
            if q.full():
                # Drop oldest: the UI is a live view, not a log.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            try:
                q.put_nowait(stamped)
            except asyncio.QueueFull:
                pass

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


def local_ts() -> str:
    return time.strftime("%H:%M:%S")


def local_date_label(ts: float | None = None) -> str:
    t = time.localtime(ts if ts is not None else time.time())
    return f"{t.tm_year}年{t.tm_mon}月{t.tm_mday}日"


class StreamSink:
    """
    The StepEngine's view of the bus: only main-agent (depth 0)
    turns get a sink, so subagent activity never leaks into the
    main stream.
    """

    def __init__(self, bus: EventBus | None) -> None:
        self._bus = bus
        # ids must be unique across the WHOLE process lifetime, not
        # just this turn: the UI keys rows and matches updates by
        # id, and one sink is created per main-agent turn.
        self._ns = uuid.uuid4().hex[:6]
        self._counter = 0

    def _id(self) -> str:
        self._counter += 1
        return f"e{self._ns}-{self._counter}"

    def _emit(self, t: str, **fields: Any) -> str:
        if self._bus is None:
            return ""
        eid = fields.pop("id", None) or self._id()
        self._bus.emit({"t": t, "id": eid, **fields})
        return eid

    # -- status -------------------------------------------------------

    def status_working(
        self,
        tools: int = 0,
        subagents: int = 0,
    ) -> None:
        self._emit(
            "status",
            state="working",
            tools=tools,
            subagents=subagents,
        )

    def status_idle(self, next_hop: str | None = None) -> None:
        self._emit("status", state="idle", next_hop=next_hop)

    # -- user echo ----------------------------------------------------

    def user_input(self, text: str) -> None:
        self._emit("user_input", id=self._id(), text=text)

    # -- output -------------------------------------------------------

    def output_started(self) -> str:
        return self._emit("output_started")

    def output_delta(self, stream_id: str, text: str) -> None:
        self._emit(
            "output_delta",
            id=stream_id,
            text=text,
        )

    def output_done(
        self,
        stream_id: str,
        duration: str = "",
    ) -> None:
        self._emit(
            "output_done",
            id=stream_id,
            ts=local_ts(),
            duration=duration,
        )

    def output_cancelled(self, stream_id: str) -> None:
        self._emit("output_cancelled", id=stream_id)

    # -- records ------------------------------------------------------

    def record_started(
        self,
        kind: str,
        name: str,
        summary: str = "",
    ) -> str:
        return self._emit(
            "record_started",
            kind=kind,
            name=name,
            summary=summary,
        )

    def record_detail(self, record_id: str, line: str) -> None:
        self._emit("record_detail", id=record_id, line=line)

    def record_done(
        self,
        record_id: str,
        summary: str = "",
        note: str = "",
    ) -> None:
        self._emit(
            "record_done",
            id=record_id,
            summary=summary,
            note=note,
        )

    def record_void(self, record_id: str) -> None:
        # The activity produced nothing user-visible: withdraw the row.
        self._emit("record_void", id=record_id)

    def record_failed(
        self,
        record_id: str,
        summary: str = "",
    ) -> None:
        self._emit("record_failed", id=record_id, summary=summary)

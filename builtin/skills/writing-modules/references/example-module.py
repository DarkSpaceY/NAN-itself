# @module
"""
Notifier: queues short notifications the model hands down, and
delivers them on a slow tick.

Complete annotated example of a channel Module:
- one downlink channel ("send") with a pydantic payload,
- a feed() override that queues payloads for the tick loop,
- a tick loop in start() that consumes them at module rhythm,
- facts-only uplink via data.publish + a cheap ask() projection,
- one-shot feedback lines drained by ask().
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, ClassVar, Mapping

from loguru import logger
from pydantic import BaseModel, Field

from nan_itself.modules import ChannelSpec


class SendPayload(BaseModel):
    """Schema for the 'send' channel; drives the model-facing JSON schema."""

    text: str = Field(..., min_length=1, description="notification text")
    urgent: bool = False


class NotifierModule(Module):

    id = "notifier"

    # Downlink: the model feeds via invoke_channels; the runtime
    # validates the schema and calls feed().
    channels: ClassVar[Mapping[str, ChannelSpec]] = {
        "send": ChannelSpec(
            SendPayload,
            description="queue a short notification",
        ),
    }

    tick_interval: float = 2.0

    def __init__(self) -> None:
        self._queue: deque[dict] = deque()  # fed payloads, FIFO
        self._events: deque[str] = deque(maxlen=64)
        self._sent: list[dict] = []      # delivered facts (uplink state)
        self._last_error: str | None = None

    def feed(self, channel: str, payload: Any) -> str:
        # The runtime checked the channel and its schema; queue
        # the payload for the tick loop.
        if channel != "send":
            return "rejected"

        self._queue.append(payload)

        return "written"

    async def start(self) -> None:
        # All provisioning happens BEFORE the loop; a failure here
        # raises: the Facade marks the module DOWN and retries.
        logger.info("notifier ticking every {}s", self.tick_interval)

        while True:
            if self._queue:
                # payload is the validated, normalized dict produced by
                # the ChannelSpec model.
                payload = self._queue.popleft()

                try:
                    self._deliver(payload)
                    self._events.append("notification delivered")
                except Exception as exc:
                    self._events.append(f"delivery failed: {exc}")

            self.data.publish(
                {
                    "sent_total": len(self._sent),
                    "last_sent": self._sent[-1] if self._sent else None,
                    "last_error": self._last_error,
                }
            )

            await asyncio.sleep(self.tick_interval)

    def _deliver(self, payload: dict) -> None:
        # Placeholder transport. Facts only -- interpretation belongs
        # to the agent.
        self._sent.append(
            {
                "text": payload["text"],
                "urgent": payload["urgent"],
                "at": time.time(),
            }
        )

    async def tell(self, record: Turn) -> None:
        # Runs in its own task after each completed agent execution.
        # Heavy per-turn processing is allowed here; keep ask() cheap.
        return None

    async def ask(self, turn: Turn) -> str | None:
        # PERFORMANCE RULE: cheap projection only -- read state that
        # start()/tell() already computed. This runs every turn on
        # the agent's critical path.
        if not self._sent:
            return None

        lines = ["[Notifier]"]

        lines.append(f"- sent total: {len(self._sent)}")

        last = self._sent[-1]

        lines.append(f"- last: {last['text']!r}")

        if self._last_error:
            lines.append(f"- last error: {self._last_error}")

        # One-shot feedback, drained: each event is rendered
        # exactly once.
        while self._events:
            lines.append(f"[event] {self._events.popleft()}")

        return "\n".join(lines)

    def serialize_state(self) -> dict:
        # JSON-only private state; restored by restore_state() on boot.
        # Downlink residue is intentionally NOT persisted.
        return {"sent": self._sent}

    def restore_state(self, state) -> None:
        if isinstance(state, dict):
            self._sent = state.get("sent", [])

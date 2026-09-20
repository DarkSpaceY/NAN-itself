# @module
"""
Notifier: queues short notifications the model hands down, and
delivers them on a slow tick.

Complete annotated example of an ActionSurface module:
- one downlink channel ("send") with a pydantic payload,
- a tick loop in start() that consumes targets at module rhythm,
- facts-only uplink via data.publish + a cheap query() projection,
- one-shot feedback via emit_event, rendered by render_action_section.
"""

from __future__ import annotations

import asyncio
import time
from typing import ClassVar, Mapping

from loguru import logger
from pydantic import BaseModel, Field

from nan_itself.modules.action import ActionSurface, ChannelSpec


class SendPayload(BaseModel):
    """Schema for the 'send' channel; drives the model-facing JSON schema."""

    text: str = Field(..., min_length=1, description="notification text")
    urgent: bool = False


class NotifierModule(ActionSurface):

    id = "notifier"

    # Downlink: the model writes via invoke_channels. depth=1 keeps a
    # single overwrite slot; use depth=N for a FIFO queue.
    channels: ClassVar[Mapping[str, ChannelSpec]] = {
        "send": ChannelSpec(
            SendPayload,
            description="queue a short notification",
            depth=4,
        ),
    }

    tick_interval: float = 2.0

    def __init__(self) -> None:
        self._sent: list[dict] = []      # delivered facts (uplink state)
        self._last_error: str | None = None

    async def start(self) -> None:
        # All provisioning happens BEFORE the loop; a failure here must
        # degrade the module to `unavailable`, never raise through.
        logger.info("notifier ticking every {}s", self.tick_interval)

        while True:
            payload = self.current_target("send")

            if payload is not None:
                # payload is the validated, normalized dict produced by
                # the ChannelSpec model.
                try:
                    self._deliver(payload)
                    self.clear_target("send")
                    self.emit_event("notification delivered")
                except Exception as exc:
                    self.emit_event(f"delivery failed: {exc}")

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

    async def on_turn(self, record: Turn) -> None:
        # Runs in its own task after each completed agent execution.
        # Heavy per-turn processing is allowed here; keep query() cheap.
        return None

    async def query(self, turn: Turn) -> str | None:
        # PERFORMANCE RULE: cheap projection only -- read state that
        # start()/on_turn() already computed. This runs every turn on
        # the agent's critical path.
        if not self._sent:
            return None

        lines = ["[Notifier]"]

        lines.append(f"- sent total: {len(self._sent)}")

        last = self._sent[-1]

        lines.append(f"- last: {last['text']!r}")

        if self._last_error:
            lines.append(f"- last error: {self._last_error}")

        # ActionSurface modules append the action section so the model
        # sees channel occupancy and one-shot feedback events.
        action = self.render_action_section(turn)

        if action:
            lines.append(action)

        return "\n".join(lines)

    def serialize_state(self) -> dict:
        # JSON-only private state; restored by restore_state() on boot.
        # Channel slot residue is intentionally NOT persisted.
        return {"sent": self._sent}

    def restore_state(self, state) -> None:
        if isinstance(state, dict):
            self._sent = state.get("sent", [])

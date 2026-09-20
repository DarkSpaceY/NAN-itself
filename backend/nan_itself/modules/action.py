"""
ActionSurface: the downlink face of reactive Modules.

A Module that wants the model to be able to *act* on it declares
channels (ClassVar specs) and inherits from ActionSurface. Writing
a target is fire-into-slot: the model never blocks and never knows
that ticks exist. The module consumes slots at its own rhythm from
the tick loop in start().

The uplink stays exactly as before: DataSpace / query() projection
plus events / inbox. This file only adds the downlink.

Re-export note: action modules import both names from here --

    from nan_itself.modules.action import ActionSurface, ChannelSpec
"""

from __future__ import annotations

from collections import deque
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from .model import (
    ChannelSpec,
    Module,
    Turn,
)


class ActionSurface(Module):
    """
    Module base class with downlink channels.

    Subclass contract:

        channels   ClassVar mapping of channel name ->
                   ChannelSpec (declare on the subclass)
        start()    long-running tick loop; consumes targets
        query()    cheap projection; include
                   render_action_section() so the model sees
                   channel occupancy and one-shot feedback

    Slot state is transient: it does not participate in
    serialize_state(), and hot reload drops residue when the
    instance is rebuilt.
    """

    channels: ClassVar[
        Mapping[str, ChannelSpec]
    ] = MappingProxyType({})

    # One-shot feedback events retained for rendering. Both slot
    # and event deques rely on CPython's atomic deque operations;
    # writers (Facade routing) and readers (tick loop, query)
    # all run on the same event loop thread.
    _EVENT_CAPACITY = 64

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._slots: dict[str, deque[Any]] = {}

        self._events: deque[str] = deque(
            maxlen=self._EVENT_CAPACITY
        )

    # ------------------------------------------------------------------
    # Model-facing write path (called by Facade routing only)
    # ------------------------------------------------------------------

    def set_target(
        self,
        channel: str,
        payload: Any,
    ) -> str:
        """
        Fire one validated payload into the slot.

        Returns 'written' | 'replaced' | 'rejected'. The payload
        must already be validated and deep-copied by the caller;
        this method only owns slot discipline and the veto hook.
        """
        slot = self._slot(channel)

        if not self.on_target(channel, payload):
            return "rejected"

        replaced = (
            self.channels[channel].depth == 1
            and len(slot) > 0
        )

        slot.append(payload)

        return "replaced" if replaced else "written"

    # ------------------------------------------------------------------
    # Module-facing consumption path
    # ------------------------------------------------------------------

    def current_target(
        self,
        channel: str,
    ) -> Any | None:
        """
        Peek the next unconsumed target without consuming it.

        None when the slot is empty. For depth=1 this is the
        current target; for depth=N it is the oldest queued one.
        """
        slot = self._slot(channel)

        if not slot:
            return None

        return slot[0]

    def clear_target(self, channel: str) -> None:
        """
        Consume the next target (internal consumption primitive).

        For depth=1 this retracts a not-yet-consumed target; for
        depth=N it pops the oldest queued one.
        """
        slot = self._slot(channel)

        if slot:
            slot.popleft()

    def on_target(
        self,
        channel: str,
        payload: Any,
    ) -> bool:
        """
        Veto hook, called before a payload enters the slot.

        Return False to reject the write. The default accepts
        everything; consumption policy stays module-private.
        """
        return True

    # ------------------------------------------------------------------
    # One-shot feedback + registry summary
    # ------------------------------------------------------------------

    def emit_event(self, text: str) -> None:
        """
        Record a one-shot feedback event for the model.

        Rendered (once) by render_action_section; the tick loop
        uses this to surface what a tick did with a target --
        failure, needs-guidance, completion -- without any new
        uplink semantics.
        """
        self._events.append(text)

    def render_action_section(
        self,
        turn: Turn,
    ) -> str | None:
        """
        Render the action section for this module's query().

        Two parts, either of which may be absent:

            - the channel registry summary (name, depth,
              occupancy) so the model does not re-send targets
              it already sent
            - one-shot feedback events, drained: each event is
              rendered exactly once

        Returns None when there is nothing to render, so the
        subclass query() can simply forward it.
        """
        lines: list[str] = []

        for name, spec in self.channels.items():
            occupancy = len(self._slot(name))

            capacity = (
                str(spec.depth)
                if spec.depth > 1
                else "1"
            )

            state = (
                "empty" if occupancy == 0 else
                f"{occupancy} pending"
            )

            lines.append(
                f"[channel] {self.id}/{name} "
                f"(depth={capacity}, {state})"
            )

        while self._events:
            lines.append(
                f"[event] {self._events.popleft()}"
            )

        if not lines:
            return None

        return "\n".join(lines)

    # ------------------------------------------------------------------

    def _slot(self, channel: str) -> deque[Any]:
        """
        Lazily create the slot for one channel.
        """
        slot = self._slots.get(channel)

        if slot is None:
            depth = self.channels[channel].depth

            slot = deque(maxlen=depth)

            self._slots[channel] = slot

        return slot

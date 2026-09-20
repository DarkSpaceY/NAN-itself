# Channels deep dive

This note expands on ChannelSpec and ActionSurface for module authors.
Canonical source: `backend/nan_itself/modules/model.py` (ChannelSpec) and
`backend/nan_itself/modules/action.py` (ActionSurface).

## ChannelSpec

```python
ChannelSpec(model, *, description="", depth=1)
```

- `model`: a **pydantic BaseModel class** classifying valid payloads.
  The JSON schema handed to the model and the validation applied at write
  time both derive from this one declaration (same annotation-driven
  mechanism `@tool` uses). `None` means the channel accepts any JSON
  value unvalidated.
- `depth`: slot discipline.
  - `1` — overwrite: the newest target replaces the current one
    (default for goals/intents).
  - `N > 1` — FIFO queue with drop-oldest overflow.
- `description`: surfaced to the model by `show_channels`.

## Write path (framework side)

The model calls `invoke_channels` with composite name
`module:<module_id>/<channel>`. The Facade routes the payload through:

1. schema check against the ChannelSpec,
2. optional `on_target(channel, payload) -> bool` veto hook
   (return False to reject),
3. deep copy into the slot.

Return values the model sees: `written`, `replaced` (overwrote an
unconsumed depth-1 target), `rejected` (with reason). Writing never
blocks on the tick loop, and the tick loop never blocks writes.

## Consumption (module side, inside start())

```python
async def start(self) -> None:
    while True:
        payload = self.current_target("goal")   # peek without consuming
        if payload is not None:
            try:
                ...                              # act on payload
                self.clear_target("goal")        # consume on success
                self.emit_event("goal accepted") # one-shot feedback
            except Exception as exc:
                self.emit_event(f"goal failed: {exc}")
        await asyncio.sleep(self.tick_interval)
```

- `current_target(channel)` — peek the next unconsumed target.
- `clear_target(channel)` — consume it (retract for depth=1, pop-oldest
  for depth=N).
- `emit_event(text)` — record a one-shot feedback event; rendered once by
  `render_action_section`, then drained.

## Rendering

Action modules must append `render_action_section(turn)` to their
`query()` output. It renders the channel registry summary (occupancy per
channel) and drains one-shot events, so the model learns what its writes
did without any new verb round-trip.

## Properties to keep in mind

- Slot state is transient: it does not participate in
  `serialize_state()`, and a hot reload drops residue when the instance
  is rebuilt.
- Channels run in parallel; a module maintains one current target per
  channel (depth=1) or a bounded queue (depth=N).
- Plain `Module` subclasses stay channel-free; only `ActionSurface`
  subclasses get downlink providers, attached/detached with the module
  runtime state (RUNNING / terminated).

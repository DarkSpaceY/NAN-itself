# Channels deep dive

This note expands on ChannelSpec and the feed() downlink for module
authors. Canonical source: `backend/nan_itself/modules/model.py`
(ChannelSpec, Module.feed).

## ChannelSpec

```python
ChannelSpec(model, *, description="")
```

- `model`: a **pydantic BaseModel class** classifying valid payloads.
  The JSON schema handed to the model and the validation applied at
  feed time both derive from this one declaration (same
  annotation-driven mechanism `@tool` uses). `None` means the channel
  accepts any JSON value unvalidated.
- `description`: surfaced to the model by `show_channels`.

## Feed path (framework side)

The model calls `invoke_channels` with composite name
`module:<module_id>/<channel>`. The Facade routes the payload through:

1. schema check against the ChannelSpec,
2. deep copy,
3. `feed(channel, payload)` on the module instance.

Return values the model sees: `written` or `rejected` (with reason for
schema failures). Feeding is synchronous for the model and never
touches the module's tick loop.

## Consumption (module side, inside start())

`feed()` is a Module base method; the default just returns
`"written"` (the runtime has already validated channel existence and
schema). Override it only to store or act on the payload:

```python
def feed(self, channel: str, payload) -> str:
    if channel != "goal":
        return "rejected"
    self._queue.append(payload)   # module-private storage
    return "written"

async def start(self) -> None:
    while True:
        if self._queue:
            payload = self._queue.popleft()
            try:
                ...                            # act on payload
            except Exception as exc:
                self._events.append(           # one-shot feedback
                    f"goal failed: {exc}"
                )
        await asyncio.sleep(self.tick_interval)
```

- Storage is module-private: a plain list/deque/attribute of your
  choosing, with whatever overflow policy your channel needs.
- Feedback to the model is plain `ask()` output: drain your private
  event lines once each so the model learns what its feeds did
  without any new verb round-trip.

## Properties to keep in mind

- Downlink residue is transient: it does not participate in
  `serialize_state()`, and a hot reload drops it when the instance
  is rebuilt.
- The model only ever receives `written`/`rejected`; malformed
  payloads are rejected at the runtime boundary and never reach
  `feed()`.
- Plain `Module` subclasses without a `channels` declaration stay
  channel-free; the routing answers "exposes no channels".

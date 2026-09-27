---
name: writing-modules
description: >-
  How to write a NAN-itself Module file (ambient perception daemon with
  channels). Covers file placement, the # @module header, the Module
  contract, lifecycle, the ask() performance rule, graceful
  degradation, persistence, and hot reload. Use when the user asks to create,
  fix, or extend a module in builtin/modules/ or workspace/modules/.
---

# Writing a Module

A Module is **one Python file** in `builtin/modules/<name>.py` (shipped) or
`workspace/modules/<name>.py` (user territory). It runs as a supervised
background daemon; the agent reads its output as ambient context each turn.

## File contract

- First 20 lines of the file must contain the literal header `# @module`
  (convention: line 1).
- Exactly one concrete `Module` subclass per file.
- `Module` and `Turn` are **injected** into the file namespace — do not
  import them. For channels, import explicitly:
  `from nan_itself.modules import ChannelSpec`.
- ClassVar `id` must be unique across builtin + workspace modules;
  a duplicate id fails discovery.

## The shape

```python
# @module
"""One-line purpose statement."""

from __future__ import annotations

import asyncio
from loguru import logger


class MyModule(Module):

    id = "my-module"

    requires: ClassVar[tuple[str, ...]] = ()   # ids of modules this reads

    async def start(self) -> None:             # service lifetime
        while True:
            ...                                 # sample / infer
            self.data.publish({...})            # uplink state
            await asyncio.sleep(self.poll_interval)

    async def tell(self, turn: Turn) -> None:          # completed turn hook
        ...

    async def ask(self, turn: Turn) -> str | None:   # cheap projection
        return "[MyModule] ..." or None
```

## Hard rules

1. **ask() must be cheap.** It runs on the agent's critical path every
   turn: only read state computed earlier, never do heavy work, never call
   the LLM. A slow ask() delays every agent.
2. **One shared event loop -- never block it.** `tell()` runs in its own
   task and may *await* long operations (thread results, async APIs), but
   a synchronous CPU-heavy or blocking call stalls the loop and freezes
   every module and agent -- run such work on threads (`start()`'s
   daemon threads; `asyncio.to_thread` for short known-blocking calls).
3. **DataSpace carries small JSON facts only.** `publish()` and
   `snapshot()` deepcopy the full state on the event loop every round;
   never push frame data or large documents through it.
4. **Publish facts, not conclusions.** Modules report observations; the
   agent interprets them.
5. **Full implement, let it crash.** Provisioning failures (missing
   weights/hardware) raise out of `start()`; the Facade marks the module
   DOWN with the error and retries with backoff; never swallow errors
   into an `unavailable` limbo state. Provision all models in `start()`
   **before** entering the loop -- do the loading on a background
   thread, never synchronously inside the coroutine.
6. **Persist via serialize_state()/restore_state().** Return JSON-only
   state; the Facade stores it under `data/modules/private/<id>.json`.
   Downlink residue is intentionally NOT persisted.
7. Uplink is `self.data.publish(mapping)` (only the owner writes; readers
   get detached snapshots via `self.dependencies[<id>].snapshot()`).

## Channels (downlink, optional)

Declare a ClassVar `channels` mapping on your `Module` subclass:

```python
class SetGoal(Module):

    id = "goal-setter"

    channels: ClassVar[Mapping[str, ChannelSpec]] = {
        "goal": ChannelSpec(GoalPayload, description="current goal"),
    }
```

- The model feeds a payload via `invoke_channels` (`module:<id>/<channel>`
  composite name). The runtime validates the channel and its schema, then
  calls `feed(channel, payload)` on your instance; return `"written"` or
  `"rejected"` (the base-class default accepts and returns `"written"`).
- Consumption policy is module-private: store fed payloads in your own
  queue/state and consume them from your `start()` loop at your own
  rhythm.
- Report tick outcomes as one-shot feedback lines in your `ask()` output
  so the model learns what its feeds did.

## Checklist

- [ ] `# @module` in the first 20 lines; exactly one subclass
- [ ] unique `id`; `requires` lists only existing module ids
- [ ] `start()` is long-running (or intentionally state-only)
- [ ] `ask()` cheap; returns str or None
- [ ] no synchronous CPU-heavy/blocking calls in coroutines (offload to
      threads); publishes stay small JSON facts
- [ ] weights provisioned in `start()` (loading on a background thread);
      failures raise (Facade retries with backoff)
- [ ] no cwd-relative paths — anchor through `nan_itself.utils.paths`

Validate the file before finishing:

```bash
uv run python builtin/skills/writing-modules/scripts/check_module.py <module.py>
```

Details: [references/example-module.py](references/example-module.py)
(complete annotated example), [references/channels.md](references/channels.md)
(ChannelSpec deep dive).

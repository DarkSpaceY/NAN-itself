---
name: writing-modules
description: >-
  How to write a NAN-itself Module file (ambient perception daemon with
  channels). Covers file placement, the # @module header, the Module /
  ActionSurface contract, lifecycle, the query() performance rule, graceful
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
  `from nan_itself.modules.action import ActionSurface, ChannelSpec`.
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

    async def on_turn(self, record: Turn) -> None:   # completed turn hook
        ...

    async def query(self, turn: Turn) -> str | None: # cheap projection
        return "[MyModule] ..." or None
```

## Hard rules

1. **query() must be cheap.** It runs on the agent's critical path every
   turn: only read state computed earlier, never do heavy work, never call
   the LLM. Heavy work belongs in `start()`'s loop and `on_turn()` (each
   runs in its own task).
2. **Publish facts, not conclusions.** Modules report observations; the
   agent interprets them.
3. **Full implement, let it crash.** Provisioning failures (missing
   weights/hardware) raise out of `start()`; the Facade marks the module
   DOWN with the error and retries with backoff; never swallow errors
   into an `unavailable` limbo state. Provision all models in `start()`
   **before** entering the loop.
4. **Persist via serialize_state()/restore_state().** Return JSON-only
   state; the Facade stores it under `data/modules/private/<id>.json`.
   Slot/channel residue is intentionally NOT persisted.
5. Uplink is `self.data.publish(mapping)` (only the owner writes; readers
   get detached snapshots via `self.dependencies[<id>].snapshot()`).

## Channels (downlink, optional)

Subclass `ActionSurface` instead of `Module` and declare:

```python
class SetGoal(ActionSurface):

    id = "goal-setter"

    channels: ClassVar[Mapping[str, ChannelSpec]] = {
        "goal": ChannelSpec(GoalPayload, description="current goal", depth=1),
    }
```

- `depth=1` overwrite slot; `depth=N` FIFO with drop-oldest.
- Model writes via `invoke_channels` (`module:<id>/<channel>` composite
  name); your tick loop consumes with `current_target()` /
  `clear_target()`.
- Optional veto hook `on_target(channel, payload) -> bool`; report tick
  outcomes with `emit_event(text)`.
- Include `render_action_section(turn)` in your `query()` output so the
  model sees channel occupancy and one-shot feedback.

## Checklist

- [ ] `# @module` in the first 20 lines; exactly one subclass
- [ ] unique `id`; `requires` lists only existing module ids
- [ ] `start()` is long-running (or intentionally state-only)
- [ ] `query()` cheap; returns str or None
- [ ] weights provisioned in `start()`; failures raise (Facade retries with backoff)
- [ ] no cwd-relative paths — anchor through `nan_itself.utils.paths`

Validate the file before finishing:

```bash
uv run python builtin/skills/writing-modules/scripts/check_module.py <module.py>
```

Details: [references/example-module.py](references/example-module.py)
(complete annotated example), [references/channels.md](references/channels.md)
(ChannelSpec + ActionSurface deep dive).

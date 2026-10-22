---
name: writing-modules
description: >-
  How to write a NAN-itself Module (ambient perception daemon with
  channels). Covers folder placement, the Module contract, lifecycle, the
  ask() performance rule, graceful degradation, persistence, and hot
  reload. Use when the user asks to create, fix, or extend a module in
  builtin/modules/ or workspace/modules/.
---

# Writing a Module

A Module is **one folder** in `builtin/modules/<name>/` (shipped) or
`workspace/modules/<name>/` (user territory). It runs as a supervised
background daemon; the agent reads its output as ambient context each turn.

```
<name>/
├── main.py        # entry: the Module subclass (required)
├── helper.py      # any sibling .py files (imported with `from . import helper`)
├── config.yaml    # optional; loaded via self.load_config()
└── data/          # private state: weights, caches, persisted json
                   # (never fingerprinted, never imported, never reloaded)
```

## File contract

- Exactly one concrete `Module` subclass in `main.py`.
- The module's identity is the **folder name** — there is no `id` to
  declare; a declared `id` is ignored.
- `Module` and `Turn` are **injected** into every `.py` of the folder —
  do not import them. For channels, import explicitly:
  `from nan_itself.modules import ChannelSpec`.
- Siblings import each other with normal package syntax
  (`from . import helper`).
- Config and private state resolve through the folder itself:
  `self.home` (folder path), `self.data_dir`, `self.config_path`,
  `self.load_config()`, `self.resolve_path("data/weights.bin")`.

## The shape

```python
"""One-line purpose statement."""

from __future__ import annotations

import asyncio
from loguru import logger


class MyModule(Module):
    requires: ClassVar[tuple[str, ...]] = ()  # folder names this reads

    async def start(self) -> None:  # service lifetime
        while True:
            ...  # sample / infer
            self.data.publish({...})  # uplink state
            await asyncio.sleep(self.poll_interval)

    async def tell(self, record: Turn) -> None:  # completed turn hook
        ...

    async def ask(self, turn: Turn) -> str | None:  # cheap projection
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
   thread, never synchronously inside the coroutine. Weights and caches
   live under the module's own `data/`.
6. **Persist via serialize_state()/restore_state().** Return JSON-only
   state; the Facade stores it under `<folder>/data/private.json` and
   the DataSpace snapshot under `<folder>/data/dataspace.json`.
   Downlink residue is intentionally NOT persisted.
7. Uplink is `self.data.publish(mapping)` (only the owner writes; readers
   get detached snapshots via `self.dependencies[<id>].snapshot()`).

## Reload semantics

The folder is fingerprinted as a whole (every `.py` + `config.yaml`,
excluding `data/`, hidden files and `__pycache__`). Editing any watched
file hot-reloads the module; writes under `data/` never trigger a
reload. Deleting `main.py` unloads the module but keeps `data/`, so
re-adding `main.py` restores the persisted state.

## Channels (downlink, optional)

Declare a ClassVar `channels` mapping on your `Module` subclass:

```python
class GoalSetter(Module):
    channels: ClassVar[Mapping[str, ChannelSpec]] = {
        "goal": ChannelSpec(GoalPayload, description="current goal"),
    }
```

- The model feeds a payload via `invoke_channels`, which takes three
  separate arguments: `module` (the folder name, e.g. `"goal-setter"`),
  `channel` (the channel name, e.g. `"goal"`) and `payload` (an object
  shaped by the channel schema). The runtime validates the channel and
  its schema, then calls `feed(channel, payload)` on your instance;
  return `"written"` or `"rejected"` (the base-class default accepts and
  returns `"written"`).
- `list_channels` enumerates each exposed channel as a bare
  `<module>/<channel>` line (for example `goal-setter/goal`), with no
  `module:` prefix.
- Consumption policy is module-private: store fed payloads in your own
  queue/state and consume them from your `start()` loop at your own
  rhythm.
- Report tick outcomes as one-shot feedback lines in your `ask()` output
  so the model learns what its feeds did.

## Checklist

- [ ] exactly one concrete Module subclass in `main.py`
- [ ] `requires` lists only existing module folder names
- [ ] `start()` is long-running (or intentionally state-only)
- [ ] `ask()` cheap; returns str or None
- [ ] no synchronous CPU-heavy/blocking calls in coroutines (offload to
      threads); publishes stay small JSON facts
- [ ] weights provisioned in `start()` (loading on a background thread);
      failures raise (Facade retries with backoff)
- [ ] paths resolve through `self.home` / `self.resolve_path()`, never
      process cwd

Validate the folder before finishing:

```bash
uv run python builtin/skills/writing-modules/scripts/check_module.py builtin/modules/<name>
```

Details: [references/example-module/](references/example-module/)
(complete annotated example), [references/channels.md](references/channels.md)
(ChannelSpec deep dive).

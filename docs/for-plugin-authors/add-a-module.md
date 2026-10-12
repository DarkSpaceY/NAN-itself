# Add a module

This guide adds a Module to NAN-itself. A Module is a supervised
background daemon that senses, computes and acts at its own rhythm while
the agent runs its turn loop; the agent reads its output as ambient
context each turn. Modules are hot-reloaded, so no restart is needed.

To add a module to the shipped set, create the folder under
`builtin/modules/`; to keep it in your own territory, create it under
`workspace/modules/`. For the full contract, see
[modules.md](modules.md).

## Create the module folder

Create `builtin/modules/<name>/` with a `main.py` containing exactly one
concrete `Module` subclass:

```python
"""One-line module description (my_module/main.py)."""

from __future__ import annotations

import asyncio
from loguru import logger


class MyModule(Module):

    requires: ClassVar[tuple[str, ...]] = ()   # folder names this reads

    async def start(self) -> None:             # service lifetime
        while True:
            ...
            self.data.publish({...})           # uplink state
            await asyncio.sleep(self.poll_interval)

    async def tell(self, turn: Turn) -> None:      # completed-turn hook
        ...

    async def ask(self, turn: Turn) -> str | None:  # cheap projection
        return "[MyModule] ..." or None
```

- The folder name is the module identity — there is no `id` to declare.
- Define exactly one concrete `Module` subclass in `main.py`; classes in
  sibling files are not entry candidates.
- `Module` and `Turn` are seeded into every `.py` of the folder — do not
  import them. Siblings import each other with normal package syntax
  (`from . import helper`). Import anything else explicitly, for example
  `from nan_itself.modules import ChannelSpec` for channels.
- `requires` lists the folder names of modules whose `DataSpace` this
  module reads; those are exposed as `self.dependencies`.
- Put models, databases and other heavyweight payloads under `data/` in
  the folder — it is never imported and never fingerprinted, so writes
  there never trigger a reload. Resolve configured paths with
  `self.resolve_path(...)`: relative values resolve against the folder.

## Make it behave

- **Provision in `start()`.** Load model weights, probe devices and
  other provisioning-length work in `start()` on a background thread,
  before entering the loop.
- **Let it crash.** Provisioning failures (missing weights, absent
  hardware) raise out of `start()`; the Facade marks the module DOWN
  with the error and retries with backoff. Do not swallow errors into an
  `unavailable` limbo state.
- **Keep `ask()` cheap.** Return a projection of already-computed state;
  never do heavy work or call the LLM there.
- **Never block the loop.** `tell()` runs in its own task and may
  `await` long operations, but a synchronous CPU-heavy or blocking call
  stalls the shared event loop — offload it to a thread.
- **Publish small JSON facts only.** `self.data.publish(mapping)` is the
  uplink; frame data or large documents do not go through it.

## Read real examples

- `builtin/modules/system/` — a polling daemon that publishes
  deterministic facts and renders a compact ambient block.
- `builtin/modules/inbox/` — a minimal state-only module with a
  delivery API.
- `builtin/modules/vision/` / `audio/` — capture and inference on
  daemon threads; each owns its weights under `data/`.
- `builtin/modules/voice/` — adds the reactive channel downlink.

## Add a reactive module (channels)

If the module needs to consume model-driven payloads at its own tick —
a perceive → decide → act loop faster than LLM round-trips — declare a
`channels` mapping of `ChannelSpec` on the class and consume fed payloads
from your `start()` loop. See
[modules.md](modules.md) for the channel
contract and the downlink verbs.

## Add module configuration

A module that needs configuration owns exactly one file — `config.yaml`
inside its folder — and loads and validates it itself (convenience:
`self.load_config()`). See
[../for-users/configuration.md](../for-users/configuration.md).

## Verify

Check that the module loads without restarting the process: the Facade
scans every second, so a valid new folder appears (marked RUNNING) within
a few seconds, and its `ask()` output shows up as ambient context on the
next turn.

## Related

- [modules.md](modules.md) — the full module subsystem reference.
- [../for-contributors/develop.md](../for-contributors/develop.md) — the
  conventions this module must follow.

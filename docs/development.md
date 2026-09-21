# Development

## Setup

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # installs runtime + dev group (pytest) by default
uv run pytest -q # run the full suite
```

Async tests use explicit `@pytest.mark.asyncio` markers (pytest-asyncio in
strict mode).

## Conventions

- **Path anchoring.** Every repo-relative path (`data/`, `models/`,
  `config/`, `builtin/`) must resolve through
  [`backend/nan_itself/utils/paths.py`](../backend/nan_itself/utils/paths.py)
  (`repo_root()` / `data_dir()` / `models_dir()`). Never use
  cwd-relative defaults or private `parents[N]` lookups — the process
  must behave identically when started from any directory.
- **One file = one provider.** Both backend kinds follow the same rule so
  every source file is a single hot-reloadable entity:
  - `builtin/tools/mcps/*.yaml` — one stdio MCP server per file
    (`name` / `command` / `args` / `env` / `cwd`).
  - `builtin/tools/local/*.py` — one `LocalToolProvider` subclass per
    file, with a `# @tool` header within the first 20 lines.
- **Local tool methods** are decorated with `@tool`, must be fully
  type-annotated (the JSON schema derives from the annotations), and may
  be `async`. Return values are JSON-serialized into the tool result;
  exceptions become error results.
- **Lazy heavy imports.** Import expensive dependencies inside the tool
  method body, not at module top level.
- **Retries** use the shared exponential backoff helper in
  [`backend/nan_itself/utils/backoff.py`](../backend/nan_itself/utils/backoff.py)
  (`next_backoff(values, index)`).
- **Logging** goes through `loguru`.
- **Comments and identifiers are in English.**

## Hot reload

Both builtin and workspace sources (`workspace/tools/mcps/`,
`workspace/tools/local/`) are rescanned every `providers.scan_interval`
seconds. Editing a file replaces its provider live (candidate generation
first, commit, then old worker teardown); deleting a file disables its
provider. Failed sources are retried with backoff and never kill the
runtime.

## Adding a builtin tool

MCP provider — create `builtin/tools/mcps/<name>.yaml`:

```yaml
name: <name>
command: uvx          # or npx
args:
  - "--from"
  - "<pypi-package>"
  - "<entrypoint>"
```

Local Python provider — create `builtin/tools/local/<name>.py`:

```python
# @tool
"""One-line provider description."""

from __future__ import annotations

from typing import Any


class MyProvider(LocalToolProvider):

    id = "<name>"

    @tool
    async def do_thing(self, query: str) -> Any:
        """What the tool does.

        Args:
            query: argument description.
        """
        ...
```

`LocalToolProvider`, `@tool`, `text_result` and `error_result` are
injected into the module namespace by the loader; the file needs no
framework imports to hot-reload.

## Adding a builtin module

Mirror the existing module shape in `builtin/modules/` (see
`vision.py` / `audio.py`; `voice.py` adds the reactive channel
downlink): background daemon threads for capture and
inference, a pure `query()` projection as the agent-facing surface,
and full-implement-let-it-crash semantics -- provisioning failures
raise out of `start()`, so the Facade marks the module DOWN with the
error and retries with backoff until weights or hardware appear.

## Tests

- The suite must stay green before every commit: `uv run pytest -q`.
- Tool-level tests fake their backends (subprocesses, network); keep the
  suite hermetic.
- Path-anchoring rules are enforced by `tests/test_paths_anchoring.py`,
  including foreign-cwd process-level checks.

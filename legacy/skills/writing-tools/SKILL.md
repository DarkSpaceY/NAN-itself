---
name: writing-tools
description: >-
  How to write a tool provider for NAN-itself: a local Python provider
  folder in builtin/tools/local/ or workspace/tools/local/ (one ToolSet
  subclass in main.py, @tool methods with type-annotation-derived
  schema), or an MCP server YAML in tools/mcps/. Use when the user asks
  to create, fix, or extend a tool.
---

# Writing a Tool

Tools come in two kinds, both hot-reloaded from `builtin/tools/`
(shipped) and `workspace/tools/` (user territory): MCP = **one YAML
file**, local Python = **one folder**.

## Kind 1: local Python provider

Folder: `tools/local/<name>/` with a `main.py` defining exactly one
concrete `ToolSet` subclass. The provider's identity is the **folder
name** — there is no `id` to declare; a declared `id` is ignored.

```
<name>/
├── main.py        # the ToolSet subclass (required)
├── helper.py      # optional siblings (`from . import helper`)
├── config.yaml    # optional (e.g. search's searxng settings)
└── data/          # private, never imported / fingerprinted
```

- `ToolSet`, `@tool`, `text_result` and `error_result` are
  **injected** into every `.py` of the folder — the files need no
  framework imports to hot-reload.

```python
"""One-line provider description."""

from __future__ import annotations

from typing import Any


class MyProvider(ToolSet):
    @tool
    async def do_thing(self, query: str, limit: int = 10) -> Any:
        """What the tool does (model-facing summary).

        Args:
            query: argument description.
            limit: argument description.
        """
        ...  # heavy imports go INSIDE the method body, not at top level
```

Method rules:

- **Fully type-annotated parameters** — the JSON schema the model sees
  derives from the annotations (same mechanism as ChannelSpec). Use
  `str | None`, `int`, `float`, `bool`, pydantic models, or `Any`.
- Docstring first line = tool summary; the `Args:` section documents
  each parameter.
- May be `async` or plain; return values are JSON-serialized into the
  tool result; raised exceptions become error results.
- **Lazy heavy imports**: import expensive dependencies inside the
  method body, not at module top level, so provider load and reload
  stay fast and a missing optional dependency only fails the call.
- Every call is timeout-bounded (300 s default,
  `providers.tool_timeout` in settings); timeouts surface as error
  results, never hang the loop.
- Subprocesses / network calls: never depend on the process cwd;
  anchor repo-relative paths to the provider folder
  (`Path(__file__).resolve().parent`) or the repo root.

## Kind 2: MCP server

File: `tools/mcps/<name>.yaml` — one stdio MCP server per file:

```yaml
name: <name>
command: uvx          # or npx, or an absolute path
args:
  - "--from"
  - "<pypi-package>"
  - "<entrypoint>"
# env: {}             # optional extra environment
# cwd: sub/dir        # optional; relative cwd is repo-anchored
```

Choose MCP when an existing server already provides the capability
(see `builtin/tools/mcps/touchpoint.yaml` for a real example with a
dependency downgrade via `uvx --with`); choose local Python for small
in-process capabilities.

## Checklist

- [ ] exactly one ToolSet subclass in `main.py`; folder name = provider
      name
- [ ] every `@tool` method fully annotated with a docstring
- [ ] heavy imports inside method bodies
- [ ] no cwd-relative paths; secrets come from settings/env, never
      hardcoded

Validate before finishing:

```bash
uv run python builtin/skills/writing-tools/scripts/check_tool.py builtin/tools/local/<name>
```

Details: [references/example-tool/](references/example-tool/)
(annotated local provider),
[references/mcp-yaml.md](references/mcp-yaml.md) (full YAML field
reference and recipes).

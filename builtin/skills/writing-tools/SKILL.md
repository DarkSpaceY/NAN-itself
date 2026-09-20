---
name: writing-tools
description: >-
  How to write a tool provider for NAN-itself: a local Python provider file
  in builtin/tools/local/ or workspace/tools/local/ (# @tool header, one
  LocalToolProvider subclass, @tool methods with type-annotation-derived
  schema), or an MCP server YAML in tools/mcps/. Use when the user asks to
  create, fix, or extend a tool.
---

# Writing a Tool

Tools come in two interchangeable kinds, both hot-reloaded from
`builtin/tools/` (shipped) and `workspace/tools/` (user territory):
**one file = one provider** in both cases.

## Kind 1: local Python provider

File: `tools/local/<name>.py`.

- First 20 lines must contain the literal header `# @tool` (convention:
  line 1, before the docstring).
- Exactly one `LocalToolProvider` subclass per file, with a unique
  ClassVar `id`.
- `LocalToolProvider`, `@tool`, `text_result` and `error_result` are
  **injected** into the file namespace — the file needs no framework
  imports to hot-reload.

```python
# @tool
"""One-line provider description."""

from __future__ import annotations

from typing import Any


class MyProvider(LocalToolProvider):

    id = "<name>"

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
- Subprocesses / network calls: never depend on the process cwd; anchor
  repo-relative paths through `nan_itself.utils.paths`.

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

- [ ] `# @tool` in the first 20 lines; exactly one provider subclass
- [ ] unique `id`; every `@tool` method fully annotated with a docstring
- [ ] heavy imports inside method bodies
- [ ] no cwd-relative paths; secrets come from settings/env, never
      hardcoded

Validate before finishing:

```bash
uv run python builtin/skills/writing-tools/scripts/check_tool.py <tool.py>
```

Details: [references/example-tool.py](references/example-tool.py)
(annotated local provider),
[references/mcp-yaml.md](references/mcp-yaml.md) (full YAML field
reference and recipes).

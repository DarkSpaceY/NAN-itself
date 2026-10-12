# Add a tool

This guide adds a tool provider to NAN-itself. There are two kinds of
provider, and you pick one: an **MCP server** (a stdio subprocess
described by a YAML file) or a **local Python provider** (an in-process
`ToolSet` subclass). Both kinds are hot-reloaded, so no restart is
needed.

To add the provider to the shipped set, create the file under
`builtin/tools/`; to keep it in your own territory, create it under
`workspace/tools/` with the same layout. For the full provider model, see
[tools.md](tools.md).

## Add an MCP provider

Create `builtin/tools/mcps/<name>.yaml` describing exactly one stdio MCP
server:

```yaml
name: <name>
command: uvx          # or npx, or an absolute path
args:
  - "--from"
  - "<pypi-package>"
  - "<entrypoint>"
# env: {}             # optional extra environment, merged over yours
# cwd: sub/dir        # optional; a relative cwd is repo-anchored
```

Choose MCP when an existing server already provides the capability. See
`builtin/tools/mcps/touchpoint.yaml` for a real example that pins a
dependency with `uvx --with`.

## Add a local Python provider

Create `builtin/tools/local/<name>/` with a `main.py` containing exactly
one concrete `ToolSet` subclass:

```python
"""One-line provider description (<name>/main.py)."""

from __future__ import annotations

from typing import Any


class MyProvider(ToolSet):

    @tool
    async def do_thing(self, query: str, limit: int = 10) -> Any:
        """What the tool does.

        Args:
            query: argument description.
            limit: argument description.
        """
        ...
```

The loader seeds `ToolSet`, `@tool`, `text_result` and `error_result`
into every `.py` of the folder, so the files need no framework imports
to hot-reload. Siblings import each other with normal package syntax
(`from . import helper`). An explicit import of the same name simply
shadows the seeding.

Method rules:

- The folder name is the provider identity — there is no `id` to
  declare.
- Fully type-annotate every parameter — the JSON schema derives from the
  annotations. `*args` / `**kwargs` are not supported.
- The docstring's first line is the tool summary; the `Args:` section
  documents each parameter.
- Put heavy imports inside the method body, not at module top level.
- Put heavyweight payloads under `data/` in the folder — it is never
  imported and never fingerprinted, so writes there never trigger a
  reload.

## Verify

Check that the provider loads without restarting the process: the runtime
scans every second, so a valid new folder appears through `list_tools` (as
`<name>/<tool>`) within a few seconds. Inspect it with `show_tool` and
call it with `invoke_tool`.

## Related

- [tools.md](tools.md) — the full tool subsystem reference.
- [../for-contributors/develop.md](../for-contributors/develop.md) — the
  conventions this provider must follow.

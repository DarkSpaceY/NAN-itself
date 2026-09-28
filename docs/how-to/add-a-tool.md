# Add a tool

This guide adds a tool provider to NAN-itself. There are two kinds of
provider, and you pick one: an **MCP server** (a stdio subprocess
described by a YAML file) or a **local Python provider** (an in-process
`ToolSet` subclass). Both kinds are hot-reloaded, so no restart is
needed.

To add the provider to the shipped set, create the file under
`builtin/tools/`; to keep it in your own territory, create it under
`workspace/tools/` with the same layout. For the full provider model, see
[../reference/tools.md](../reference/tools.md).

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

Create `builtin/tools/local/<name>.py` containing exactly one concrete
`ToolSet` subclass:

```python
# @tool
"""One-line provider description."""

from __future__ import annotations

from typing import Any


class MyProvider(ToolSet):

    id = "<name>"

    @tool
    async def do_thing(self, query: str, limit: int = 10) -> Any:
        """What the tool does.

        Args:
            query: argument description.
            limit: argument description.
        """
        ...
```

The loader injects `ToolSet`, `@tool`, `text_result` and `error_result`
into the file's namespace, so the file needs no framework imports to
hot-reload. An explicit import of the same name simply shadows the
injection.

Method rules:

- Keep the `# @tool` header within the first 20 lines (convention: line
  1, before the docstring).
- Give the class a unique `id`.
- Fully type-annotate every parameter — the JSON schema derives from the
  annotations. `*args` / `**kwargs` are not supported.
- The docstring's first line is the tool summary; the `Args:` section
  documents each parameter.
- Put heavy imports inside the method body, not at module top level.
- Anchor any repo-relative path through `nan_itself.utils.paths`; never
  depend on the process working directory.

## Verify

Check that the provider loads without restarting the process: the runtime
scans every second, so a valid new file appears through `list_tools` (as
`<name>/<tool>`) within a few seconds. Inspect it with `show_tool` and
call it with `invoke_tool`.

## Related

- [../reference/tools.md](../reference/tools.md) — the full tool
  subsystem reference.
- [develop.md](develop.md) — the conventions this provider must follow.

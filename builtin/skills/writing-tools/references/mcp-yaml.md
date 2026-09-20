# MCP provider YAML reference

One YAML file in `builtin/tools/mcps/` or `workspace/tools/mcps/`
declares exactly one stdio MCP server. Canonical source:
`backend/nan_itself/tools/spec.py` and `runtime.py`.

## Fields

| Field    | Required | Meaning |
|---|---|---|
| `name`   | yes | Provider name; must be unique across builtin + workspace sources. The file name should match. |
| `command`| yes | Executable to spawn: `uvx`, `npx`, or an absolute path. |
| `args`   | no  | Argument list passed to `command`. |
| `env`    | no  | Extra environment variables merged over the inherited user environment. |
| `cwd`    | no  | Working directory for the subprocess. A **relative** `cwd` is anchored to the repo root, never to the process working directory. |

## Recipes

PyPI-hosted server (preferred — zero local install):

```yaml
name: touchpoint
command: uvx
args:
  - "--from"
  - "touchpoint-py"
  - "touchpoint-mcp"
```

With a dependency constraint (the server package does not pin its own
dependency and needs a downgrade):

```yaml
name: touchpoint
command: uvx
args:
  - "--from"
  - "touchpoint-py"
  - "--with"
  - "mcp<2"
  - "touchpoint-mcp"
```

npm-hosted server:

```yaml
name: some-server
command: npx
args: ["-y", "some-mcp-server"]
```

## Behavior notes

- Servers are spawned as stdio MCP subprocesses and inherit the user
  environment; they are trusted, unrestricted subprocesses by design —
  review a config before adding it.
- A dedicated supervisor reaps dead workers and reschedules their
  sources with exponential backoff; a crashing server degrades its own
  provider and never kills the runtime.
- Editing the YAML hot-reloads the provider transactionally: the new
  worker is started first, committed only after connecting, then the old
  worker is stopped. Deleting the file disables the provider.
- All tool calls through the provider are timeout-bounded
  (`providers.tool_timeout`, 300 s default).

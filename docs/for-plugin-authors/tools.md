# Tools

A tool is a callable capability the agent reaches through the
`invoke_tool` verb. Tools are grouped into **providers**; each provider
is one source file, loaded and hot-reloaded by
[`ProviderRuntime`](../../backend/nan_itself/tools/runtime.py). This
document describes the tool subsystem: its two provider kinds, the
source layout, the reload model, the call bounds, and the verb surface.
It does not describe how to create a provider — see
[../how-to/add-a-tool.md](../how-to/add-a-tool.md) for that.

Tools are never addressed by the model directly. The model sees the
verb triples `list_tools` / `show_tool` / `invoke_tool` and refers to a
tool by its composite `provider/tool` name.

## Provider kinds

Two interchangeable kinds exist, distinguished by `ProviderSpec.kind`
(`backend/nan_itself/tools/spec.py`):

| Kind | Source | Backend | Live object |
|---|---|---|---|
| `mcp` | one YAML file | stdio MCP server (subprocess) | `Provider.stack` + `Provider.session` |
| `local` | one Python file | in-process Python class | `Provider.instance` (a `ToolSet`) |

Both kinds populate the same `Provider.tools` table (`dict[str,
mcp.types.Tool]`) and are invoked through the same `Provider.call_tool`
entry point. The provider layer does not depend on the local backend;
`instance` is duck-typed.

### MCP providers

An MCP provider is one stdio MCP server described by a YAML file with
the fields `name`, `command`, `args`, `env` and `cwd`:

- `name` is optional; when omitted the file stem is used.
- `env` is merged *over* the inherited user environment.
- A relative `cwd` is anchored to the repository root, never to the
  process working directory.
- The `mcp_servers` multi-provider mapping is rejected: a source file
  declares exactly one provider.

On connect, the runtime starts the subprocess, performs the MCP
handshake (`initialize` + `list_tools`), and caches the returned tool
table. The handshake is bounded by `mcp_start_timeout` (60 s default);
the MCP SDK itself has no default read timeout. A tool that appears in a
live server after connection is picked up by a one-shot `list_tools`
refresh when `resolve_tool` misses on the cached table.

### Local providers

A local provider is a Python file whose first 20 lines contain the
literal header `# @tool` and which defines exactly one concrete
`ToolSet` subclass with a unique `ClassVar id`. The loader injects the
local-tool vocabulary — `ToolSet`, `@tool`, `text_result`,
`error_result` — into the file's namespace, so a provider file needs no
framework imports to hot-reload.

`ToolSet` (`backend/nan_itself/tools/local.py`) is the base class for
in-process providers. On construction it walks the class MRO base-first
and collects every `@tool`-decorated method into a `LocalToolMethod`
table; an override that redeclares `@tool` replaces the base version,
and two `@tool` methods with the same name in one class are an error.

### `@tool` and schema derivation

`@tool` marks a method as exposed. It is usable as `@tool`, `@tool()`,
or `@tool(name="other_name", description="...")`. Without an explicit
description the method docstring is used.

For each `@tool` method the loader derives a pydantic input model from
the signature (the same annotation-driven mechanism `ChannelSpec`
uses):

- Every parameter must be type-annotated; unannotated parameters are an
  error, and `*args` / `**kwargs` are not supported.
- Parameters with defaults become optional; parameters without become
  required.
- The generated model forbids unknown argument names
  (`extra="forbid"`), so a mistyped argument surfaces as a validation
  error rather than silently applying defaults.

`LocalToolMethod.input_schema()` returns that model's JSON schema with
`title` fields stripped — this is the schema handed to the model through
`show_tool`. At call time the arguments are validated against the same
model, so the schema and the validation derive from one source.

## Sources and discovery

Providers live in two parallel roots scanned with identical semantics:

| Root | Contents |
|---|---|
| `builtin/tools/mcps/` | Shipped MCP YAML configs (one file = one provider) |
| `builtin/tools/local/` | Shipped local Python providers (one file = one provider) |
| `workspace/tools/mcps/` | User territory, same shape (starts empty) |
| `workspace/tools/local/` | User territory, same shape (starts empty) |

Discovery rules:

- **One file = one provider.** An MCP YAML file declares exactly one
  server; a local Python file defines exactly one concrete `ToolSet`.
- A provider name belongs to exactly one source file; a second source
  claiming a live name fails that source's reload and leaves the
  existing provider intact.
- Files whose name starts with `_` are ignored. Only `.yaml` / `.yml`
  files are picked up under `mcps/`; only `.py` files that carry the
  `# @tool` header are picked up under `local/` (recursively, unlike the
  flat `mcps/` scan).

The live providers are held in `ProviderRuntime.providers` (name →
`Provider`); MCP workers are additionally held in `_mcp_workers`.

## Hot reload

Reload is a transaction. A replacement MCP worker is created as an
**unregistered candidate** (`register=False`): it owns its own MCP
connection but is not visible through `providers` until the transaction
commits. Only after the candidate is fully connected is it committed
into the live tables, and only then is the old generation asked to stop.
Old and new generations of the same provider name therefore coexist
during the handoff. Deleting a source file removes its provider.

A dedicated supervisor task reconciles sources every
`providers.scan_interval` seconds (1.0 s default), reaps dead MCP
workers, and reschedules failed sources with the shared exponential
backoff. Local providers are rebuilt by re-reading and re-compiling the
source file (never through the bytecode cache, which could serve stale
code); the previous synthetic module name is evicted from `sys.modules`.
A degraded provider never kills the runtime.

## Call bounds and results

Every tool call is timeout-bounded (`providers.tool_timeout`,
`DEFAULT_TOOL_TIMEOUT` = 300 s). The wrapper is `asyncio.wait_for`
around `Provider.call_tool`; a timeout is returned to the model as an
error result, never raised into the agent loop.

Local call semantics:

- An `async` handler is awaited directly; a synchronous handler is
  offloaded to the default thread pool (`asyncio.to_thread`) so one
  blocking call cannot freeze the shared event loop, and an awaitable
  returned by a sync handler is awaited back on the loop.
- Return values are serialized by `serialize_value`: `None` → `"null"`,
  a `str` passes through unquoted, anything else is JSON-encoded.
- A `pydantic.ValidationError` or any other exception becomes an error
  result (`error_result` sets `isError=True`).

Both provider kinds return `mcp.types.CallToolResult`. `text_result`
builds a plain text result; `error_result` builds one with
`isError=True`.

## Verb surface

Tools are reached only through the agent's verb triple
(`backend/nan_itself/agent/verbs.py`):

| Verb | Behaviour |
|---|---|
| `list_tools` | Returns one `provider/tool` line per available tool, in stable order. |
| `show_tool` | Takes a `provider/tool` name and returns its description and input schema as JSON. |
| `invoke_tool` | Takes a `provider/tool` name and an `arguments` object and calls the tool. |

`invoke_tool` renders the result structurally: an
`mcp.types.CallToolResult` is dumped with `model_dump_json()`, a plain
`str` passes through unquoted, and anything else is JSON-encoded with
`str()` as the fallback for non-serializable values.

The same `list_*` / `show_*` / `invoke_*` mental model applies to skills
(`list_skills` / `show_skill` / `invoke_skill`) and module channels
(`list_channels` / `show_channels` / `invoke_channels`).

## Live sources

The shipped providers are:

- MCP configs in [`builtin/tools/mcps/`](../../builtin/tools/mcps/)
  (`calculation`, `command`, `files`, `playwright`, `prolog`,
  `touchpoint`).
- Local Python providers in
  [`builtin/tools/local/`](../../builtin/tools/local/) (the web search
  provider `SearchProvider`, id `search`, exposing `web_search`).

The tool surface is still evolving; the sources above are the
authoritative list.

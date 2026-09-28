# Develop NAN-itself

This guide covers the day-to-day developer workflow: setting up the
environment, running the test suite, following the conventions the test
suite enforces, and opening a pull request. It is the single source of
truth for those conventions.

NAN-itself is in early development. Open an issue first and align on the
approach before writing code, so the change lands in one pass.

## Set up your environment

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
```

The project has two dependency sets:

- the **framework** (`dependencies`) — the agent core, the gateway and
  the tool runtime. `uv sync` alone installs these.
- the **builtin plugins** (the `perception` extra) — what the shipped
  modules and tools under `builtin/` import. Without them those plugins
  come up loudly failed; `--all-extras` installs them.

CI installs the framework only.

## Run the test suite

```bash
uv run pytest -q # run everything
```

The suite has two parts:

- `tests/` — the framework and contract suite. Hermetic: it fakes
  subprocesses, network and hardware, so it runs anywhere, including CI.
- `tests/builtin/` — tests for the builtin content: the modules and
  tools that ship as plugins (`audio`, `voice`, `vision`, `search`).
  These import the real builtin sources and therefore need the full
  local-inference stack. **CI excludes them**; run them locally when you
  have the dependencies installed:

```bash
uv run pytest -q tests/builtin        # builtin content only
uv run pytest -q --ignore=tests/builtin   # framework suite only
```

- The suite must stay green before every commit.
- Async tests use explicit `@pytest.mark.asyncio` markers
  (pytest-asyncio in strict mode).
- Tool-level tests fake their backends (subprocesses, network); keep the
  suite hermetic.
- Path-anchoring rules are enforced by `tests/test_paths_anchoring.py`,
  including foreign-cwd process-level checks. The one test that must
  boot the real builtin modules is deselected in CI.

## Follow the conventions

These are load-bearing — violating any of them breaks guarantees the test
suite enforces.

- **Path anchoring.** Every repo-relative path (`data/`, `models/`,
  `config/`, `builtin/`) must resolve through
  [`backend/nan_itself/utils/paths.py`](../../backend/nan_itself/utils/paths.py)
  (`repo_root()` / `data_dir()` / `models_dir()`). Never use
  cwd-relative defaults or private `parents[N]` lookups — the process
  must behave identically when started from any directory.
- **One file = one unit.** Each source file in `builtin/` or
  `workspace/` declares exactly one provider, module class, or skill. For
  tool providers specifically: `builtin/tools/mcps/*.yaml` is one stdio
  MCP server per file (`name` / `command` / `args` / `env` / `cwd`), and
  `builtin/tools/local/*.py` is one `ToolSet` subclass per file with a
  `# @tool` header within the first 20 lines.
- **Reload is a transaction.** A replacement is started as an
  unregistered candidate and committed only after it is fully connected;
  the old generation keeps serving until that moment. Never mutate live
  tables in place.
- **Import discipline.** `agent/` may import `modules/model.py`
  (contracts), never `modules/runtime.py`; the root package stays
  import-light (see `tests/test_layering.py`).
- **Modules are self-contained.** `backend/nan_itself/utils/` holds only
  core-architecture helpers (path anchoring, backoff). Module-related
  code — DSP toolkits, model adapters, registries — lives inside the
  module file itself (`builtin/modules/audio.py`, `voice.py`,
  `vision.py`), even when that makes the file long. Modules never import
  from each other; they share data only through facts (uplink) and
  channels (downlink).
- **`ask()` stays cheap.** Heavy work belongs in `start()` loops and
  `tell()`; a slow `ask()` delays every agent.
- **Local tool methods** are decorated with `@tool`, must be fully
  type-annotated (the JSON schema derives from the annotations), and may
  be `async`. Return values are JSON-serialized into the tool result;
  exceptions become error results.
- **Lazy heavy imports.** Import expensive dependencies inside the tool
  method body, not at module top level.
- **Retries** use the shared exponential backoff helper in
  [`backend/nan_itself/utils/backoff.py`](../../backend/nan_itself/utils/backoff.py)
  (`next_backoff(values, index)`).
- **Logging** goes through `loguru`.
- **Comments and identifiers are in English.**

### Terminology

One term per concept. Use the identifier column verbatim in prose and
code; the last column lists the synonyms that are wrong and should not
appear in docs or comments.

| Term | Identifier | Meaning | Do not call it |
|---|---|---|---|
| Agent core | `Agent` (`agent/core.py`) | The turn-driven loop; holds the last `Turn`. | bot, assistant, runner |
| Engine | `StepEngine` (`agent/engine.py`) | Executes one round: builds the snapshot, calls the LLM, renders the turn. | runtime, orchestrator, scheduler |
| Turn | `Turn` (`modules/model.py`) | The sole record of one LLM round. | round, iteration, message |
| Verb | `list_tools` / `show_tool` / `invoke_tool`, plus the same triple for skills and channels (`agent/verbs.py`) | A cognitive action the model can take. Tools and skills use singular names, channels plural. | command, action, function |
| Tool provider | `Provider` / `ProviderRuntime` (`tools/`) | One source file exposing a group of tools. | plugin, connector, tool manager |
| MCP provider | kind `"mcp"` (`tools/spec.py`) | A provider backed by a stdio MCP server described by a YAML file. | remote tool, external tool |
| Local provider | kind `"local"` (`tools/spec.py`) | A provider backed by an in-process Python class. | in-process tool, Python tool |
| ToolSet | `ToolSet` (`tools/local.py`) | Base class for a local provider; holds its `@tool` methods. | LocalToolProvider, Tool, Provider |
| `@tool` | `@tool` (`tools/local.py`) | Marks a `ToolSet` method as an exposed tool; the schema derives from its annotations. | @tool_method, @action |
| Tool | `mcp.types.Tool` | One callable a provider exposes, addressed as `provider/tool`. | function, capability |
| Module | `Module` (`modules/model.py`) | A long-lived background service contributing ambient context. | plugin, service, sensor, daemon |
| Facade | `Facade` (`modules/runtime.py`) | The module runtime supervisor: discovery, hot reload, dependency graph, lifecycle, channels. | ModuleManager, registry |
| Channel | `channels` / `ChannelSpec` (`modules/model.py`) | A model-writable downlink endpoint a module opts into. | output, queue, command |
| DataSpace | `DataSpace` / `DataSpaceReader` (`modules/model.py`) | A module's published state; readers get detached deep copies. | state store, blackboard, bus |
| Skill | `SKILL.md` directory (`skills/`) | A capability package: a directory with `SKILL.md` plus optional resources. | plugin, prompt, tool |
| Gateway | `Gateway` (`gateway.py`) | The single inbound HTTP listener, bound to loopback by default. | server, backend, API |
| builtin | `builtin/` | The sources shipped with the repository. | core, default, system |
| workspace | `workspace/` | The user-editable root, scanned with the same semantics as `builtin/`. | overrides, user dir, local dir |
| Persona | `workspace/persona.md` | The user-editable prompt identity prepended to the agent's context. | system prompt, character |
| Ambient context | `Module.ask()` | The per-turn projection a module contributes at turn start. | perception, telemetry, sensing |
| Hot reload | reload transaction (`tools/runtime.py`, `modules/reload.py`) | Replacing a live source: the replacement starts unregistered and is committed only once connected. | refresh, restart, rescan |

## Open a pull request

- Keep the full test suite green: `uv run pytest -q`.
- Add tests for behavior changes; contract changes need a doc update in
  `docs/`.
- English for code, comments, and documentation.

## Related

- [../for-plugin-authors/add-a-tool.md](../for-plugin-authors/add-a-tool.md) — add a tool provider.
- [../for-plugin-authors/add-a-module.md](../for-plugin-authors/add-a-module.md) — add an ambient or reactive module.
- [../for-plugin-authors/add-a-skill.md](../for-plugin-authors/add-a-skill.md) — add a skill package.

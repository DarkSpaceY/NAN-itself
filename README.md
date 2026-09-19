# NAN-itself

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-117%20passing-brightgreen)](#development)

A general-purpose, local-first autonomous agent framework. NAN-itself runs an
agent loop on your own machine, wiring an LLM to hot-reloadable tools (MCP
servers and in-process Python providers), ambient perception modules
(audio/vision/system), skills, and an HTTP gateway with a web UI.

> **NAN-itself is in early development.** Interfaces may change without notice.

## Features

- **Turn-based agent core** — every round is captured in a single `Turn`
  record (history snapshot + messages), so any round can be restored or
  inspected exactly as the model saw it.
- **Hot-reloadable tool providers** — two interchangeable kinds, reloaded
  live from source with zero restarts:
  - **MCP providers**: one YAML file = one stdio MCP server.
  - **Local providers**: one Python file = one in-process tool class.
- **Parallel builtin + workspace sources** — builtin and workspace tool
  directories are scanned with identical hot-reload semantics; deleting a
  source file disables its provider.
- **Ambient perception modules** — audio, vision (photometry → flow →
  faces → YOLO → OCR → VLM captioning), inbox, memory, network, plan and
  system modules run as background daemons and are queried by the agent.
- **Skills** — hot-reloadable skill packages following the same
  builtin/workspace split.
- **Bounded by design** — every tool call is timeout-bounded, the only
  inbound listener is the gateway bound to `127.0.0.1`, and all repo paths
  resolve through a single path-anchoring module.

## Architecture

```mermaid
flowchart LR
    U[User] --> G[Gateway<br/>FastAPI on 127.0.0.1:8765]
    G <--> C[Agent core<br/>turn loop + snapshot history]
    C <--> T[Tool runtime<br/>hot-reload providers]
    T --> MCP[MCP servers<br/>stdio subprocesses]
    T --> LP[Local Python tools<br/>in-process]
    C <--> SK[Skills<br/>hot-reload]
    C <--> MD[Modules<br/>audio / vision / inbox / memory / ...]
```

See [docs/architecture.md](docs/architecture.md) for the full picture.

## Quick start

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/DarkSpaceY/NAN-itself.git
cd NAN-itself
uv sync

# Point the LLM settings at your provider, then start the agent:
# config/settings.yaml -> llm.api_key / llm.base_url / llm.model
uv run nan-itself
```

The gateway listens on `http://127.0.0.1:8765` by default.

## Development

```bash
uv sync              # installs the dev group (pytest) by default
uv run pytest -q     # full test suite
```

Contributor conventions (path anchoring, one-file-one-provider, hot-reload
contracts) are documented in [docs/development.md](docs/development.md).

## Project layout

```
NAN-itself/
├── builtin/
│   ├── modules/          # Ambient perception modules (audio, vision, ...)
│   ├── skills/           # Built-in skills
│   └── tools/
│       ├── mcps/         # MCP server configs (one YAML = one provider)
│       └── local/        # Local Python tool providers (one file = one provider)
├── config/               # settings.yaml and tool settings (e.g. searxng.yml)
├── docs/                 # Documentation
├── frontend/             # Web UI
├── models/               # Model weights (gitignored, auto-downloaded)
├── src/nan_itself/       # Framework source
│   ├── agent/            # Core agent loop (turns, engine, verbs)
│   ├── tools/            # Provider runtime (MCP + local backends, hot reload)
│   ├── modules/          # Module base machinery
│   ├── skills/           # Skill machinery
│   ├── gateway.py        # HTTP gateway
│   └── utils/            # paths, backoff, vision/audio helpers
├── tests/                # Pytest suite
└── workspace/            # User workspace (gitignored), hot-reloadable overrides
```

## Documentation

| Document | Contents |
|---|---|
| [Architecture](docs/architecture.md) | Layers, turn snapshot model, tool runtime, security model |
| [Configuration](docs/configuration.md) | `settings.yaml` reference and environment variables |
| [Development](docs/development.md) | Setup, conventions, adding tools and modules |
| [Tools](docs/tools.md) | Built-in tool reference *(placeholder)* |
| [Modules](docs/modules.md) | Built-in module reference *(placeholder)* |

## Built-in tools

> The tool surface is still evolving, so a detailed catalog is intentionally
> deferred. See [`builtin/tools/`](builtin/tools/) for the live sources:
> YAML MCP configs in `mcps/` and Python providers in `local/`.

## Built-in modules

> Same as above — the module set is still moving. See
> [`builtin/modules/`](builtin/modules/) for the live sources.

## License

[Apache-2.0](LICENSE)

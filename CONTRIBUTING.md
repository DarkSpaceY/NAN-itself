# Contributing to NAN

NAN is an early project. This file gets you from a clone to a running
window, then to a pull request that is easy to review. If something here
is wrong or missing, that is itself a good first issue.

## What kind of help is useful

- **Core runtime** (`nan/`): turn loop, agent lifecycle, sandbox, recovery,
  snapshot and replay.
- **Desktop** (`electron/`): the Electron shell and the React renderer.
- **Services** (`workspace/services/`, `lib/nan-service-kit/`): injection
  services, SEGV, MCP integration.
- **Docs and testing**: docs are part of the product; Linux/WSL2 setup
  reports and doc drift are useful contributions.

## Set up

Requirements: git, Node.js 18+, and a POSIX shell. The setup script
installs `uv` when it is missing; Python 3.12 is fetched by `uv`.

The fast path:

```bash
git clone https://github.com/DarkSpaceY/NAN-itself.git
cd NAN-itself
sh scripts/install.sh
```

Without a checkout, the same script clones into `~/nan`:

```bash
curl -fsSL https://raw.githubusercontent.com/DarkSpaceY/NAN-itself/main/scripts/install.sh | sh
```

Then edit `config/nan.yaml` and set the provider, model and API key, and
run:

```bash
uv run nan
```

If you prefer to see every step, the manual path is:

```bash
uv sync
cp config/nan.yaml.example config/nan.yaml
npm --prefix electron install
npm --prefix electron run build
uv run nan
```

### First run downloads

The first start is network-heavy. Expect:

- MCP servers fetched by `npx --yes` / `uvx` on first use, and the local
  search helper installing its own dependency.
- Playwright downloading its browser into the MCP service's cache
  directory.
- SEGV, which `workspace/autostart.sh` starts by default, needs Ollama on
  `127.0.0.1:11434` with two small models:

  ```bash
  ollama pull qwen3.5:4b-mlx      # the shipped config uses the MLX build
  ollama pull embeddinggemma:300m
  ```

  The reranker (`BAAI/bge-reranker-base`) runs in-process and its weights
  are fetched from Hugging Face on first use. If the shipped model names
  are not available on your platform, point the `model` fields in
  `workspace/services/segv/config.yaml` at models you have. Without these
  models SEGV cannot serve memory.

## Development loop

```bash
uv run pytest -q                      # core tests
uv run pytest tests/test_agent.py -q  # one file
uv run ruff check .                   # lint
uv run ruff format .                  # format
uv run mypy                           # types over nan/
```

Services and the shared kit are separate uv projects:

```bash
uv run --directory workspace/services/mcp --frozen pytest -q
uv run --directory workspace/services/segv --frozen pytest -q
uv run --directory lib/nan-service-kit --frozen pytest -q
```

Renderer:

```bash
npm --prefix electron run check   # tsc --noEmit
npm --prefix electron run test    # vitest
npm --prefix electron run build   # production bundle
```

For renderer-only work, run `uv run nan` for the gateway and
`npm --prefix electron run dev` for Vite on `:5173`; the dev proxy
forwards `/ws` to `127.0.0.1:8765`.

Pre-commit runs the same checks as CI, scoped by path:

```bash
uv sync
npm --prefix electron ci
uv run pre-commit install
```

## Where things live

| Path | What |
|---|---|
| `nan/` | core runtime: agents, turns, execution, gateway, persistence |
| `lib/nan-service-kit/` | shared config, hub link and HTTP helpers for services |
| `workspace/services/` | services the agent uses (system, network, mcp, segv) |
| `builtin/` | user-owned skills, services and autostart scripts |
| `electron/` | Electron shell and React renderer |
| `docs/` | user, contributor and service-author docs |
| `tests/` | core test suite |

Start with [architecture](docs/for-contributors/architecture.md),
[principles](docs/for-contributors/principles.md) and
[developing](docs/for-contributors/develop.md). If you change behavior,
update the matching doc in the same PR.

## Picking up work

- Issues labeled `good first issue` are small and self-contained.
- Issues labeled `help wanted` are larger; comment before starting.
- For a new feature or a cross-cutting change, open an issue first and
  describe the problem, the shape of the change and the alternatives.
  Design discussion is cheap; a 2000-line PR is not.
- Security-sensitive reports go through [SECURITY.md](SECURITY.md), not a
  public issue.

## Pull requests

- One logical change per PR; keep it reviewable.
- Use Conventional Commits-style subjects: `feat:`, `fix:`, `docs:`,
  `refactor:`, `test:`, `chore:`.
- Run the checks that cover the files you touched; CI must be green.
- Add or update tests for behavior changes. For docs-only changes, say so.
- Public docs and code comments are in English.
- By contributing you agree that your work is licensed under
  [Apache-2.0](LICENSE).

## Known rough edges

These are real, and some are good first issues:

- **Workspace files are not snapshotted.** Recovery restores agent and
  turn state, not the files the agent changed; there is no rollback for
  the workspace yet.
- **First-run downloads are heavy**: MCP servers, Playwright, models.
- **No packaged app.** The desktop window runs from the checkout.
- **SEGV needs Ollama models** (above). macOS and Linux are supported;
  Windows runs inside WSL2.

If you get stuck, open an issue with the command you ran, what you
expected, and the relevant log output.

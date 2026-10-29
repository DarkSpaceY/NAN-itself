# Developing

## Setup

    uv sync                      # runtime + dev tools
    npm --prefix frontend ci     # for the tsc / vitest hooks
    uv run pre-commit install

## Test

    uv run pytest tests/ -q

The suite (83 tests) covers the agent loop (pause/resume, death
cascade, persona re-read, depth/children limits), injection
hub/client, sandbox behavior, gateway entry guard and config
loading. Shared fakes (scripted LLM, slow LLM) live in
`tests/nan_fakes.py`.

## Service tests

Services under `workspace/services/` and `builtin/services/` are
independent uv projects, each with its own `tests/`. Run a suite
from its directory:

    cd workspace/services/<name> && uv run --frozen pytest

Services share `lib/nan-service-kit` (config, hub lifecycle,
logging, contract test doubles). The suites CI runs use `--frozen`
(a stale `uv.lock` fails the build, so dependency resolution stays
pinned); the list lives in `.github/workflows/ci.yml`.

## Lint and format

    uv run ruff check .
    uv run ruff format .

Line length 100; isort-style import order enforced by ruff.

## Type check

    uv run mypy nan/

## Conventions (For AI Agents)

- Comments and docs describe what the code does today; no changelog,
  task markers, or references to earlier behaviour.
- Docstrings state behavior contracts. Absolute marketing voice
  ("never", "always" as defense of a design) is avoided; state
  the actual mechanism.
- New configuration keys go into `config/nan.yaml` with a
  comment, the matching pydantic model in `nan/config/`, and the
  configuration reference in `docs/for-users/configuration.md`.

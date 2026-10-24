# Developing

## Setup

    uv sync              # runtime + dev tools
    pre-commit install   # ruff check + format on commit

## Test

    uv run pytest tests/ -q

The suite (30 tests) covers the agent loop (pause/resume, death
cascade, persona re-read, depth/children limits), injection
hub/client, sandbox behavior, gateway entry guard and config
loading. Shared fakes (scripted LLM, slow LLM) live in
`tests/nan_fakes.py`.

## Service tests

Services under `workspace/services/` are independent uv projects;
the core test suite never touches them. Each carries its own
`tests/`, run from its own directory:

    cd workspace/services/<name> && uv run --frozen pytest

CI runs every service suite with `--frozen` (a stale `uv.lock`
fails the build instead of silently resolving).

## Lint and format

    uv run ruff check nan/ tests/
    uv run ruff format nan/ tests/

Line length 100; isort-style import order enforced by ruff.

## Type check

    uv run mypy nan/

Checks run on `nan/` only. `legacy/` (the parked pre-rewrite
resources) is excluded from ruff and mypy and is not part of the
runtime.

## Conventions

- Comments describe what the code does, not what it used to do;
  no changelog or task-marker comments.
- Docstrings state behavior contracts. Absolute marketing voice
  ("never", "always" as defense of a design) is avoided; state
  the actual mechanism.
- New configuration keys go into `config/nan.yaml` with a
  comment, the matching pydantic model in `nan/config/`, and the
  configuration reference in `docs/for-users/configuration.md`.

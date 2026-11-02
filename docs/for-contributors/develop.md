# Developing

## Setup

    uv sync                      # runtime + dev tools
    npm --prefix frontend ci     # for the tsc / vitest hooks
    uv run pre-commit install

## Test

    uv run pytest tests/ -q

The suite (214 tests) is organized one directory per core feature:

    tests/agent/        the loop (control fold, pause/resume, epochs,
                        persona, pacing, failed-turn terminal), the spawn
                        tree, failure classification, and each verb
    tests/turn/         the Turn record (JSON, fresh shape) and rendering
    tests/persistence/  snapshots and in-flight recovery
    tests/boot/         restore of the whole tree
    tests/execution/    the command sandbox
    tests/injection/    hub registry, guard and the two delivery modes
    tests/llm/          provider mapping (reasoning replay)
    tests/gateway/      entry guard, WS protocol, surfaces
    tests/events/       the sink ring and stream coalescing
    tests/config/       settings, paths and the YAML template
    tests/utils/        loopback policy and autostart
    tests/app/          Core lifecycle and gateway callbacks
    tests/smoke/        end-to-end agent-tree scenarios

Shared fakes (scripted LLM, slow LLM) live in `tests/nan_fakes.py`;
`pythonpath` includes `tests/` so every subdirectory can import it.

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

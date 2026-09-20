# Contributing

Thanks for your interest in NAN-itself. The project is in early
development, so the fastest way to contribute is to open an issue first
and align on the approach before writing code.

## Setup

See [docs/development.md](docs/development.md) for the full guide.
Short version:

```bash
uv sync          # runtime + dev group
uv run pytest -q # full suite
```

## Conventions

The codebase has a few load-bearing conventions — violating any of them
breaks guarantees the test suite enforces:

- **Path anchoring.** Never use the process working directory; resolve
  everything through `backend/nan_itself/utils/paths.py`.
- **One file = one provider / module / skill.** Each source file in
  `builtin/` or `workspace/` declares exactly one provider, module
  class, or skill.
- **Hot reload is a transaction.** Replacements are validated before
  they are committed; never mutate live tables in place.
- **Import discipline.** `agent/` may import `modules/model.py`
  (contracts), never `modules/runtime.py`; the root package stays
  import-light (see `tests/test_layering.py`).
- **`query()` stays cheap.** Heavy work belongs in `start()` loops and
  `on_turn()`; a slow `query()` delays every agent.

## Pull requests

- Keep the full test suite green: `uv run pytest -q`.
- Add tests for behavior changes; contract changes need a doc update in
  `docs/`.
- English for code, comments, and documentation.

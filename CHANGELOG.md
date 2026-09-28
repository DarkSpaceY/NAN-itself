# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/)
once a stable release line starts.

## [Unreleased]

### Added

- Reactive modules: modules can declare **channels** — write-only
  downlink data slots the model reaches through the
  `list_channels` / `show_channels` / `invoke_channels` verb triple
  (pydantic-annotated payload schemas, declarative depth: overwrite or
  FIFO, schema validation at the write boundary).
- All lazy imports moved to module top level for deterministic import
  ordering.
- Repository restructured: `src/nan_itself/` → `backend/nan_itself/`
  (import name unchanged), symmetric with `frontend/` and `infraend/`.
- Open-source facade: CI workflow, CONTRIBUTING, SECURITY, changelog.

### Changed

- Model weight provisioning moved into module `start()` — no downloads
  inside tick loops.
- Dependency alignment: `webrtcvad-wheels` replaces `webrtcvad`;
  `setuptools` pinned `<81` for `face_recognition`; vision VLM loader
  migrated to the transformers v5 API.
- Test suite split: the tests for the builtin content — the modules and
  tools that ship as plugins and need the full local-inference stack —
  moved to `tests/builtin/`, which CI excludes.
- Dependencies split: what the builtin plugins import moved out of
  `dependencies` into the `perception` extra. `uv sync` now installs the
  framework alone and `uv sync --all-extras` adds the plugins, so CI no
  longer needs a dozen per-package exclusions. `pyyaml`, imported
  directly by the config loader and the MCP backend, and `soundfile`,
  imported by the voice module, are now declared instead of relying on
  transitively installed copies.
- Documentation reorganized by audience — users, plugin authors and
  contributors — replacing the Diataxis quadrant structure; the
  documentation writing standard was dropped, with its terminology
  glossary folded into
  [`docs/for-contributors/develop.md`](docs/for-contributors/develop.md).
- Record stream (protocol v2): the backend no longer emits pre-rendered
  display text. Records carry a semantic `category` with a typed
  `payload`, details are typed entries (`text` / `item` / `field` /
  `code`), and durations are numbers of seconds instead of formatted
  strings; the frontend owns all rendering. See
  [`frontend/PROTOCOL.md`](frontend/PROTOCOL.md).
- Module registration is validate-then-install: the dependency graph is
  checked on a copy before the record reaches any live table, so a
  dependency cycle is rejected with nothing installed. Previously the
  record was installed first and rolled back on failure, which left the
  graph cyclic and stalled `start()`, `stop()` and reconciliation for
  every module.
- Module removal now releases the module: its state is persisted, its
  `DataSpace` is dropped, and it is detached from every dependent, so a
  dependent stops reading a removed module's last published values. A
  later re-add restores the persisted state and re-attaches those
  dependents. Previously the `DataSpace` lived on as a tombstone for the
  rest of the process.
- A module stop is bounded by a timeout, so a module whose `stop()`
  blocks — or whose task swallows cancellation — can no longer stall the
  module supervisor.

### Removed

- `memory` and `plan` builtin modules (superseded by the channels
  design and upcoming reactive flows).
- Root `CONTRIBUTING.md`, merged into
  [`docs/for-contributors/develop.md`](docs/for-contributors/develop.md).

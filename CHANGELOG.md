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

### Removed

- `memory` and `plan` builtin modules (superseded by the channels
  design and upcoming reactive flows).

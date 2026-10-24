"""Configuration: YAML loading, path resolution, provider
profiles, and the no-environment-variables rule."""

from __future__ import annotations

from pathlib import Path

import pytest

from nan.config import PACKAGE_DIR, REPO_ROOT
from nan.config.loader import load_settings, resolve_llm

YAML = """
llm:
  provider: local27b
  model: fallback-model
  providers:
    local27b:
      provider: openai
      base_url: http://127.0.0.1:8080/v1
      model: qwen-27b
      api_key: none
      timeout: 30.0

agent:
  autonomous_interval: 5.0
  shell_timeout: 10.0
  max_depth: -1
  history_char_limit: 1234

sandbox:
  enabled: false
  required: true

injection:
  port: 9001

gateway:
  port: 8000
  frontend_dir: frontend/dist

workspace: ws
builtin_dir: builtin

autostart:
  enabled: false
"""


def test_load_explicit_yaml(tmp_path: Path) -> None:
    config = tmp_path / "config" / "nan.yaml"

    config.parent.mkdir()

    config.write_text(YAML, encoding="utf-8")

    cfg = load_settings(str(config))

    # relative paths anchor at the config file's grandparent
    assert cfg.workspace == (tmp_path / "ws").resolve()
    assert cfg.builtin_dir == (tmp_path / "builtin").resolve()
    assert cfg.gateway.frontend_dir == (tmp_path / "frontend" / "dist").resolve()

    # persona defaults into the nan package, autostart into the workspace
    assert cfg.agent.persona_path == PACKAGE_DIR / "persona.md"
    assert cfg.autostart.path == cfg.workspace / "autostart.sh"
    assert cfg.autostart.enabled is False

    assert cfg.agent.autonomous_interval == 5.0
    assert cfg.agent.shell_timeout == 10.0
    assert cfg.agent.max_depth == -1
    assert cfg.agent.history_char_limit == 1234
    assert cfg.sandbox.enabled is False
    assert cfg.sandbox.required is True
    assert cfg.injection.port == 9001
    assert cfg.gateway.port == 8000

    # provider profile resolution: named profile overrides the base
    llm = resolve_llm(cfg)

    assert llm.provider == "openai"
    assert llm.base_url == "http://127.0.0.1:8080/v1"
    assert llm.model == "qwen-27b"
    assert llm.api_key == "none"
    assert llm.timeout == 30.0
    assert llm.max_retries == 2  # inherited from the base section


def test_resolve_llm_without_profile_uses_base() -> None:
    cfg = load_settings()

    assert cfg.llm.providers == {}

    llm = resolve_llm(cfg)

    assert llm is cfg.llm


def test_default_yaml_is_the_committed_template() -> None:
    """load_settings() with no argument reads config/nan.yaml at
    the repo root; the committed template carries defaults."""
    cfg = load_settings()

    assert cfg.workspace == REPO_ROOT / "workspace"
    assert cfg.builtin_dir == REPO_ROOT / "builtin"
    assert cfg.gateway.port == 8765
    assert cfg.injection.port == 8901
    assert cfg.sandbox.enabled is True
    assert cfg.sandbox.required is False
    assert cfg.agent.autonomous_interval == 0.0
    assert cfg.autostart.enabled is True


def test_environment_variables_are_not_a_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hostile environment must not leak into Settings."""
    monkeypatch.setenv("gateway", "not-a-dict")
    monkeypatch.setenv("NAN_GATEWAY_PORT", "9999")
    monkeypatch.setenv("LLM", "garbage")

    config = tmp_path / "c.yaml"

    config.write_text("gateway:\n  port: 8000\n", encoding="utf-8")

    cfg = load_settings(str(config))

    assert cfg.gateway.port == 8000


def test_non_mapping_yaml_rejected(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"

    config.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_settings(str(config))

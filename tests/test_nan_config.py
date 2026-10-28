"""Configuration: YAML loading, the no-environment-variables rule,
and paths being code (not config)."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from nan.config import Settings, settings
from nan.llm.provider import _proxy_kwargs

YAML = """
llm:
  provider: anthropic
  model: fallback-model

agent:
  autonomous_interval: 5.0
  shell_timeout: 10.0
  max_depth: -1
  history_char_limit: 1234

sandbox:
  required: true

injection:
  port: 9001

gateway:
  port: 8000

autostart:
  enabled: false
"""


def _settings(text: str) -> Settings:
    data: Any = yaml.safe_load(text)

    return Settings(**data)


def test_load_yaml_text() -> None:
    cfg = _settings(YAML)

    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.model == "fallback-model"

    assert cfg.agent.autonomous_interval == 5.0
    assert cfg.agent.shell_timeout == 10.0
    assert cfg.agent.max_depth == -1
    assert cfg.agent.history_char_limit == 1234
    assert cfg.sandbox.required is True
    assert cfg.injection.port == 9001
    assert cfg.gateway.port == 8000
    assert cfg.autostart.enabled is False


def test_proxy_is_explicit() -> None:
    assert _settings("llm:\n  proxy: env\n").llm.proxy == "env"

    default = _settings("llm:\n  model: x\n")

    assert default.llm.proxy == ""
    assert _proxy_kwargs(default.llm.proxy) == {"trust_env": False, "proxy": None}
    assert _proxy_kwargs("http://127.0.0.1:7890") == {
        "trust_env": False,
        "proxy": "http://127.0.0.1:7890",
    }
    assert _proxy_kwargs("env") == {"trust_env": True}


def test_unknown_protocol_rejected() -> None:
    with pytest.raises(ValueError):
        _settings("llm:\n  provider: gemini\n")


def test_paths_are_not_configuration() -> None:
    """
    Path-like keys in the YAML are inert: paths come from
    config.Paths, so a config cannot move the workspace.
    """
    cfg = _settings(
        """
workspace: elsewhere
builtin_dir: elsewhere
agent:
  persona_path: ./p.md
gateway:
  frontend_dir: /tmp/frontend
autostart:
  path: /tmp/autostart.sh
"""
    )

    assert not hasattr(cfg, "workspace")
    assert not hasattr(cfg, "builtin_dir")
    assert not hasattr(cfg.agent, "persona_path")
    assert not hasattr(cfg.gateway, "frontend_dir")
    assert not hasattr(cfg.autostart, "path")


def test_live_config_and_committed_template_agree() -> None:
    """The process-wide settings load config/nan.yaml at import.

    The live file is git-ignored (it holds the key); the template is
    what a checkout actually carries, so it must stay loadable and
    keep the same non-secret defaults."""
    assert settings.paths.config.is_file()
    assert settings.llm.provider == "openai"
    assert settings.gateway.host == "127.0.0.1"
    assert settings.gateway.port == 8765
    assert settings.injection.host == "127.0.0.1"
    assert settings.injection.port == 8901
    assert settings.sandbox.required is False
    assert settings.agent.autonomous_interval == 0.0
    assert settings.autostart.enabled is True

    template = settings.paths.config.with_name("nan.yaml.example")
    assert template.is_file()
    loaded = yaml.safe_load(template.read_text(encoding="utf-8"))
    assert loaded["llm"]["api_key"] == ""
    assert loaded["gateway"]["port"] == settings.gateway.port


def test_environment_variables_are_not_a_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hostile environment must not leak into Settings."""
    monkeypatch.setenv("gateway", "not-a-dict")
    monkeypatch.setenv("NAN_GATEWAY_PORT", "9999")
    monkeypatch.setenv("LLM", "garbage")

    cfg = _settings("gateway:\n  port: 8000\n")

    assert cfg.gateway.port == 8000

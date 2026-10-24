"""
Loading + path resolution for nan Settings.

Relative paths in the YAML resolve against the directory that
contains the config (its parent, conventionally the repo root).
Defaults anchor to the nan package for persona, workspace for autostart.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger

from . import REPO_ROOT, LLMSettings, Settings

_LLM_OVERRIDE_FIELDS = ("provider", "base_url", "api_key", "model", "timeout", "max_retries")


def load_settings(path: str | None = None) -> Settings:
    """
    Load Settings. None -> the default config/nan.yaml via the
    pydantic-settings YAML source; an explicit path is read
    directly (base for relative paths: the file's grandparent,
    conventionally the repo root).
    """
    if path is None:
        return _resolve_paths(Settings(), base=REPO_ROOT)

    config_file = Path(path)

    data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config {config_file} must be a YAML mapping")

    base = config_file.resolve().parent.parent

    logger.info("Loaded nan config from {}", config_file)

    return _resolve_paths(Settings(**data), base=base)


def resolve_llm(settings: Settings) -> LLMSettings:
    """
    Merge the base llm section with the named provider profile
    (llm.provider picks the profile; profile fields override).
    """
    base = settings.llm

    profile = base.providers.get(base.provider)

    if profile is None:
        return base

    overrides = {
        field: getattr(profile, field)
        for field in _LLM_OVERRIDE_FIELDS
        if getattr(profile, field) is not None
    }

    return LLMSettings(**{**base.model_dump(exclude={"providers"}), **overrides})


def _resolve_paths(settings: Settings, *, base: Path) -> Settings:
    """
    Anchor every configurable path: relative -> against `base`;
    persona defaults into the nan package, autostart into the
    workspace.
    """
    settings.workspace = _anchor(settings.workspace, base)

    settings.builtin_dir = _anchor(settings.builtin_dir, base)

    settings.agent.persona_path = _anchor(settings.agent.persona_path, base)

    if settings.autostart.path is None:
        settings.autostart.path = settings.workspace / "autostart.sh"
    else:
        settings.autostart.path = _anchor(settings.autostart.path, base)

    if settings.gateway.frontend_dir is not None:
        settings.gateway.frontend_dir = _anchor(settings.gateway.frontend_dir, base)

    return settings


def _anchor(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()

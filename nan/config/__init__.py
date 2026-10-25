"""
Configuration: one process-wide object.

`settings` is loaded once at import from config/nan.yaml and read
directly by every component. Paths live in the same object
(`paths`), fixed by the installation.

The schema is in models.py; this module owns the live instance and
is the single import point.
"""

from __future__ import annotations

from .models import (
    AgentSettings,
    AutostartSettings,
    GatewaySettings,
    InjectionSettings,
    LLMSettings,
    Paths,
    ProviderName,
    SandboxSettings,
    Settings,
)

# The single configuration object: loaded once, read everywhere.
settings = Settings()

__all__ = [
    "AgentSettings",
    "AutostartSettings",
    "GatewaySettings",
    "InjectionSettings",
    "LLMSettings",
    "Paths",
    "ProviderName",
    "SandboxSettings",
    "Settings",
    "settings",
]

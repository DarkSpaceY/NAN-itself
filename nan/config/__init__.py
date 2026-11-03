"""
Configuration: the schema, and the one live object its owner builds.

The schema is in models.py. The composition root constructs one
`Settings` instance at startup and hands it to every component.

Paths live in the installation's fixed layout (`models.PATHS`) and are
read through the instance's `paths` property.
"""

from __future__ import annotations

from .models import (
    AgentSettings,
    AutostartSettings,
    DenyRule,
    GatewaySettings,
    InjectionSettings,
    LLMSettings,
    Paths,
    ProviderName,
    SandboxSettings,
    Settings,
)

__all__ = [
    "AgentSettings",
    "AutostartSettings",
    "DenyRule",
    "GatewaySettings",
    "InjectionSettings",
    "LLMSettings",
    "Paths",
    "ProviderName",
    "SandboxSettings",
    "Settings",
]

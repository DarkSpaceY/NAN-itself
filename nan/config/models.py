"""
The configuration schema: the shape of config/nan.yaml, plus the
fixed filesystem layout.

`nan.config` builds the process-wide `settings` instance from
these models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# nan/config/models.py -> nan/config -> nan/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]

_PACKAGE_DIR = Path(__file__).resolve().parents[1]

# The protocols nan.llm.provider implements.
ProviderName = Literal["openai", "anthropic"]


class Paths(BaseModel):
    """
    Fixed layout: every component reads the same object.
    """

    package: Path = _PACKAGE_DIR
    config: Path = _REPO_ROOT / "config" / "nan.yaml"
    workspace: Path = _REPO_ROOT / "workspace"  # agents' writable territory
    builtin: Path = _REPO_ROOT / "builtin"  # read-only resources
    persona: Path = _PACKAGE_DIR / "persona.md"  # re-read every turn
    frontend: Path = _REPO_ROOT / "frontend" / "dist"  # built desktop
    autostart: Path = _REPO_ROOT / "builtin" / "autostart.sh"  # user-owned, unsandboxed
    agent_autostart: Path = _REPO_ROOT / "workspace" / "autostart.sh"  # agent-writable, sandboxed


class LLMSettings(BaseModel):
    """The endpoint in force: one protocol, one set of fields."""

    provider: ProviderName = "openai"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: float = 120.0
    max_retries: int = 2
    proxy: str = ""


class AgentSettings(BaseModel):
    autonomous_interval: float = 0.0
    history_char_limit: int = 100_000
    shell_timeout: float = 30.0
    shell_output_limit: int = Field(default=20_000, ge=2)
    backoff: list[float] = Field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0])
    max_depth: int = 3
    max_children: int = 5


class DenyRule(BaseModel):
    """One command-level refusal rule: a regex and why it refuses."""

    pattern: str
    reason: str


class SandboxSettings(BaseModel):
    required: bool = True
    # Rules come from config; an absent `deny` means no rules.
    deny: list[DenyRule] = Field(default_factory=list)


class InjectionSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8901
    timeout: float = 2.0
    heartbeat_ttl: float = 15.0


class GatewaySettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    dedup_cache_size: int = 256
    surface_ttl: float = 30.0


class AutostartSettings(BaseModel):
    enabled: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file=str(Paths().config),
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    paths: Paths = Field(default_factory=Paths)

    llm: LLMSettings = Field(default_factory=LLMSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    injection: InjectionSettings = Field(default_factory=InjectionSettings)
    gateway: GatewaySettings = Field(default_factory=GatewaySettings)
    autostart: AutostartSettings = Field(default_factory=AutostartSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # YAML is the only source.
        return (init_settings, YamlConfigSettingsSource(settings_cls))

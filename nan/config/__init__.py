"""
Configuration: pydantic-settings models sourced from a YAML
file (config/nan.yaml).
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

# nan/config/__init__.py -> nan/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

# The nan package itself (persona / autostart default location)
PACKAGE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_YAML_PATH = REPO_ROOT / "config" / "nan.yaml"


class ProviderProfile(BaseModel):
    """
    A named LLM endpoint profile. Fields left as None inherit
    the base `llm` section. A profile becomes active when
    `llm.provider` names its key.
    """

    provider: Literal["openai", "anthropic"] | None = None
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout: float | None = None
    max_retries: int | None = None


class LLMSettings(BaseModel):
    """The active LLM section. `provider` names the active entry:
    "openai" | "anthropic" directly, or the key of a profile in
    `providers` (whose own provider field carries the protocol
    type). `providers` holds optional named endpoint profiles."""

    provider: str = "openai"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: float = 120.0
    max_retries: int = 2
    providers: dict[str, ProviderProfile] = Field(default_factory=dict)


class AgentSettings(BaseModel):
    autonomous_interval: float = 0.0
    history_char_limit: int = 100_000
    shell_timeout: float = 30.0
    shell_output_limit: int = 20_000
    backoff: list[float] = Field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0])
    max_depth: int = 3
    max_children: int = 5
    persona_path: Path = PACKAGE_DIR / "persona.md"


class SandboxSettings(BaseModel):
    enabled: bool = True
    required: bool = False


class InjectionSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8901
    timeout: float = 2.0
    heartbeat_ttl: float = 15.0


class GatewaySettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    frontend_dir: Path | None = None


class AutostartSettings(BaseModel):
    enabled: bool = True
    path: Path | None = None  # default: nan/autostart.sh


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file=str(DEFAULT_YAML_PATH),
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    injection: InjectionSettings = Field(default_factory=InjectionSettings)
    gateway: GatewaySettings = Field(default_factory=GatewaySettings)
    autostart: AutostartSettings = Field(default_factory=AutostartSettings)

    workspace: Path = REPO_ROOT / "workspace"
    builtin_dir: Path = REPO_ROOT / "builtin"

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

"""配置管理"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "openai"
    api_key: str = "local"
    model: str = "llama3.2:3b"
    base_url: str = "http://127.0.0.1:11434/v1"
    timeout: int = 600
    max_retries: int = 2

class GatewayConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765


class AgentConfig(BaseModel):
    max_subagent_depth: int = 3
    history_char_limit: int = 100_000


class SkillsConfig(BaseModel):
    resource_char_limit: int = 100_000
    script_timeout: float = 300.0


class TurnConfig(BaseModel):
    grace: float = 5.0


class RetryConfig(BaseModel):
    backoff: tuple[float, ...] = (
        1.0,
        2.0,
        4.0,
        8.0,
        15.0,
        30.0,
        60.0,
    )


class RuntimeConfig(BaseModel):
    turn: TurnConfig = Field(default_factory=TurnConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)


class ModulesConfig(BaseModel):
    retry_interval: float = 1.0
    scan_interval: float = 1.0


class ProvidersConfig(BaseModel):
    scan_interval: float = 1.0
    tool_timeout: float = 300.0


class EventsConfig(BaseModel):
    history_limit: int = 500
    subscriber_queue_size: int = 2000
    input_dedup_cache_size: int = 256


class Settings(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    modules: ModulesConfig = Field(default_factory=ModulesConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)


def load_settings(
    config_file: str = "config/settings.yaml",
) -> Settings:
    """加载配置"""
    path = Path(config_file)

    if not path.exists():
        return Settings()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid settings YAML: {path}"
        )

    return Settings(**data)


_settings: Settings | None = None


def get_settings() -> Settings:
    """获取配置"""
    global _settings

    if _settings is None:
        _settings = load_settings()

    return _settings


def reload_settings() -> Settings:
    global _settings

    _settings = load_settings()

    return _settings
"""
What every service needs to know about itself: who it is, where the
core runs, which port it serves, and how the hub should treat it.

A service reads one YAML file next to its main.py. Precedence is
defaults < file: a key the file omits keeps its default. Services
subclass ServiceConfig for their own knobs, so the shared keys stay
typed in one place and every service parses them the same way.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

Mode = Literal["inject", "archive"]

HUB_PORT = 8901
GATEWAY_PORT = 8765


def _hub_url() -> str:
    """The hub the core runs on; it exports its port to autostart."""
    port = os.environ.get("NAN_INJECTION_PORT") or HUB_PORT

    return f"http://127.0.0.1:{port}"


class Surface(BaseModel):
    """The panel a service hosts in the desktop, when it has one."""

    model_config = ConfigDict(extra="allow")

    title: str
    url: str = ""  # empty: the service's own root, on its own port
    icon: str | None = None
    every: float = 10.0  # well under the gateway's surface TTL (30s)


class ServiceConfig(BaseModel):
    """
    The keys every service has. Unknown keys are ignored, so a service
    adds its own without touching this schema.
    """

    model_config = ConfigDict(extra="allow")

    service_id: str  # identity in the observation stream; defaults to the directory name
    host: str = "127.0.0.1"
    port: int
    mode: Mode = "inject"
    hub_url: str = Field(default_factory=_hub_url)
    gateway_url: str = f"http://127.0.0.1:{GATEWAY_PORT}"
    surface: Surface | None = None
    log_level: str = "INFO"

    _home: Path = PrivateAttr(default=Path())

    @property
    def home(self) -> Path:
        """The service's own directory, where its config.yaml sits."""
        return self._home

    @property
    def logs_dir(self) -> Path:
        """Where this service writes its logs; disposable."""
        return self._home / "logs"

    @property
    def endpoint(self) -> str:
        """The URL the hub calls for a turn."""
        return f"http://{self.host}:{self.port}/turn"


def read_config[ConfigT: ServiceConfig](model: type[ConfigT], path: Path) -> ConfigT:
    """
    Parse one YAML file into `model`, defaulting service_id to the
    file's directory name. The service's home is bound here, so
    logs_dir is always under the directory the config was shipped in.
    Everything else a service writes is its own choice under `home`.
    """
    path = path.resolve()

    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"{path} must hold a YAML mapping")

    raw.setdefault("service_id", path.parent.name)

    config = model.model_validate(raw)

    config._home = path.parent

    return config

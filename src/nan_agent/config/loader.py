import os
from pathlib import Path
from typing import Any, Optional

import yaml

from nan_agent.exceptions import ConfigError
from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)

DEFAULTS_DIR = Path(__file__).resolve().parent
DEFAULT_USER_CONFIG_PATH = Path.home() / ".nan_agent" / "config.yaml"


class ConfigLoader:
    def __init__(self, user_config_path: Optional[str] = None):
        if user_config_path is not None:
            self.user_config_path = user_config_path
            self._user_config_explicit = True
        else:
            self.user_config_path = str(DEFAULT_USER_CONFIG_PATH)
            self._user_config_explicit = False

    def load(self) -> dict:
        defaults_path = DEFAULTS_DIR / "defaults.yaml"
        config = self._load_yaml(str(defaults_path))

        if os.path.exists(self.user_config_path):
            user_config = self._load_yaml(self.user_config_path)
            config = self._deep_merge(config, user_config)
        elif self._user_config_explicit:
            raise ConfigError(
                f"Config file not found: {self.user_config_path}",
                error_code="E101",
                details={"path": self.user_config_path},
            )

        config = self._apply_env_overrides(config)

        return config

    def _load_yaml(self, path: str) -> dict:
        if not os.path.exists(path):
            raise ConfigError(
                f"Config file not found: {path}",
                error_code="E101",
                details={"path": path},
            )

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(
                f"Invalid YAML in config file: {path}",
                error_code="E102",
                details={"path": path, "error": str(e)},
            ) from e

        if data is None:
            return {}

        if not isinstance(data, dict):
            raise ConfigError(
                f"Config file must contain a mapping, got {type(data).__name__}: {path}",
                error_code="E103",
                details={"path": path},
            )

        return data

    def _deep_merge(self, base: dict, override: dict) -> dict:
        result = dict(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_env_overrides(self, config: dict) -> dict:
        for env_key, env_value in os.environ.items():
            if not env_key.startswith("NAN_"):
                continue

            key_str = env_key[len("NAN_"):].lower()
            key_path = self._resolve_key_path(config, key_str)
            coerced = self._coerce_value(env_value)
            self._set_nested(config, key_path, coerced)

        return config

    def _resolve_key_path(self, config: dict, key_str: str) -> list:
        parts = key_str.split("_")
        resolved = []
        i = 0
        while i < len(parts):
            matched = False
            for j in range(len(parts), i, -1):
                candidate = "_".join(parts[i:j])
                if candidate in config:
                    resolved.append(candidate)
                    if isinstance(config[candidate], dict):
                        config = config[candidate]
                    i = j
                    matched = True
                    break
            if not matched:
                resolved.append(parts[i])
                i += 1
        return resolved

    def _coerce_value(self, value: str) -> Any:
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        try:
            return int(value)
        except ValueError as e:
            logger.debug("config_coerce_int_failed", value=value, error=str(e))

        try:
            return float(value)
        except ValueError as e:
            logger.debug("config_coerce_float_failed", value=value, error=str(e))

        return value

    def _set_nested(self, config: dict, keys: list, value: Any) -> None:
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

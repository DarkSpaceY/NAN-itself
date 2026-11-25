"""NAN's view of the same YAML: hub identity plus the core's knobs."""

from __future__ import annotations

from pathlib import Path

from nan_service_kit import ServiceConfig, read_config

from segv.config import CoreConfig


class NanConfig(ServiceConfig):
    """Two hub registrations served by one process: archive and inject."""

    archive_id: str = "segv-archive"

    inject_id: str = "segv"

    def core(self) -> CoreConfig:
        config = CoreConfig.model_validate(self.model_dump())

        config._home = self.home

        return config


def load_config(path: Path) -> NanConfig:
    return read_config(NanConfig, path)

"""
Where a service writes its own log.

Two files, two owners. The boot script captures stdout/stderr in
logs/<name>.boot.log before loguru exists: uv failures, import errors,
uvicorn's warnings. From the moment a service is up, loguru writes
logs/<service_id>.log and nothing else, so the boot capture stays a
small startup record and this file is the one that rotates.

Logs live under logs/, apart from whatever state the service keeps;
deleting that directory loses no state.
"""

from __future__ import annotations

from loguru import logger

from .config import ServiceConfig

ROTATION = "10 MB"
RETENTION = 3


def setup_logging(config: ServiceConfig) -> None:
    """
    Point loguru at this service's log file and drop its stderr sink;
    call once, before the first log line.
    """
    config.logs_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        config.logs_dir / f"{config.service_id}.log",
        level=config.log_level,
        rotation=ROTATION,
        retention=RETENTION,
        encoding="utf-8",
    )

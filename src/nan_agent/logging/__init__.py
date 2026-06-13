"""NAN-Agent logging package.

Exposes a thin, opinionated wrapper around :mod:`structlog` that adds:

- rotating file output (opt-in via config)
- per-module level overrides
- correlation IDs (``bind_correlation_id`` / ``new_correlation_id``)
- standardized key/value events (``log_event``)
- a lightweight :class:`Timer` context manager
"""

from nan_agent.logging.logger import (
    DEFAULT_FORMAT,
    Timer,
    bind_correlation_id,
    clear_correlation_id,
    current_correlation_id,
    get_logger,
    log_event,
    new_correlation_id,
    setup_logger,
)

__all__ = [
    "DEFAULT_FORMAT",
    "Timer",
    "bind_correlation_id",
    "clear_correlation_id",
    "current_correlation_id",
    "get_logger",
    "log_event",
    "new_correlation_id",
    "setup_logger",
]

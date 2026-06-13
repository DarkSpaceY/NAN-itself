"""NAN-Agent logging system.

Features
--------
- Console (colored) or JSON output.
- Optional rotating file output with independent level/format.
- Per-module log level overrides (e.g. silence noisy libraries).
- Correlation IDs via :mod:`structlog.contextvars` that propagate across
  ``asyncio`` tasks spawned with ``contextvars.copy_context()``.
- Standardized event helper :func:`log_event` (event name + key/value payload).
- :class:`Timer` context manager for cheap performance metrics.

Public API
----------
- :func:`setup_logger`  — configure handlers/levels (call once at startup).
- :func:`get_logger`    — obtain a bound structlog logger.
- :func:`bind_correlation_id` / :func:`new_correlation_id` /
  :func:`clear_correlation_id` — manage correlation IDs.
- :func:`log_event`     — emit a standardized key/value event.
- :class:`Timer`        — measure and log elapsed time.

Backward compatibility
----------------------
``setup_logger(level=..., output=...)`` keeps its old signature.  The new
keyword arguments ``file`` (dict) and ``module_levels`` (dict) are optional.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import structlog


DEFAULT_FORMAT = "console"


def _build_processors() -> list:
    """Return the shared processor list used by both console and JSON output."""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]


def _make_renderer(output: str):
    if output == "json":
        return structlog.processors.JSONRenderer()
    return structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())


def _parse_size(size_str: str) -> int:
    """Parse a human-readable size like ``"10 MB"`` into bytes."""
    s = size_str.strip().upper().replace(" ", "")
    if not s:
        return 10 * 1024 * 1024
    units = {"B": 1, "K": 1024, "KB": 1024, "M": 1024**2, "MB": 1024**2,
             "G": 1024**3, "GB": 1024**3}
    for suffix in sorted(units.keys(), key=len, reverse=True):
        if s.endswith(suffix):
            try:
                return int(float(s[: -len(suffix)]) * units[suffix])
            except ValueError:
                return 10 * 1024 * 1024
    try:
        return int(s)
    except ValueError:
        return 10 * 1024 * 1024


def _build_file_handler(
    file_cfg: Dict[str, Any],
    renderer_output: str,
) -> Optional[logging.Handler]:
    if not file_cfg.get("enabled", False):
        return None

    path = file_cfg.get("path", "./logs/nan_agent.log")
    rotation = file_cfg.get("rotation", "10 MB")
    file_level = getattr(logging, str(file_cfg.get("level", "INFO")).upper(), logging.INFO)
    file_format = file_cfg.get("format", renderer_output)

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if rotation.isdigit():
        handler: logging.Handler = logging.handlers.TimedRotatingFileHandler(
            filename=str(log_path),
            when="midnight",
            backupCount=int(rotation),
            encoding="utf-8",
        )
    else:
        handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path),
            maxBytes=_parse_size(rotation),
            backupCount=file_cfg.get("backup_count", 5),
            encoding="utf-8",
        )

    handler.setLevel(file_level)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=_make_renderer(file_format),
            foreign_pre_chain=_build_processors(),
        )
    )
    return handler


def setup_logger(
    level: str = "INFO",
    output: str = DEFAULT_FORMAT,
    *,
    file: Optional[Dict[str, Any]] = None,
    module_levels: Optional[Dict[str, str]] = None,
) -> None:
    """Configure root logging + structlog.

    Parameters
    ----------
    level:
        Default root log level.  One of ``DEBUG``/``INFO``/``WARNING``/``ERROR``.
    output:
        ``"console"`` (default) or ``"json"``.
    file:
        Optional dict with keys ``enabled``, ``path``, ``rotation``,
        ``level``, ``format``, ``backup_count`` for a rotating file handler.
    module_levels:
        Optional mapping ``{module_name: level}`` to override per-module levels.
    """
    log_level = getattr(logging, str(level).upper(), logging.INFO)
    file = file or {}
    module_levels = module_levels or {}

    structlog.configure(
        processors=_build_processors() + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=_make_renderer(output),
        foreign_pre_chain=_build_processors(),
    )

    root = logging.getLogger()
    root.setLevel(log_level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = _build_file_handler(file, output)
    if file_handler is not None:
        root.addHandler(file_handler)

    for module_name, mod_level in module_levels.items():
        numeric = getattr(logging, str(mod_level).upper(), None)
        if numeric is not None:
            logging.getLogger(module_name).setLevel(numeric)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger.

    The returned logger automatically merges the active correlation ID (set via
    :func:`bind_correlation_id` or :func:`new_correlation_id`) into every log
    record.
    """
    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# Correlation IDs
# ---------------------------------------------------------------------------

_CORRELATION_KEY = "correlation_id"


def new_correlation_id(prefix: str = "run") -> str:
    """Generate and bind a fresh correlation ID.  Returns the new ID."""
    cid = f"{prefix}-{uuid.uuid4().hex[:12]}"
    structlog.contextvars.bind_contextvars(**{_CORRELATION_KEY: cid})
    return cid


def bind_correlation_id(correlation_id: str) -> None:
    """Bind an externally-provided correlation ID (e.g. from a CLI request)."""
    structlog.contextvars.bind_contextvars(**{_CORRELATION_KEY: correlation_id})


def clear_correlation_id() -> None:
    """Remove the current correlation ID from the contextvars stack."""
    structlog.contextvars.unbind_contextvars(_CORRELATION_KEY)


def current_correlation_id() -> Optional[str]:
    """Return the currently bound correlation ID, or ``None``."""
    ctx = structlog.contextvars.get_contextvars()
    return ctx.get(_CORRELATION_KEY)


# ---------------------------------------------------------------------------
# Standardized events
# ---------------------------------------------------------------------------

def log_event(
    logger: structlog.stdlib.BoundLogger,
    event: str,
    *,
    level: str = "info",
    **fields: Any,
) -> None:
    """Emit a standardized key/value event.

    Use this when you want a uniform shape across the codebase::

        log_event(logger, "task_completed", task_id=tid, duration_ms=12.3)

    The ``event`` string is also duplicated into ``event_name`` to make JSON
    log queries easier (``event`` is the human message; ``event_name`` is the
    stable machine identifier).
    """
    payload = dict(fields)
    payload.setdefault("event_name", event)
    bound = getattr(logger, level, None)
    if not callable(bound):
        bound = logger.info
    bound(event, **payload)


# ---------------------------------------------------------------------------
# Performance timing
# ---------------------------------------------------------------------------

class Timer:
    """Context manager that logs elapsed time when the block exits.

    Usage::

        with Timer(logger, "got_step", node_id=node.node_id):
            ...
    """

    def __init__(
        self,
        logger: structlog.stdlib.BoundLogger,
        event: str = "operation_complete",
        *,
        level: str = "debug",
        warn_threshold_ms: Optional[float] = None,
        **fields: Any,
    ) -> None:
        self._logger = logger
        self._event = event
        self._level = level
        self._warn_threshold_ms = warn_threshold_ms
        self._fields = fields
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, _exc, _tb) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        fields = dict(self._fields)
        fields["duration_ms"] = round(elapsed_ms, 3)
        if exc_type is not None:
            fields["error"] = exc_type.__name__
        threshold = self._warn_threshold_ms
        level = self._level
        if threshold is not None and elapsed_ms > threshold:
            level = "warning"
        log_event(self._logger, self._event, level=level, **fields)

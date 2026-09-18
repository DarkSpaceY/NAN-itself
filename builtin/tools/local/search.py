# @tool
"""
Builtin web search provider.

Backed by the PyPI `searxng-cli` package, which embeds a trimmed
SearXNG core (no SearXNG instance required) and prints one JSON
object per search. Every call runs a one-shot `uvx` subprocess —
the package's own one-process-per-search model.

Settings: `config/searxng.yml` (outgoing proxy, timeouts) is
injected via SEARXNG_CLI_SETTINGS. Set NAN_SEARXNG_SETTINGS to
point at a different settings file.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from typing import Any

from nan_itself.utils import paths as _paths

SEARXNG_CLI_SPEC = "searxng-cli==0.1.0"

DEFAULT_LANGUAGE = "all"

DEFAULT_MAX_RESULTS = 10

DEFAULT_TIMEOUT_S = 30.0


def _settings_file() -> str | None:
    override = os.environ.get(
        "NAN_SEARXNG_SETTINGS"
    )

    if override:
        return override

    default = (
        _paths.repo_root()
        / "config"
        / "searxng.yml"
    )

    return (
        str(default)
        if default.is_file()
        else None
    )


def _uvx_command() -> str:
    resolved = shutil.which("uvx")

    if not resolved:
        raise RuntimeError(
            "uvx executable not found on PATH; "
            "install uv to use the search tool"
        )

    return resolved


class SearchProvider(LocalToolProvider):

    id = "search"

    @tool
    async def web_search(
        self,
        query: str,
        engines: str | None = None,
        category: str | None = None,
        language: str = DEFAULT_LANGUAGE,
        safesearch: int = 0,
        time_range: str | None = None,
        pageno: int = 1,
        max_results: int = DEFAULT_MAX_RESULTS,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> Any:
        """Search the web with the embedded SearXNG core.

        Returns a JSON object with `results` (url/title/content/
        engine/score), `engine_stats`, `elapsed_sec` and
        `result_count`.

        Args:
            query: search query text.
            engines: comma-separated engine whitelist,
                e.g. "bing,duckduckgo". Omit for the curated
                per-category default engines.
            category: search category, e.g. general/it/science/
                images/news. Defaults to general.
            language: result language, e.g. "all"/"en"/"zh-CN".
            safesearch: safe-search level: 0 off, 1 moderate,
                2 strict.
            time_range: optional recency filter: day/week/month/
                year.
            pageno: result page number, starting at 1.
            max_results: maximum number of results returned.
            timeout: per-search timeout in seconds.
        """
        argv = [
            _uvx_command(),
            "--from",
            SEARXNG_CLI_SPEC,
            "searxng-cli",
            query,
            "--language",
            language,
            "--safesearch",
            str(safesearch),
            "--max-results",
            str(max_results),
            "--timeout",
            str(timeout),
        ]

        if engines:
            argv += ["--engines", engines]

        if category:
            argv += ["--category", category]

        if time_range:
            argv += [
                "--time-range",
                time_range,
            ]

        if pageno != 1:
            argv += ["--pageno", str(pageno)]

        env = dict(os.environ)

        settings = _settings_file()

        if settings:
            env["SEARXNG_CLI_SETTINGS"] = (
                settings
            )

        process = (
            await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        )

        try:
            stdout, stderr = (
                await process.communicate()
            )

        except asyncio.CancelledError:
            process.kill()

            raise

        if process.returncode != 0:
            detail = stderr.decode(
                "utf-8", "replace"
            ).strip()

            raise RuntimeError(
                f"searxng-cli exited with "
                f"{process.returncode}: {detail}"
            )

        try:
            payload = json.loads(
                stdout.decode("utf-8", "replace")
            )

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"searxng-cli produced invalid "
                f"JSON: {exc}"
            ) from exc

        return payload

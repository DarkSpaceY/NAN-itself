# /// script
# requires-python = ">=3.12"
# dependencies = ["fastmcp<3"]
# ///
"""
Web search MCP server, backed by the PyPI `searxng-cli` package
(raises a trimmed SearXNG core, no SearXNG instance needed). Every
call runs a one-shot `uvx` subprocess -- the package's own
one-process-per-search model.

Settings: the file named by SEARXNG_CLI_SETTINGS (proxy, timeouts)
is injected into every subprocess.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from typing import Any

from fastmcp import FastMCP

SEARXNG_CLI_SPEC = "searxng-cli==0.1.0"

DEFAULT_LANGUAGE = "all"

DEFAULT_MAX_RESULTS = 10

DEFAULT_TIMEOUT_S = 30.0

mcp = FastMCP("search")


def _uvx_command() -> str:
    resolved = shutil.which("uvx")

    if not resolved:
        raise RuntimeError("uvx executable not found on PATH; install uv to use the search tool")

    return resolved


@mcp.tool
async def web_search(
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
    engine/score), `engine_stats`, `elapsed_sec` and `result_count`.

    Args:
        query: search query text.
        engines: comma-separated engine whitelist,
            e.g. "bing,duckduckgo". Omit for the curated
            per-category default engines.
        category: search category, e.g. general/it/science/
            images/news. Defaults to general.
        language: result language, e.g. "all"/"en"/"zh-CN".
        safesearch: safe-search level: 0 off, 1 moderate, 2 strict.
        time_range: optional recency filter: day/week/month/year.
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
        argv += ["--time-range", time_range]

    if pageno != 1:
        argv += ["--pageno", str(pageno)]

    env = dict(os.environ)

    settings = os.environ.get("SEARXNG_CLI_SETTINGS")

    if settings and os.path.isfile(settings):
        env["SEARXNG_CLI_SETTINGS"] = settings

    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    try:
        stdout, stderr = await process.communicate()

    except asyncio.CancelledError:
        process.kill()

        raise

    if process.returncode != 0:
        detail = stderr.decode("utf-8", "replace").strip()

        raise RuntimeError(f"searxng-cli exited with {process.returncode}: {detail}")

    try:
        return json.loads(stdout.decode("utf-8", "replace"))

    except json.JSONDecodeError as exc:
        raise RuntimeError(f"searxng-cli produced invalid JSON: {exc}") from exc


if __name__ == "__main__":
    mcp.run()

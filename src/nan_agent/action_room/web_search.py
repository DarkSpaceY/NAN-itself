"""
Web 搜索模块 - DuckDuckGo 搜索引擎

提供基于 DuckDuckGo 的网页搜索、内容抓取、HTML 文本提取、
结果去重和相关性排序。内置速率限制和指数退避重试机制，
确保搜索的可靠性和礼貌性。

核心组件：
- WebSearch: 搜索引擎主类
- SearchResult: 搜索结果数据模型
- _TextExtractor: HTML 纯文本提取器
"""

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from ddgs import DDGS

from nan_agent.exceptions import ActionError
from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_RESULTS = 10
_DEFAULT_RATE_LIMIT = 1.0
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0
_BACKOFF_MAX = 30.0


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float = 0.5

    def __repr__(self) -> str:
        return (
            f"SearchResult(title={self.title!r}, url={self.url!r}, "
            f"source={self.source!r}, score={self.relevance_score:.2f})"
        )


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._text_parts: list[str] = []
        self._skip_tags = {"script", "style", "noscript", "meta", "link", "head"}
        self._current_skip: Optional[str] = None

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._current_skip = tag

    def handle_endtag(self, tag):
        if self._current_skip == tag:
            self._current_skip = None

    def handle_data(self, data):
        if self._current_skip is not None:
            return
        text = data.strip()
        if text:
            self._text_parts.append(text)

    def get_text(self) -> str:
        combined = " ".join(self._text_parts)
        combined = re.sub(r"\s+", " ", combined)
        return combined


def _url_fingerprint(url: str) -> str:
    parsed = urlparse(url)
    normalized = parsed._replace(fragment="", query="").geturl().rstrip("/").lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


class WebSearch:
    def __init__(
        self,
        max_results: int = _DEFAULT_MAX_RESULTS,
        rate_limit: float = _DEFAULT_RATE_LIMIT,
        timeout: float = _DEFAULT_TIMEOUT,
        proxy: Optional[str] = None,
    ):
        self._max_results = max_results
        self._rate_limit = rate_limit
        self._timeout = timeout
        self._proxy = proxy

        self._last_request_time: float = 0.0
        self._history: list[dict[str, Any]] = []
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            kwargs: dict[str, Any] = {
                "timeout": httpx.Timeout(self._timeout),
                "follow_redirects": True,
            }
            if self._proxy:
                kwargs["proxy"] = self._proxy
            self._http_client = httpx.AsyncClient(**kwargs)
        return self._http_client

    async def close(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def _rate_limit_wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit:
            await asyncio.sleep(self._rate_limit - elapsed)
        self._last_request_time = time.monotonic()

    async def _retry_with_backoff(self, coro_func, *args, **kwargs):
        last_exception: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                return await coro_func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < _MAX_RETRIES - 1:
                    delay = min(_BACKOFF_BASE ** attempt, _BACKOFF_MAX)
                    logger.warning(
                        "web_search_retry",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)
        raise last_exception  # type: ignore[misc]

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        region: str = "us-en",
    ) -> list[SearchResult]:
        await self._rate_limit_wait()

        max_n = max_results if max_results is not None else self._max_results

        try:
            raw = await self._retry_with_backoff(self._search_duckduckgo, query, max_n, region)
        except Exception as e:
            raise ActionError(
                f"Search failed: {e}",
                error_code="E502",
                details={"query": query, "backend": "duckduckgo"},
            ) from e

        results = self._deduplicate(raw)
        self._rank_by_relevance(results, query)

        self._history.append({
            "query": query,
            "backend": "duckduckgo",
            "result_count": len(results),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        logger.info(
            "web_search_completed",
            query=query,
            backend="duckduckgo",
            result_count=len(results),
        )

        return results

    async def _search_duckduckgo(
        self, query: str, max_results: int, region: str
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        loop = asyncio.get_running_loop()

        def _run():
            out = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, region=region, max_results=max_results):
                    out.append(r)
            return out

        ddg_results = await loop.run_in_executor(None, _run)

        for r in ddg_results:
            snippet = r.get("body", "")
##################这里进行了截断处理###############
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=snippet,
                source="duckduckgo",
                relevance_score=0.5,
            ))

        return results

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        seen: set[str] = set()
        unique: list[SearchResult] = []
        for r in results:
            fp = _url_fingerprint(r.url)
            if fp not in seen:
                seen.add(fp)
                unique.append(r)
        return unique

    def _rank_by_relevance(self, results: list[SearchResult], query: str) -> None:
        query_lower = query.lower()
        query_terms = set(t for t in query_lower.split() if t)

        # 权威域名加分
        _AUTHORITY_DOMAINS = {".edu": 0.1, ".gov": 0.1, ".org": 0.05, ".wikipedia.org": 0.15}

        for r in results:
            score = 0.5
            title_lower = r.title.lower()
            snippet_lower = r.snippet.lower()

            # 精确匹配
            if query_lower and query_lower in title_lower:
                score += 0.3
            elif query_lower and query_lower in snippet_lower:
                score += 0.15

            # 术语匹配
            matched_terms = sum(1 for t in query_terms if t and t in title_lower)
            score += matched_terms * 0.05
            matched_terms_snippet = sum(1 for t in query_terms if t and t in snippet_lower)
            score += matched_terms_snippet * 0.02

            # 域名权威性
            url_lower = r.url.lower()
            for domain, bonus in _AUTHORITY_DOMAINS.items():
                if domain in url_lower:
                    score += bonus

            # snippet 适度性（20-200 字符最佳）
            snippet_len = len(r.snippet)
            if snippet_len < 20:
                score -= 0.05
            elif snippet_len > 300:
                score -= 0.03

            r.relevance_score = min(max(round(score, 3), 0.0), 1.0)

        results.sort(key=lambda r: r.relevance_score, reverse=True)

    async def fetch_content(self, url: str) -> str:
        await self._rate_limit_wait()

        try:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            raise ActionError(
                f"Failed to fetch content from URL: {e}",
                error_code="E504",
                details={"url": url},
            ) from e

    def extract_text(self, html: str, max_length: int = 0) -> str:
        try:
            extractor = _TextExtractor()
            extractor.feed(html)
            text = extractor.get_text()
            if max_length > 0 and len(text) > max_length:
                text = text[:max_length] + "..."
            return text
        except Exception as e:
            raise ActionError(
                f"Failed to extract text from HTML: {e}",
                error_code="E505",
            ) from e

    @property
    def backend(self) -> str:
        return "duckduckgo"

    @property
    def max_results(self) -> int:
        return self._max_results
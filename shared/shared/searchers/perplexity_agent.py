"""Perplexity Agent API searcher — web_search + fetch_url for full-page text."""

import asyncio
import logging
import os
from typing import Any

import httpx

from .base import Searcher, SearchResult

PPLX_AGENT_URL = "https://api.perplexity.ai/v1/agent"
logger = logging.getLogger(__name__)


class PerplexityAgentSearcher(Searcher):
    """Perplexity /v1/agent: web_search to discover URLs, then fetch_url for full-page text.

    Step 1: web_search → top-N URLs + short snippets.
    Step 2: fetch_url per URL (parallel) → fetch_url_results.contents[].snippet, up to max_tokens.
    Falls back to web_search snippet if a fetch fails or returns no content.
    """

    def __init__(
        self,
        api_key: str | None = None,
        name: str = "pplx-agent",
        excluded_domains: list[str] | None = None,
        model: str = "openai/gpt-5-mini",
        max_tokens_per_page: int = 20000,
        max_steps: int = 1,
        max_tokens: int | None = None,
    ):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY") or os.getenv("PPLX_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY or PPLX_API_KEY required")
        self.name = name
        self.excluded_domains = excluded_domains or []
        self.model = model
        self.max_tokens_per_page = max_tokens_per_page
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(timeout=120.0)

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        # Step 1: web_search to discover URLs.
        web_search_tool: dict[str, Any] = {"type": "web_search"}
        if self.excluded_domains:
            domain_filter = [d if d.startswith("-") else f"-{d}" for d in self.excluded_domains]
            web_search_tool["filters"] = {"search_domain_filter": domain_filter}

        payload: dict[str, Any] = {
            "model": self.model,
            "input": query,
            "tools": [web_search_tool],
            "max_steps": self.max_steps,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        data = await self._post(payload)

        url_meta: list[tuple[str, str, str]] = []  # (url, title, snippet)
        for item in (data.get("output") or []):
            if item.get("type") == "search_results":
                for r in (item.get("results") or [])[:num_results]:
                    url_meta.append((r.get("url", ""), r.get("title", ""), r.get("snippet", "")))

        if not url_meta:
            return []

        # Step 2: fetch full content for each URL in parallel.
        fetched = await asyncio.gather(
            *[self._fetch_url(url, title, snippet) for url, title, snippet in url_meta],
            return_exceptions=True,
        )

        out: list[SearchResult] = []
        for i, result in enumerate(fetched):
            if isinstance(result, Exception):
                url, title, snippet = url_meta[i]
                logger.debug(f"[{self.name}] fetch failed for {url}: {result}")
                out.append(SearchResult(url=url, title=title, text=snippet, metadata={"rank": i}))
            else:
                out.append(result)
        return out

    async def _fetch_url(self, url: str, title: str, fallback_snippet: str) -> SearchResult:
        if not url:
            return SearchResult(url=url, title=title, text=fallback_snippet, metadata={"rank": 0})

        data = await self._post({
            "model": self.model,
            "input": url,
            "tools": [{"type": "fetch_url", "max_tokens": self.max_tokens_per_page}],
        })

        for item in (data.get("output") or []):
            if item.get("type") == "fetch_url_results":
                for content in (item.get("contents") or []):
                    text = content.get("snippet", "")
                    if text:
                        fetched_title = content.get("title") or title
                        return SearchResult(
                            url=content.get("url") or url,
                            title=fetched_title,
                            text=text,
                            metadata={"rank": 0},
                        )

        return SearchResult(url=url, title=title, text=fallback_snippet, metadata={"rank": 0})

    async def _post(self, payload: dict[str, Any]) -> dict:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = await self._client.post(
                    PPLX_AGENT_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503) and attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError):
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise

    async def close(self):
        await self._client.aclose()

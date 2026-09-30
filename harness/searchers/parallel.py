"""Parallel Search and Extract API adapter (v1).

Search sends ``search_queries`` (keyword queries) plus an optional natural-language
``objective`` in one request, so ``batches_queries`` is true: the Scout search tool
passes the model's objective and query list through unchanged. ``mode`` selects
``turbo``, ``fast``, ``basic`` or ``advanced``. Extract fetches known URLs and
returns objective-focused excerpts. Parallel does not report cost, so requests are
priced at the published rates.

API reference: https://docs.parallel.ai/search/search-quickstart
"""

import os
import time
from typing import Any

import httpx

from .base import Searcher, SearchResponse, SearchResult

PARALLEL_SEARCH_URL = "https://api.parallel.ai/v1/search"
PARALLEL_EXTRACT_URL = "https://api.parallel.ai/v1/extract"
PARALLEL_MODES = ("turbo", "fast", "basic", "advanced")
# https://parallel.ai/pricing — per request by mode, plus per result beyond 10.
PARALLEL_SEARCH_COST_PER_REQUEST = {
    "turbo": 0.001,
    "fast": 0.001,
    "basic": 0.005,
    "advanced": 0.005,
}
PARALLEL_COST_PER_EXTRA_RESULT = 0.001
_MAX_QUERY_CHARS = 5000


def clip(text: str, max_chars: int = _MAX_QUERY_CHARS) -> str:
    """Clip text to Parallel's per-field limit at a word boundary."""
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0]


def parse_results(data: dict[str, Any]) -> list[SearchResult]:
    """Map Parallel results to rows; excerpts become highlights, full content text."""
    results = []
    for rank, result in enumerate(data.get("results") or []):
        excerpts = result.get("excerpts") or []
        if not isinstance(excerpts, list):
            excerpts = [str(excerpts)]
        results.append(
            SearchResult(
                url=result.get("url") or "",
                title=result.get("title") or "",
                text=result.get("full_content") or "\n\n".join(excerpts),
                highlights=list(excerpts),
                metadata={
                    "rank": rank,
                    "author": result.get("author"),
                    "published_date": result.get("publish_date") or result.get("published_date"),
                },
            )
        )
    return results


class ParallelSearcher(Searcher):
    name = "parallel"
    batches_queries = True

    def __init__(
        self,
        api_key: str | None = None,
        mode: str = "advanced",
        source_policy: dict | None = None,
        excerpt_max_chars: int | None = None,
        fetch_policy: dict | None = None,
        timeout: float = 120.0,
    ):
        if mode not in PARALLEL_MODES:
            raise ValueError(f"mode must be one of {PARALLEL_MODES}, got {mode!r}")
        self.api_key = api_key or os.getenv("PARALLEL_API_KEY") or os.getenv("PARALLELS_API_KEY")
        if not self.api_key:
            raise ValueError("Parallel API key required - set PARALLEL_API_KEY or pass api_key")

        self.mode = mode
        self.source_policy = source_policy
        self.excerpt_max_chars = excerpt_max_chars
        self.fetch_policy = fetch_policy
        self._client = httpx.AsyncClient(
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    def search_payload(
        self,
        query: str,
        num_results: int,
        *,
        objective: str | None = None,
        search_queries: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build the ``/v1/search`` request body."""
        queries = [clip(q) for q in (search_queries or [query]) if q.strip()]
        advanced: dict[str, Any] = {"max_results": num_results}
        if self.excerpt_max_chars:
            advanced["excerpt_settings"] = {"max_chars_per_result": self.excerpt_max_chars}
        if self.source_policy:
            advanced["source_policy"] = self.source_policy
        if self.fetch_policy:
            advanced["fetch_policy"] = self.fetch_policy
        payload: dict[str, Any] = {
            "search_queries": queries,
            "mode": self.mode,
            "advanced_settings": advanced,
        }
        if objective and objective.strip():
            payload["objective"] = clip(objective.strip())
        return payload

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        return (await self.run(query, num_results)).results

    async def run(
        self,
        query: str,
        num_results: int = 10,
        *,
        objective: str | None = None,
        search_queries: list[str] | None = None,
    ) -> SearchResponse:
        start = time.perf_counter()
        payload = self.search_payload(
            query, num_results, objective=objective, search_queries=search_queries
        )
        data = await self._post(PARALLEL_SEARCH_URL, payload)
        cost = PARALLEL_SEARCH_COST_PER_REQUEST[self.mode] + PARALLEL_COST_PER_EXTRA_RESULT * max(
            0, num_results - 10
        )
        return SearchResponse(
            results=parse_results(data)[:num_results],
            cost_usd=cost,
            latency_ms=(time.perf_counter() - start) * 1000,
            request_id=data.get("search_id"),
            queries=payload["search_queries"],
        )

    async def extract(self, url: str, query: str | None = None) -> list[SearchResult]:
        payload: dict[str, Any] = {"urls": [url], "objective": clip(query or url)}
        if self.excerpt_max_chars:
            payload["advanced_settings"] = {
                "excerpt_settings": {"max_chars_per_result": self.excerpt_max_chars}
            }
        return parse_results(await self._post(PARALLEL_EXTRACT_URL, payload))

    async def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._client.post(url, json=payload)
        if not response.is_success:
            raise httpx.HTTPStatusError(
                f"Parallel API error {response.status_code}: {response.text[:500]}",
                request=response.request,
                response=response,
            )
        return response.json()

    async def close(self):
        await self._client.aclose()

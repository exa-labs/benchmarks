"""Perplexity Search API adapter.

Calls Perplexity's ``/search`` endpoint, which returns ranked results with
titles, URLs and page snippets. This is the raw retrieval API, not the Sonar
answer engine, so it composes with any synthesizer the same way the other
searchers do. ``search_type="fast"`` selects the low-latency backend. Perplexity
does not report cost, so requests are priced at the published rates.

API reference: https://docs.perplexity.ai/api-reference/search-post
"""

import asyncio
import os
import time
from typing import Any

import httpx

from .base import Searcher, SearchResponse, SearchResult

PERPLEXITY_SEARCH_URL = "https://api.perplexity.ai/search"
# https://docs.perplexity.ai/getting-started/pricing — $5 / 1k web, $1 / 1k fast.
PERPLEXITY_SEARCH_COST_PER_REQUEST = {"web": 0.005, "fast": 0.001}
_MAX_QUERY_CHARS = 8192
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class PerplexitySearcher(Searcher):
    name = "perplexity"

    def __init__(
        self,
        api_key: str | None = None,
        search_type: str = "web",
        max_attempts: int = 4,
        timeout: float = 60.0,
        **search_args: Any,
    ):
        if search_type not in PERPLEXITY_SEARCH_COST_PER_REQUEST:
            raise ValueError(f"search_type must be 'web' or 'fast', got {search_type!r}")
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY required")
        self.search_type = search_type
        self.search_args = search_args
        self.max_attempts = max_attempts
        self._client = httpx.AsyncClient(timeout=timeout)

    def search_payload(self, query: str, num_results: int) -> dict[str, Any]:
        """Build the ``/search`` request body; the default backend omits ``search_type``."""
        clipped = (
            query if len(query) <= _MAX_QUERY_CHARS else query[:_MAX_QUERY_CHARS].rsplit(" ", 1)[0]
        )
        payload: dict[str, Any] = {"query": clipped, "max_results": num_results, **self.search_args}
        if self.search_type != "web":
            payload["search_type"] = self.search_type
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
        del objective, search_queries
        start = time.perf_counter()
        payload = self.search_payload(query, num_results)
        data = await self._post(payload)
        results = [
            SearchResult(
                url=r.get("url") or "",
                title=r.get("title") or "",
                text=r.get("snippet") or "",
                metadata={
                    "rank": rank,
                    "published_date": r.get("date"),
                    "last_updated": r.get("last_updated"),
                },
            )
            for rank, r in enumerate(data.get("results") or [])
        ]
        return SearchResponse(
            results=results,
            cost_usd=PERPLEXITY_SEARCH_COST_PER_REQUEST[self.search_type],
            latency_ms=(time.perf_counter() - start) * 1000,
            request_id=data.get("id"),
            queries=[payload["query"]],
        )

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with exponential backoff on rate limits, server errors and timeouts."""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(self.max_attempts):
            try:
                response = await self._client.post(
                    PERPLEXITY_SEARCH_URL, headers=headers, json=payload
                )
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == self.max_attempts - 1:
                    raise
            else:
                if response.is_success:
                    return response.json()
                if (
                    response.status_code not in _RETRYABLE_STATUS
                    or attempt == self.max_attempts - 1
                ):
                    raise httpx.HTTPStatusError(
                        f"Perplexity API error {response.status_code}: {response.text[:500]}",
                        request=response.request,
                        response=response,
                    )
            await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")

    async def close(self):
        await self._client.aclose()

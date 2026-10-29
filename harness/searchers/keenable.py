"""Keenable Search adapter returning ranked snippets for retrieval and RAG.

``mode`` selects ``pro`` (deeper retrieval, the API default) or ``realtime``.
Calls the keyless endpoint, which needs an app title and no API key. Responses
omit usage, so each request is priced at the published pay-as-you-go rate.

API reference: https://docs.keenable.ai/api-reference/search
"""

import asyncio
import time
from typing import Any

import httpx

from .base import Searcher, SearchResponse, SearchResult

KEENABLE_SEARCH_URL = "https://api.keenable.ai/v1/search/public"
KEENABLE_APP_TITLE = "exa-benchmarks"
# https://keenable.ai/pricing — $4 / 1,000 requests. One search is one credit.
KEENABLE_COST_PER_CREDIT = 0.004
KEENABLE_MODES = ("pro", "realtime")
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504, 529})
_MAX_RESULTS = 50


class KeenableSearcher(Searcher):
    name = "keenable"

    def __init__(
        self,
        mode: str = "pro",
        max_attempts: int = 4,
        timeout: float = 60.0,
    ):
        if mode not in KEENABLE_MODES:
            raise ValueError(f"unsupported Keenable mode: {mode!r}")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.mode = mode
        self.max_attempts = max_attempts
        self._client = httpx.AsyncClient(timeout=timeout)

    def search_payload(self, query: str, num_results: int) -> dict[str, Any]:
        """Pin the search mode and stay inside the provider's result limit."""
        if num_results < 1:
            raise ValueError("num_results must be at least 1")
        return {
            "query": query,
            "mode": self.mode,
            "max_results": min(num_results, _MAX_RESULTS),
        }

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
        """Search once and preserve ranked snippets, dates and request cost."""
        del objective, search_queries
        start = time.perf_counter()
        payload = self.search_payload(query, num_results)
        data = await self._post(payload)
        results = [
            SearchResult(
                url=row.get("url") or "",
                title=row.get("title") or "",
                text=row.get("snippet") or "",
                metadata={
                    "rank": rank,
                    "published_at": row.get("published_at"),
                    "acquired_at": row.get("acquired_at"),
                },
            )
            for rank, row in enumerate(data.get("results") or [])
        ]
        usage = data.get("usage")
        if isinstance(usage, dict) and usage.get("credits") is not None:
            credits = usage["credits"]
        else:
            credits = 1
        return SearchResponse(
            results=results,
            cost_usd=float(credits) * KEENABLE_COST_PER_CREDIT,
            latency_ms=(time.perf_counter() - start) * 1000,
            request_id=data.get("request_id"),
            queries=[payload["query"]],
        )

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Retry transient provider failures, propagating permanent or exhausted errors."""
        headers = {"X-Keenable-Title": KEENABLE_APP_TITLE}
        for attempt in range(self.max_attempts):
            try:
                response = await self._client.post(
                    KEENABLE_SEARCH_URL, headers=headers, json=payload
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
                    response.raise_for_status()
            await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")

    async def close(self) -> None:
        await self._client.aclose()

"""Exa Search and Contents API adapter.

``search_type`` selects the retrieval mode (``instant``, ``fast``, ``auto``,
``neural``, ``keyword``); ``contents`` is sent verbatim as the request's
contents object when given (for example ``{"highlights": True}``), otherwise it
is built from the ``include_text`` / ``include_highlights`` switches. Cost is the
``costDollars.total`` the API reports, falling back to published list prices.

API reference: https://exa.ai/docs/reference/search
"""

import asyncio
import os
import time
from typing import Any

import httpx

from .base import Searcher, SearchResponse, SearchResult

# https://exa.ai/pricing — per request with up to 10 results, then per extra result.
EXA_SEARCH_COST_PER_REQUEST = 0.007
EXA_COST_PER_EXTRA_RESULT = 0.001
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class ExaSearcher(Searcher):
    name = "exa"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.exa.ai",
        include_text: bool = False,
        include_highlights: bool = True,
        category: str | None = None,
        search_type: str = "auto",
        max_characters: int | None = None,
        max_age_hours: int | None = None,
        livecrawl_timeout: int = 30000,
        extract_mode: str = "text",
        contents: dict[str, Any] | None = None,
        max_attempts: int = 5,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY required - get one at https://exa.ai")

        self.base_url = base_url
        self.include_text = include_text
        self.include_highlights = include_highlights
        self.category = category
        self.search_type = search_type
        self.max_characters = max_characters
        self.max_age_hours = max_age_hours
        self.livecrawl_timeout = livecrawl_timeout
        self.extract_mode = extract_mode
        self.contents = contents
        self.max_attempts = max_attempts
        self._client = httpx.AsyncClient(timeout=timeout)

    def search_payload(self, query: str, num_results: int) -> dict[str, Any]:
        """Build the ``/search`` request body."""
        payload: dict[str, Any] = {
            "query": query,
            "numResults": num_results,
            "type": self.search_type,
        }
        if self.category:
            payload["category"] = self.category

        if self.contents is not None:
            payload["contents"] = dict(self.contents)
        elif self.include_text or self.include_highlights:
            contents: dict[str, Any] = {}
            if self.include_text:
                contents["text"] = (
                    {"maxCharacters": self.max_characters} if self.max_characters else True
                )
            if self.include_highlights:
                highlights_config: dict[str, Any] = {"query": query}
                if self.max_characters:
                    highlights_config["maxCharacters"] = self.max_characters
                contents["highlights"] = highlights_config
            payload["contents"] = contents
        if self.max_age_hours is not None:
            payload.setdefault("contents", {})
            payload["contents"]["maxAgeHours"] = self.max_age_hours
            payload["contents"]["livecrawlTimeout"] = self.livecrawl_timeout
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
        data = await self._request("/search", self.search_payload(query, num_results))
        cost = _reported_cost(data)
        if cost is None:
            cost = EXA_SEARCH_COST_PER_REQUEST + EXA_COST_PER_EXTRA_RESULT * max(
                0, num_results - 10
            )
        return SearchResponse(
            results=_parse_results(data)[:num_results],
            cost_usd=cost,
            latency_ms=(time.perf_counter() - start) * 1000,
            request_id=data.get("requestId"),
            queries=[query],
        )

    async def extract(self, url: str, query: str | None = None) -> list[SearchResult]:
        payload: dict[str, Any] = {"urls": [url]}

        if self.extract_mode == "highlights":
            highlights_config: dict[str, Any] = {}
            if query:
                highlights_config["query"] = query
            if self.max_characters:
                highlights_config["maxCharacters"] = self.max_characters
            payload["highlights"] = highlights_config or True
        else:
            if self.max_characters:
                payload["text"] = {"maxCharacters": self.max_characters}
            else:
                payload["text"] = True

        if self.max_age_hours is not None:
            payload["maxAgeHours"] = self.max_age_hours
            payload["livecrawlTimeout"] = self.livecrawl_timeout

        return _parse_results(await self._request("/contents", payload))

    async def _request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with exponential backoff on rate limits, server errors and timeouts."""
        for attempt in range(self.max_attempts):
            try:
                response = await self._client.post(
                    f"{self.base_url}{endpoint}",
                    headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as error:
                retryable = error.response.status_code in _RETRYABLE_STATUS
                if not retryable or attempt == self.max_attempts - 1:
                    raise httpx.HTTPStatusError(
                        f"{error}; body={error.response.text[:500]}",
                        request=error.request,
                        response=error.response,
                    ) from error
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == self.max_attempts - 1:
                    raise
            await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")

    async def close(self):
        await self._client.aclose()


def _reported_cost(data: dict[str, Any]) -> float | None:
    """Return the request cost Exa reports, if any."""
    cost = data.get("costDollars")
    if isinstance(cost, dict) and isinstance(cost.get("total"), (int, float)):
        return float(cost["total"])
    return None


def _parse_results(data: dict[str, Any]) -> list[SearchResult]:
    """Map Exa result objects to ``SearchResult`` rows."""
    results = []
    for rank, r in enumerate(data.get("results", [])):
        highlights = r.get("highlights") or []
        if highlights and isinstance(highlights[0], dict):
            highlights = [h.get("text", "") for h in highlights]
        results.append(
            SearchResult(
                url=r.get("url", ""),
                title=r.get("title") or "",
                text=r.get("text") or "",
                highlights=highlights,
                metadata={
                    "rank": rank,
                    "score": r.get("score"),
                    "published_date": r.get("publishedDate"),
                    "author": r.get("author"),
                },
            )
        )
    return results

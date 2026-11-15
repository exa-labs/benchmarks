"""Kagi Search and Extract API adapter.

Search returns ranked web results with Kagi's ``<strong>`` highlight tags removed.
Kagi treats quotation marks as exact-phrase operators, so a quoted query with no web
results is retried once without them. ``extract_count`` replaces the top snippets with
page Markdown in the same request. Kagi does not report cost, so requests are priced
at the published rates.

API reference: https://kagi.com/api/docs/openapi/search
"""

import asyncio
import html
import json
import os
import re
import time
from typing import Any

import httpx

from .base import Searcher, SearchResponse, SearchResult

KAGI_API_URL = "https://kagi.com/api/v1"
# https://kagi.com/api/pricing — $12 / 1k searches, $4 / 1k extracted pages.
KAGI_SEARCH_COST_PER_REQUEST = 0.012
KAGI_EXTRACT_COST_PER_PAGE = 0.004
_MAX_EXTRACT_COUNT = 10
_MAX_RESULTS = 1024
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
# Kagi's spelling. An exhausted plan returns 429 with this code; retrying cannot help.
_USAGE_LIMIT_CODE = "general.usage_limt"
_HIGHLIGHT_TAGS = re.compile(r"</?strong>")
_QUOTES = re.compile(r'["“”„«»]')


def unquote_query(query: str) -> str:
    """Replace quotation marks with spaces so Kagi stops matching exact phrases."""
    return re.sub(r"\s+", " ", _QUOTES.sub(" ", query)).strip()


def clip(text: str, max_characters: int | None) -> str:
    return text[:max_characters] if max_characters else text


class KagiSearcher(Searcher):
    name = "kagi"

    def __init__(
        self,
        api_key: str | None = None,
        extract_count: int = 0,
        lens_id: str | None = None,
        max_characters: int | None = None,
        max_attempts: int = 4,
        timeout: float = 60.0,
    ):
        if not 0 <= extract_count <= _MAX_EXTRACT_COUNT:
            raise ValueError(f"extract_count must be between 0 and {_MAX_EXTRACT_COUNT}")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.api_key = api_key or os.getenv("KAGI_API_KEY")
        if not self.api_key:
            raise ValueError("KAGI_API_KEY required")
        self.extract_count = extract_count
        self.lens_id = lens_id
        self.max_characters = max_characters
        self.max_attempts = max_attempts
        self._client = httpx.AsyncClient(timeout=timeout)

    def search_payload(self, query: str, num_results: int) -> dict[str, Any]:
        """Stay inside Kagi's result limit and extract no more pages than results."""
        if num_results < 1:
            raise ValueError("num_results must be at least 1")
        payload: dict[str, Any] = {"query": query, "limit": min(num_results, _MAX_RESULTS)}
        if self.lens_id:
            payload["lens_id"] = self.lens_id
        if self.extract_count:
            payload["extract"] = {"count": min(self.extract_count, num_results)}
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
        """Search, retrying an empty quoted query once without quotes, and price both."""
        del objective, search_queries
        start = time.perf_counter()
        payload = self.search_payload(query, num_results)
        data = await self._post("/search", payload)
        requests = 1
        if not _web_results(data) and unquote_query(query) != query:
            payload["query"] = unquote_query(query)
            data = await self._post("/search", payload)
            requests = 2
        results = [
            SearchResult(
                url=row.get("url") or "",
                title=html.unescape(row.get("title") or ""),
                text=clip(_HIGHLIGHT_TAGS.sub("", row.get("snippet") or ""), self.max_characters),
                metadata={
                    "rank": rank,
                    "published_date": row.get("time"),
                    "group_id": (row.get("props") or {}).get("group_id"),
                },
            )
            for rank, row in enumerate(_web_results(data)[:num_results])
        ]
        return SearchResponse(
            results=results,
            cost_usd=requests * KAGI_SEARCH_COST_PER_REQUEST
            + min(self.extract_count, len(results)) * KAGI_EXTRACT_COST_PER_PAGE,
            latency_ms=(time.perf_counter() - start) * 1000,
            request_id=(data.get("meta") or {}).get("trace"),
            queries=[payload["query"]],
        )

    async def extract(self, url: str, query: str | None = None) -> list[SearchResult]:
        """Fetch one page as Markdown; Kagi Extract takes no question, so ``query`` is unused."""
        del query
        data = await self._post("/extract", {"pages": [{"url": url}]})
        return [
            SearchResult(
                url=page.get("url") or url,
                text=clip(page["markdown"], self.max_characters),
                metadata={"text_length": len(page["markdown"])},
            )
            for page in data.get("data") or []
            if page.get("markdown")
        ]

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Retry transient provider failures, propagating permanent or exhausted errors."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.max_attempts):
            try:
                response = await self._client.post(
                    f"{KAGI_API_URL}{path}", headers=headers, json=payload
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
                    or _USAGE_LIMIT_CODE in _error_codes(response)
                ):
                    raise httpx.HTTPStatusError(
                        f"Kagi API error {response.status_code}: {response.text[:500]}",
                        request=response.request,
                        response=response,
                    )
            await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")

    async def close(self) -> None:
        await self._client.aclose()


def _web_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ``data.search``, Kagi's ranked web results; other result groups are dropped."""
    return (data.get("data") or {}).get("search") or []


def _error_codes(response: httpx.Response) -> list[str]:
    """Return the codes in Kagi's ``{"errors": [{"code": ...}]}`` body.

    The OpenAPI spec names the list ``error``; the live API sends ``errors``. Gateway
    errors such as an HTML 502 page are not JSON and have no codes.
    """
    try:
        return [error["code"] for error in json.loads(response.text)["errors"]]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []

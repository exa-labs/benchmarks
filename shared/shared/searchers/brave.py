"""Brave Search API adapter (web search and LLM Context).

``search_type="llm_context"`` calls Brave's LLM Context endpoint, which returns
model-ready snippets grouped per URL; ``"web"`` calls the standard web search
endpoint. Queries are stripped of punctuation and clipped to Brave's length
limits; a ``422`` is retried once with a shorter query and ``429`` backs off.
Brave does not report cost, so each request is priced at the published rate.

API reference: https://api-dashboard.search.brave.com/app/documentation
"""

import asyncio
import json
import os
import re
import time
from typing import Any

import httpx

from .base import Searcher, SearchResponse, SearchResult

BRAVE_WEB_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_LLM_CONTEXT_URL = "https://api.search.brave.com/res/v1/llm/context"
# https://brave.com/search/api/ — $5 / 1k requests.
BRAVE_SEARCH_COST_PER_REQUEST = 0.005

_MAX_QUERY_CHARS = 400
_MAX_QUERY_WORDS = 50
_RETRY_QUERY_CHARS = 200
_RETRY_QUERY_WORDS = 25
_MAX_ATTEMPTS = 3
_AUTH_ERROR_CODES = frozenset(
    {
        "SUBSCRIPTION_TOKEN_INVALID",
        "SUBSCRIPTION_TOKEN_MISSING",
        "SUBSCRIPTION_TOKEN_EXPIRED",
        "ACCOUNT_SUSPENDED",
        "PLAN_LIMIT_REACHED",
    }
)


class BraveAuthError(RuntimeError):
    """Brave rejected the subscription token or account; retrying cannot help."""


def sanitize_query(query: str) -> str:
    """Replace punctuation with spaces; Brave rejects many symbols with a 422."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", query)).strip()


def truncate_query(query: str, max_chars: int, max_words: int) -> str:
    """Clip a query to Brave's character and word limits at a word boundary."""
    result = " ".join(query.split()[:max_words])
    if len(result) > max_chars:
        result = result[:max_chars].rsplit(" ", 1)[0]
    return result


def parse_web(data: dict[str, Any]) -> list[SearchResult]:
    """Map web-search hits to results; text is the description plus extra snippets."""
    results = []
    for hit in (data.get("web") or {}).get("results", []):
        if not isinstance(hit, dict) or "url" not in hit:
            continue
        snippets = [hit.get("description", ""), *hit.get("extra_snippets", [])]
        results.append(
            SearchResult(
                url=hit["url"],
                title=hit.get("title", ""),
                text="\n\n".join(s for s in snippets if s),
                metadata={
                    "rank": len(results),
                    "published_date": hit.get("page_age") or hit.get("age"),
                },
            )
        )
    return results


def parse_llm_context(data: dict[str, Any]) -> list[SearchResult]:
    """Map LLM Context grounding (generic, point-of-interest and map entries) to results."""
    grounding = data.get("grounding") or {}
    sources = data.get("sources") or {}
    results = []
    for entry in grounding.get("generic", []):
        url = entry.get("url", "")
        age = (sources.get(url) or {}).get("age")
        published = None
        if isinstance(age, list):
            published = next((age[i] for i in (3, 1, 0) if i < len(age) and age[i]), None)
        results.append(
            SearchResult(
                url=url,
                title=entry.get("title") or (sources.get(url) or {}).get("title", ""),
                text="\n\n".join(entry.get("snippets") or []),
                metadata={"rank": len(results), "published_date": published},
            )
        )
    extras = [grounding.get("poi")] + list(grounding.get("map") or [])
    for place in extras:
        if isinstance(place, dict) and place.get("url"):
            results.append(
                SearchResult(
                    url=place["url"],
                    title=place.get("title") or place.get("name", ""),
                    text="\n\n".join(place.get("snippets") or []),
                    metadata={"rank": len(results)},
                )
            )
    return results


class BraveSearcher(Searcher):
    name = "brave"

    def __init__(
        self,
        api_key: str | None = None,
        search_type: str = "web",
        site_filter: str | None = None,
        max_tokens_per_url: int = 4096,
        timeout: float = 60.0,
        **brave_args: Any,
    ):
        if search_type not in ("web", "llm_context"):
            raise ValueError(f"search_type must be 'web' or 'llm_context', got {search_type!r}")
        self.api_key = api_key or os.getenv("BRAVE_SEARCH_API_KEY") or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("Brave API key required - set BRAVE_SEARCH_API_KEY or pass api_key")

        self.search_type = search_type
        self.site_filter = site_filter
        self.max_tokens_per_url = max_tokens_per_url
        self.brave_args = brave_args
        self._client = httpx.AsyncClient(timeout=timeout)

    @property
    def endpoint(self) -> str:
        return BRAVE_LLM_CONTEXT_URL if self.search_type == "llm_context" else BRAVE_WEB_URL

    def search_params(self, query: str, num_results: int) -> dict[str, Any]:
        """Build the query-string parameters for one request."""
        search_query = sanitize_query(query)
        if self.site_filter:
            search_query = f"site:{self.site_filter} {search_query}"
        params: dict[str, Any] = {
            "q": truncate_query(search_query, _MAX_QUERY_CHARS, _MAX_QUERY_WORDS),
            "count": num_results,
        }
        if self.search_type == "llm_context":
            params["maximum_number_of_tokens_per_url"] = self.max_tokens_per_url
        params.update(
            {k: str(v).lower() if isinstance(v, bool) else v for k, v in self.brave_args.items()}
        )
        return params

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
        params = self.search_params(query, num_results)
        data = await self._request_with_retry(params)
        parse = parse_llm_context if self.search_type == "llm_context" else parse_web
        return SearchResponse(
            results=parse(data),
            cost_usd=BRAVE_SEARCH_COST_PER_REQUEST,
            latency_ms=(time.perf_counter() - start) * 1000,
            queries=[params["q"]],
        )

    async def _request_with_retry(self, params: dict[str, Any]) -> dict[str, Any]:
        """GET with backoff on 429 and one shortened-query retry on 422."""
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        truncated = False
        for attempt in range(_MAX_ATTEMPTS):
            response = await self._client.get(self.endpoint, headers=headers, params=params)
            if response.is_success:
                return response.json()
            code = _error_code(response)
            if code in _AUTH_ERROR_CODES:
                raise BraveAuthError(f"Brave API auth error ({code}): {response.text[:300]}")
            if attempt < _MAX_ATTEMPTS - 1:
                if response.status_code == 429:
                    await asyncio.sleep(2**attempt)
                    continue
                if response.status_code == 422 and not truncated:
                    params = {
                        **params,
                        "q": truncate_query(params["q"], _RETRY_QUERY_CHARS, _RETRY_QUERY_WORDS),
                    }
                    truncated = True
                    continue
            raise httpx.HTTPStatusError(
                f"Brave API error {response.status_code}: {response.text[:500]}",
                request=response.request,
                response=response,
            )
        raise AssertionError("unreachable")

    async def close(self):
        await self._client.aclose()


def _error_code(response: httpx.Response) -> str | None:
    """Return Brave's structured error code from an error body, if present."""
    try:
        return (json.loads(response.text).get("error") or {}).get("code")
    except (json.JSONDecodeError, AttributeError):
        return None

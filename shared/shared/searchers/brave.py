import os
from typing import Any

import httpx

from .base import Searcher, SearchResult


class BraveSearcher(Searcher):
    name = "brave"

    def __init__(
        self,
        api_key: str | None = None,
        search_type: str = "web",
        site_filter: str | None = None,
        **brave_args: Any,
    ):
        self.api_key = api_key or os.getenv("BRAVE_SEARCH_API_KEY") or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("Brave API key required - set BRAVE_SEARCH_API_KEY or pass api_key")

        self.search_type = search_type
        self.site_filter = site_filter
        self.brave_args = brave_args
        self._client = httpx.AsyncClient(timeout=60.0)

    def _base_url(self) -> str:
        if self.search_type == "llm_context":
            return "https://api.search.brave.com/res/v1/llm/context"
        return "https://api.search.brave.com/res/v1/web/search"

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        search_query = query
        if self.site_filter:
            search_query = f"site:{self.site_filter} {search_query}"

        if len(search_query) > 400:
            search_query = search_query[:400]
        elif len(search_query.split()) > 50:
            search_query = " ".join(search_query.split()[:50])

        params: dict[str, Any] = {
            "q": search_query,
            "count": num_results,
            **self.brave_args,
        }

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }

        response = await self._client.get(
            self._base_url(),
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        if self.search_type == "llm_context":
            return self._parse_llm_context(data)
        return self._parse_web(data)

    def _parse_web(self, data: dict) -> list[SearchResult]:
        results = []
        for i, hit in enumerate(data.get("web", {}).get("results", [])):
            if not isinstance(hit, dict) or "url" not in hit:
                continue
            pub_date = hit.get("page_age") or hit.get("age")
            snippets = hit.get("extra_snippets", [])
            text = hit.get("description", "")
            if snippets:
                text = text + "\n" + "\n".join(snippets)

            results.append(
                SearchResult(
                    url=hit["url"],
                    title=hit.get("title", ""),
                    text=text,
                    metadata={"rank": i, "published_date": pub_date},
                )
            )
        return results

    def _parse_llm_context(self, data: dict) -> list[SearchResult]:
        results = []
        for i, item in enumerate(data.get("results", data.get("web", {}).get("results", []))):
            if not isinstance(item, dict) or "url" not in item:
                continue
            snippets = item.get("snippets", [])
            text = "\n".join(snippets) if snippets else item.get("description", "")
            results.append(
                SearchResult(
                    url=item["url"],
                    title=item.get("title", ""),
                    text=text,
                    metadata={"rank": i},
                )
            )
        return results

    async def close(self):
        await self._client.aclose()

import os
from typing import Any

import httpx

from benchmarks.shared.searchers import SearchResult, Searcher


class BraveSearcher(Searcher):
    name = "brave"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.search.brave.com/res/v1/web/search",
        site_filter: str | None = None,
        **brave_args: Any,
    ):
        self.api_key = api_key or os.getenv("BRAVE_SEARCH_API_KEY") or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("Brave API key required - set BRAVE_SEARCH_API_KEY or pass api_key")

        self.base_url = base_url
        self.site_filter = site_filter
        self.brave_args = brave_args
        self._client = httpx.AsyncClient(timeout=60.0)

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        search_query = query
        if self.site_filter:
            search_query = f"site:{self.site_filter} {search_query}"

        # Brave API limits: 400 chars, 50 words
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
            self.base_url,
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        web_results = data.get("web", {}).get("results", [])

        for i, hit in enumerate(web_results):
            if not isinstance(hit, dict) or "url" not in hit:
                continue

            # Handle date fields
            pub_date = hit.get("page_age") or hit.get("age")

            results.append(
                SearchResult(
                    url=hit["url"],
                    title=hit.get("title", ""),
                    text=hit.get("description", ""),
                    metadata={
                        "rank": i,
                        "published_date": pub_date,
                    },
                )
            )

        return results

    async def close(self):
        await self._client.aclose()

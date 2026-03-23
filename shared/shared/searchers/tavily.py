import os
from typing import Any

import httpx

from .base import Searcher, SearchResult


class TavilySearcher(Searcher):
    name = "tavily"

    def __init__(
        self,
        api_key: str | None = None,
        search_depth: str = "advanced",
        chunks_per_source: int | None = None,
    ):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY required")

        self.search_depth = search_depth
        self.chunks_per_source = chunks_per_source
        self._client = httpx.AsyncClient(timeout=60.0)

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        payload: dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "max_results": num_results,
            "search_depth": self.search_depth,
            "include_raw_content": False,
        }

        response = await self._client.post(
            "https://api.tavily.com/search",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for i, r in enumerate(data.get("results", [])):
            results.append(
                SearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    text=r.get("content", ""),
                    metadata={"rank": i, "score": r.get("score")},
                )
            )
        return results

    async def extract(self, url: str, query: str | None = None) -> list[SearchResult]:
        payload: dict[str, Any] = {
            "api_key": self.api_key,
            "urls": [url],
        }
        if query:
            payload["query"] = query
        if self.chunks_per_source:
            payload["chunks_per_source"] = self.chunks_per_source

        response = await self._client.post(
            "https://api.tavily.com/extract",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for r in data.get("results", []):
            raw_content = r.get("raw_content", "")
            chunks = r.get("chunks", [])
            text = raw_content if raw_content else "\n\n".join(chunks) if chunks else ""

            results.append(
                SearchResult(
                    url=r.get("url", url),
                    title="",
                    text=text,
                    highlights=chunks,
                    metadata={},
                )
            )
        return results

    async def close(self):
        await self._client.aclose()

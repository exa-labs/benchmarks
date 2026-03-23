import os
from typing import Any

import httpx

from .base import Searcher, SearchResult


class ParallelSearcher(Searcher):
    name = "parallel"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.parallel.ai/v1beta/search",
        processor: str = "base",
        source_policy: dict | None = None,
        excerpts: bool = True,
        excerpt_max_chars: int | None = None,
    ):
        self.api_key = api_key or os.getenv("PARALLEL_API_KEY") or os.getenv("PARALLELS_API_KEY")
        if not self.api_key:
            raise ValueError("Parallel API key required - set PARALLEL_API_KEY or pass api_key")

        self.base_url = base_url
        self.processor = processor
        self.source_policy = source_policy
        self.excerpts = excerpts
        self.excerpt_max_chars = excerpt_max_chars
        self._client = httpx.AsyncClient(timeout=60.0)

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        payload: dict[str, Any] = {
            "max_results": num_results,
            "processor": self.processor,
            "objective": query,
        }
        if self.source_policy:
            payload["source_policy"] = self.source_policy
        if self.excerpts:
            excerpts_config: dict[str, Any] = {}
            if self.excerpt_max_chars:
                excerpts_config["max_chars_per_result"] = self.excerpt_max_chars
            payload["excerpts"] = excerpts_config if excerpts_config else True

        return await self._do_request(payload)

    async def extract(self, url: str, query: str | None = None) -> list[SearchResult]:
        payload: dict[str, Any] = {
            "max_results": 1,
            "processor": self.processor,
            "objective": query or url,
            "source_policy": {"urls": [url]},
        }
        if self.excerpts:
            excerpts_config: dict[str, Any] = {}
            if self.excerpt_max_chars:
                excerpts_config["max_chars_per_result"] = self.excerpt_max_chars
            payload["excerpts"] = excerpts_config if excerpts_config else True

        return await self._do_request(payload)

    async def _do_request(self, payload: dict[str, Any]) -> list[SearchResult]:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "parallel-beta": "search-extract-2025-10-10",
        }

        response = await self._client.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        results = []
        for i, result in enumerate(data.get("results", [])):
            excerpts = result.get("excerpts", [])
            text = " ".join(excerpts) if isinstance(excerpts, list) else str(excerpts)

            results.append(
                SearchResult(
                    url=result.get("url", ""),
                    title=result.get("title", ""),
                    text=text,
                    metadata={
                        "rank": i,
                        "author": result.get("author"),
                        "published_date": result.get("published_date")
                        or result.get("publishedDate"),
                    },
                )
            )

        return results

    async def close(self):
        await self._client.aclose()

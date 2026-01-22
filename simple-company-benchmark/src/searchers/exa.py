"""Exa search implementation for company search benchmark."""

import asyncio
import os
from typing import Any

import httpx

from benchmarks.shared.searchers import SearchResult, Searcher


class ExaSearcher(Searcher):
    name = "exa"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.exa.ai",
        include_text: bool = True,
        category: str | None = None,
        search_type: str = "auto",
    ):
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY required - get one at https://exa.ai")

        self.base_url = base_url
        self.include_text = include_text
        self.category = category
        self.search_type = search_type
        self._client = httpx.AsyncClient(timeout=120.0)

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        payload: dict[str, Any] = {
            "query": query,
            "numResults": num_results,
            "type": self.search_type,
        }

        if self.category:
            payload["category"] = self.category

        if self.include_text:
            payload["contents"] = {"text": True}

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = await self._client.post(
                    f"{self.base_url}/search",
                    headers={
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise

        results = []
        for r in data.get("results", []):
            results.append(
                SearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    text=r.get("text", ""),
                    metadata={
                        "score": r.get("score"),
                        "published_date": r.get("publishedDate"),
                        "author": r.get("author"),
                    },
                )
            )

        return results

    async def close(self):
        await self._client.aclose()

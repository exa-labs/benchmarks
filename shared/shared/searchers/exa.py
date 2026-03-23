import asyncio
import os
from typing import Any

import httpx

from .base import Searcher, SearchResult


class ExaSearcher(Searcher):
    name = "exa"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.exa.ai",
        include_text: bool = True,
        category: str | None = None,
        search_type: str = "auto",
        max_characters: int | None = None,
        max_age_hours: int | None = None,
        livecrawl_timeout: int = 30000,
        extract_mode: str = "text",
    ):
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY required - get one at https://exa.ai")

        self.base_url = base_url
        self.include_text = include_text
        self.category = category
        self.search_type = search_type
        self.max_characters = max_characters
        self.max_age_hours = max_age_hours
        self.livecrawl_timeout = livecrawl_timeout
        self.extract_mode = extract_mode
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
            contents: dict[str, Any] = {"text": True}
            if self.max_characters:
                contents["text"] = {"maxCharacters": self.max_characters}
            if self.max_age_hours is not None:
                contents["maxAgeHours"] = self.max_age_hours
                contents["livecrawlTimeout"] = self.livecrawl_timeout
            payload["contents"] = contents

        return await self._request("/search", payload)

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

        return await self._request("/contents", payload)

    async def _request(self, endpoint: str, payload: dict[str, Any]) -> list[SearchResult]:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = await self._client.post(
                    f"{self.base_url}{endpoint}",
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
                    await asyncio.sleep(2**attempt)
                    continue
                raise
            except (httpx.ReadTimeout, httpx.ConnectTimeout):
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise

        results = []
        for r in data.get("results", []):
            highlights = r.get("highlights", [])
            if isinstance(highlights, list) and highlights and isinstance(highlights[0], dict):
                highlights = [h.get("text", "") for h in highlights]

            results.append(
                SearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    text=r.get("text", ""),
                    highlights=highlights,
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

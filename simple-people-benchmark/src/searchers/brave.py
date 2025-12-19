import asyncio
import logging
import os
from typing import Any

import httpx

from .base import SearchResult, Searcher

logger = logging.getLogger(__name__)


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
            raise ValueError(
                "Brave API key required - set BRAVE_SEARCH_API_KEY or pass api_key"
            )

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

        # Retry with exponential backoff, respecting Retry-After headers
        max_retries = 10
        last_exception = None
        response = None
        
        for attempt in range(max_retries):
            try:
                response = await self._client.get(
                    self.base_url,
                    headers=headers,
                    params=params,
                )
                
                if response.status_code == 429:
                    # Check for Retry-After header
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = int(retry_after)
                        except ValueError:
                            wait_time = min(2 ** attempt, 120)
                    else:
                        # Exponential backoff: 2^attempt seconds, max 120 seconds
                        wait_time = min(2 ** attempt, 120)
                    
                    if attempt < max_retries - 1:
                        logger.debug(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()
                
                response.raise_for_status()
                break
                
            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    retry_after = e.response.headers.get("Retry-After") if e.response else None
                    if retry_after:
                        try:
                            wait_time = int(retry_after)
                        except ValueError:
                            wait_time = min(2 ** attempt, 120)
                    else:
                        wait_time = min(2 ** attempt, 120)
                    logger.debug(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    await asyncio.sleep(wait_time)
                    continue
                # For other HTTP errors or final retry, raise
                if attempt == max_retries - 1:
                    logger.warning(f"Brave search failed after {max_retries} retries: {e}")
                    raise
                raise
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 10)
                    logger.debug(f"Error, waiting {wait_time}s before retry {attempt + 1}/{max_retries}: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                raise
        
        if response is None:
            if last_exception:
                raise last_exception
            raise httpx.HTTPStatusError("No response received", request=None, response=None)
        
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


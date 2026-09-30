"""Searcher interface shared by every benchmark and by the harness.

A ``Searcher`` wraps one provider endpoint. ``search`` returns ranked
``SearchResult`` rows; ``run`` returns the same rows plus the request's cost and
latency, which the harness records per call. Searchers that bill per request
override ``run``; older adapters that only implement ``search`` inherit a
``run`` that reports the cost as unknown.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    url: str = ""
    title: str = ""
    text: str = ""
    highlights: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        """Result body, falling back to highlights when full text isn't present."""
        return self.text or "\n".join(self.highlights)

    @property
    def evidence(self) -> str:
        """Query-relevant evidence for a model: highlights first, then full text."""
        return "\n".join(h for h in self.highlights if h) or self.text


@dataclass
class SearchResponse:
    """One provider request: ranked results, USD cost (``None`` if unknown) and latency."""

    results: list[SearchResult]
    cost_usd: float | None = None
    latency_ms: float = 0.0
    request_id: str | None = None
    queries: list[str] = field(default_factory=list)


class Searcher(ABC):
    name: str = "base"

    # Whether ``run`` sends several ``search_queries`` in one provider request.
    batches_queries: bool = False

    @abstractmethod
    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        pass

    async def run(
        self,
        query: str,
        num_results: int = 10,
        *,
        objective: str | None = None,
        search_queries: list[str] | None = None,
    ) -> SearchResponse:
        """Execute one search request and report its cost and latency.

        ``objective`` and ``search_queries`` are honored by providers that accept
        a natural-language goal plus keyword queries in one request
        (``batches_queries``); other searchers ignore them.
        """
        del objective, search_queries
        start = time.perf_counter()
        results = await self.search(query, num_results)
        return SearchResponse(
            results=results,
            cost_usd=None,
            latency_ms=(time.perf_counter() - start) * 1000,
            queries=[query],
        )

    async def extract(self, url: str, query: str | None = None) -> list[SearchResult]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support URL extraction")

    async def close(self) -> None:
        """Release network resources. Safe to call more than once."""

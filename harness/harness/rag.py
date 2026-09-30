"""Single-step RAG: one search, then one tools-free synthesis over its results.

This is the retrieval-only baseline. The synthesizer is instructed to use only
the provided results, so the score reflects what one search request returned.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from shared.searchers import Searcher, SearchResponse, SearchResult

from harness.llm.clients import ModelClient, create_client
from harness.tools import normalize_results

RAG_SYSTEM_PROMPT = """You are a tabula rasa answer-synthesis engine. You have NO internal knowledge
about the world — no facts, no opinions, no training data to draw from. The ONLY
information that exists for you is the search results provided below.

Today's date: {current_date}

Rules (absolute, no exceptions):
1. You MUST answer using ONLY information explicitly present in the search results.
2. You MUST NOT use any knowledge from your training data, even if you "know" the answer.
3. If the search results do not contain enough information to answer the question,
   reply exactly: "I don't know"
4. Do NOT hedge with "based on my knowledge" or similar — you have no knowledge.
5. Match the requested answer format, including complete lists and tables when requested.
6. Prefer information from earlier (higher-ranked) results — they are more relevant.
7. If you cite a fact, it must be traceable to a specific search result.
8. Use today's date to resolve relative time references (e.g. "yesterday", "last week", "3 days ago")."""

RAG_USER_PROMPT = """Question: {question}

Search results:
{search_results}

Answer the question using ONLY the search results above. If they don't contain the answer, say "I don't know"."""


@dataclass
class RAGResult:
    """One single-step answer with its evidence and accounting."""

    answer: str
    citations: list[dict[str, Any]]
    search_queries: list[str]
    model_cost_usd: float
    search_cost_usd: float
    cost_known: bool
    usage: dict[str, int]
    latency_ms: float
    messages: list[dict[str, Any]]
    skipped_results: list[dict[str, str]]

    @property
    def total_cost_usd(self) -> float:
        return self.model_cost_usd + self.search_cost_usd

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "total_cost_usd": self.total_cost_usd}


async def enrich_results(results: list[SearchResult], extractor: Searcher) -> None:
    """Add full page text while preserving ranked URLs and provider metadata."""
    for result in results:
        fetched = await extractor.extract(result.url)
        if fetched:
            result.text = fetched[0].content or result.content
            result.highlights = []


class SingleStepRAG:
    """Retrieve once with the question as the query, then synthesize."""

    def __init__(
        self,
        searcher: Searcher,
        model: str,
        *,
        num_results: int = 10,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        client: ModelClient | None = None,
        enrichment: Searcher | None = None,
    ) -> None:
        self.searcher = searcher
        self.num_results = num_results
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.client = client or create_client(model)
        self.enrichment = enrichment

    async def run(self, question: str, *, url: str | None = None) -> RAGResult:
        start = time.monotonic()
        if url is None:
            response = await self.searcher.run(question, self.num_results)
        else:
            response = SearchResponse(
                results=await self.searcher.extract(url, query=question), queries=[url]
            )
        if self.enrichment is not None:
            await enrich_results(response.results, self.enrichment)
        kept, skipped = normalize_results(response.results)
        citations = [
            {
                "url": r["url"],
                "title": r["title"],
                "text": r["text"],
                "published_date": r["published_date"],
            }
            for r in kept
        ]
        current_date = datetime.now(timezone.utc).strftime("%B %d, %Y")
        messages = [
            {
                "role": "system",
                "content": RAG_SYSTEM_PROMPT.replace("{current_date}", current_date),
            },
            {
                "role": "user",
                "content": RAG_USER_PROMPT.format(
                    question=question,
                    search_results="\n".join(json.dumps(c, ensure_ascii=False) for c in citations),
                ),
            },
        ]
        generation = await self.client.generate(
            messages,
            max_output_tokens=self.max_output_tokens,
            reasoning_effort=self.reasoning_effort,
        )
        answer = generation.text.strip() or "I don't know"
        messages.append({"role": "assistant", "content": answer})
        return RAGResult(
            answer=answer,
            citations=citations,
            search_queries=response.queries,
            model_cost_usd=generation.cost_usd or 0.0,
            search_cost_usd=response.cost_usd or 0.0,
            cost_known=generation.cost_usd is not None
            and response.cost_usd is not None
            and self.enrichment is None,
            usage=generation.usage.to_dict(),
            latency_ms=(time.monotonic() - start) * 1000,
            messages=messages,
            skipped_results=skipped,
        )

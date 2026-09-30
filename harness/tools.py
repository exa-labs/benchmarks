"""Search tools the Scout loop exposes to the model.

A ``SearchTool`` binds one model-facing function schema to one ``Searcher``.
Every provider gets the same ``search({query})`` definition except Parallel,
whose API takes a natural-language objective plus exactly three keyword queries
in one request; that tool forwards both unchanged. Tool output is a numbered
plain-text list of title, URL and evidence (highlights first, then text).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from harness.searchers import Searcher, SearchResult

_SEARCH_DESCRIPTION = (
    "Searches the web for current and factual information, returning relevant results "
    "with titles, URLs, and content snippets."
)

QUERY_SEARCH_TOOL: dict[str, Any] = {
    "name": "search",
    "description": _SEARCH_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A concise, self-contained web search query.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}

# Matches Parallel's published tool-calling definition for its Search API.
OBJECTIVE_SEARCH_TOOL: dict[str, Any] = {
    "name": "search",
    "description": _SEARCH_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "objective": {
                "type": "string",
                "description": (
                    "A concise, self-contained search query. Must include the key entity "
                    "or topic being searched for."
                ),
            },
            "search_queries": {
                "type": "array",
                "description": (
                    "Exactly 3 keyword search queries, each 3-6 words. Must be diverse — vary "
                    "entity names, synonyms, and angles. Each query must include the key entity "
                    "or topic. NEVER write sentences, instructions, or use site: operators."
                ),
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 3,
            },
        },
        "required": ["objective", "search_queries"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass
class ToolOutcome:
    """The result of one tool call as the loop accounts for it."""

    content: str
    error: str | None = None
    final_answer: str | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    num_searches: int = 0
    queries_used: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True
    latency_ms: float = 0.0

    @property
    def done(self) -> bool:
        return self.final_answer is not None


def normalize_results(
    results: list[SearchResult],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Keep results that have a URL and evidence text; report the rest as skipped."""
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for rank, result in enumerate(results):
        text = result.evidence
        if not result.url:
            skipped.append({"reason": "missing_url", "title": result.title})
        elif not text:
            skipped.append({"reason": "empty_text", "url": result.url})
        else:
            kept.append(
                {
                    "url": result.url,
                    "title": result.title,
                    "text": text,
                    "published_date": result.metadata.get("published_date"),
                    "rank": rank,
                }
            )
    return kept, skipped


def format_records(records: list[dict[str, Any]]) -> str:
    """Render search records as the numbered plain-text list the model reads."""
    sections = []
    for record in records:
        lines = [f"Search results for '{record['query']}':", ""]
        for index, result in enumerate(record["results"], start=1):
            lines += [
                f"{index}. {result['title']}",
                f"   URL: {result['url']}",
                f"   {result['text']}",
                "",
            ]
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


class SearchTool:
    """One model-facing search function backed by a ``Searcher``."""

    def __init__(self, searcher: Searcher, *, num_results: int = 10, name: str = "search") -> None:
        self.searcher = searcher
        self.num_results = num_results
        schema = OBJECTIVE_SEARCH_TOOL if searcher.batches_queries else QUERY_SEARCH_TOOL
        self.schema = {**copy.deepcopy(schema), "name": name}
        self.name = name
        items = self.schema["parameters"]["properties"].get("search_queries", {})
        self.max_queries_per_call: int | None = items.get("maxItems")

    def requested_queries(self, arguments: dict[str, Any]) -> tuple[str, list[str]]:
        """Return the call's objective and non-empty queries, capped per call."""
        objective = arguments.get("objective")
        objective = objective.strip() if isinstance(objective, str) else ""
        raw = arguments.get("search_queries")
        queries = (
            [q.strip() for q in raw if isinstance(q, str) and q.strip()]
            if isinstance(raw, list)
            else []
        )
        if not queries and isinstance(arguments.get("query"), str) and arguments["query"].strip():
            queries = [arguments["query"].strip()]
        if not queries and objective:
            queries = [objective]
        if self.max_queries_per_call is not None:
            queries = queries[: self.max_queries_per_call]
        return objective, queries

    async def __call__(
        self, arguments: dict[str, Any], *, max_queries: int | None = None
    ) -> ToolOutcome:
        """Run the call's searches; ``max_queries`` is the trajectory's remaining allowance."""
        objective, queries = self.requested_queries(arguments)
        if max_queries is not None:
            queries = queries[:max_queries]
        if not queries:
            exhausted = max_queries == 0
            message = "Trajectory search limit exceeded" if exhausted else "Empty search query"
            return ToolOutcome(
                content=f"{'Skipped' if exhausted else 'Error'}: {message}", error=message
            )

        if self.searcher.batches_queries:
            responses = [
                await self.searcher.run(
                    objective or queries[0],
                    self.num_results,
                    objective=objective or None,
                    search_queries=queries,
                )
            ]
            labels = [objective or queries[0]]
        else:
            responses = [await self.searcher.run(q, self.num_results) for q in queries]
            labels = queries

        records = []
        for label, response in zip(labels, responses, strict=True):
            kept, skipped = normalize_results(response.results)
            record: dict[str, Any] = {
                "tool": self.name,
                "query": label,
                "search_queries": response.queries or [label],
                "results": kept,
                "num_results": len(response.results),
                "cost_usd": response.cost_usd,
                "latency_ms": response.latency_ms,
                "request_id": response.request_id,
            }
            if objective:
                record["objective"] = objective
            if skipped:
                record["skipped_results"] = skipped
            records.append(record)

        return ToolOutcome(
            content=format_records(records),
            records=records,
            evidence=[result for record in records for result in record["results"]],
            num_searches=len(records),
            queries_used=len(queries),
            cost_usd=sum(r.cost_usd or 0.0 for r in responses),
            cost_known=all(r.cost_usd is not None for r in responses),
            latency_ms=sum(r.latency_ms for r in responses),
        )

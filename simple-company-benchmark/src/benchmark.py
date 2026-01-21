"""
Company Search Benchmark runner.

Supports two evaluation tracks:
- Retrieval: Return ranked lists of companies matching criteria
- RAG: Extract specific facts about companies

Header author: Devin
"""

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table

from .graders import RAGGrader, RetrievalGrader
from .metrics import RAGMetrics, RetrievalMetrics, compute_rag_metrics, compute_retrieval_metrics
from .searchers import SearchResult, Searcher

console = Console()
logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class Query:
    query_id: str
    text: str
    track: str = ""
    bucket: str = ""
    split: str = ""
    metadata: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)
    gold_company_homepage: str | None = None
    constraints: dict | None = None
    expected_answer: str | None = None
    homepage: str | None = None


@dataclass
class BenchmarkConfig:
    limit: int | None = None
    num_results: int = 10
    output_file: str | None = None
    enrich_exa_contents: bool = False
    track: str | None = None
    split: str | None = None


def load_queries(
    track: str | None = None,
    split: str | None = None,
    limit: int | None = None,
) -> list[Query]:
    """Load queries from data files.

    Args:
        track: Filter by track ('retrieval' or 'rag')
        split: Filter by split ('static' or 'dynamic')
        limit: Maximum number of queries to load
    """
    queries = []

    tracks = [track] if track else ["retrieval", "rag"]
    splits = [split] if split else ["static", "dynamic"]

    for t in tracks:
        for s in splits:
            filepath = DATA_DIR / "company" / t / f"{s}.jsonl"
            if not filepath.exists():
                continue

            with open(filepath) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    queries.append(
                        Query(
                            query_id=data.get("query_id", ""),
                            text=data.get("text", ""),
                            track=data.get("track", t),
                            bucket=data.get("bucket", ""),
                            split=data.get("split", s),
                            metadata=data.get("metadata", {}),
                            tags=data.get("tags", []),
                            gold_company_homepage=data.get("gold_company_homepage"),
                            constraints=data.get("constraints"),
                            expected_answer=data.get("expected_answer"),
                            homepage=data.get("homepage"),
                        )
                    )

    return queries[:limit] if limit else queries


async def fetch_exa_contents(urls: list[str], api_key: str | None = None) -> dict[str, str]:
    """Fetch page contents via Exa API."""
    api_key = api_key or os.getenv("EXA_API_KEY")
    if not api_key or not urls:
        return {}

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.exa.ai/contents",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={"urls": urls, "text": True, "livecrawl": "fallback"},
        )
        resp.raise_for_status()
        return {
            r["url"]: r["text"]
            for r in resp.json().get("results", [])
            if r.get("url") and r.get("text")
        }


async def enrich_results(results: list[SearchResult]) -> list[SearchResult]:
    """Enrich search results with full page contents."""
    try:
        contents = await fetch_exa_contents([r.url for r in results if r.url])
    except Exception as e:
        logger.warning(f"Content fetch failed: {e}")
        return results
    return [
        SearchResult(r.url, r.title, contents.get(r.url, r.text), r.metadata) for r in results
    ]


class Benchmark:
    """Company search benchmark runner."""

    def __init__(self, searchers: list[Searcher], grading_concurrency: int = 50):
        self.searchers = searchers
        self.retrieval_grader = RetrievalGrader()
        self.rag_grader = RAGGrader()
        self._grade_semaphore = asyncio.Semaphore(grading_concurrency)

    async def _grade_retrieval(
        self, query: Query, results: list[SearchResult]
    ) -> list[dict]:
        """Grade retrieval results."""

        async def grade_one(rank: int, r: SearchResult) -> dict:
            async with self._grade_semaphore:
                g = await self.retrieval_grader.grade(
                    query.text,
                    r,
                    gold_homepage=query.gold_company_homepage,
                    constraints=query.constraints,
                )
            return {
                "query_id": query.query_id,
                "rank": rank,
                "is_match": g.scores.get("is_match", 0),
            }

        return await asyncio.gather(*[grade_one(i, r) for i, r in enumerate(results, 1)])

    async def _grade_rag(
        self, query: Query, answer: str
    ) -> dict:
        """Grade RAG answer."""
        async with self._grade_semaphore:
            g = await self.rag_grader.grade(
                query.text,
                query.expected_answer or "",
                answer,
                bucket=query.bucket,
            )
        return {
            "query_id": query.query_id,
            "is_correct": g.scores.get("is_correct", 0),
        }

    async def _run_retrieval(
        self,
        searcher: Searcher,
        queries: list[Query],
        config: BenchmarkConfig,
        progress: Progress,
        task_id: TaskID,
    ) -> list[dict]:
        """Run retrieval track evaluation."""
        grades = []
        semaphore = asyncio.Semaphore(20)

        async def process(q: Query):
            async with semaphore:
                results = await searcher.search(q.text, config.num_results)
                if config.enrich_exa_contents:
                    results = await enrich_results(results)
                grades.extend(await self._grade_retrieval(q, results))
                progress.advance(task_id)

        await asyncio.gather(*[process(q) for q in queries])
        return grades

    async def _run_rag(
        self,
        searcher: Searcher,
        queries: list[Query],
        config: BenchmarkConfig,
        progress: Progress,
        task_id: TaskID,
    ) -> list[dict]:
        """Run RAG track evaluation.

        For RAG, we search and then extract the answer from results.
        """
        grades = []
        semaphore = asyncio.Semaphore(20)

        async def process(q: Query):
            async with semaphore:
                results = await searcher.search(q.text, config.num_results)
                if config.enrich_exa_contents:
                    results = await enrich_results(results)

                combined_text = "\n\n".join(
                    f"[{r.title}]\n{r.text}" for r in results if r.text
                )

                answer = await self._extract_answer(q.text, combined_text)
                grade = await self._grade_rag(q, answer)
                grades.append(grade)
                progress.advance(task_id)

        await asyncio.gather(*[process(q) for q in queries])
        return grades

    async def _extract_answer(self, query: str, context: str) -> str:
        """Extract answer from search results using LLM."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        try:
            response = await client.chat.completions.create(
                model="gpt-4.1",
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract the answer to the question from the provided context. "
                        "Give a concise, direct answer. If the answer is not found, say 'unknown'.",
                    },
                    {
                        "role": "user",
                        "content": f"Question: {query}\n\nContext:\n{context[:8000]}",
                    },
                ],
            )
            return response.choices[0].message.content or "unknown"
        except Exception as e:
            logger.warning(f"Answer extraction failed: {e}")
            return "unknown"

    async def run(self, config: BenchmarkConfig | None = None) -> dict[str, Any]:
        """Run the benchmark."""
        config = config or BenchmarkConfig()
        queries = load_queries(track=config.track, split=config.split, limit=config.limit)

        if not queries:
            console.print("[red]No queries found![/red]")
            return {}

        retrieval_queries = [q for q in queries if q.track == "retrieval"]
        rag_queries = [q for q in queries if q.track == "rag"]

        console.print(f"\n[bold]Company Search Benchmark[/bold]")
        console.print(f"  Searchers: {[s.name for s in self.searchers]}")
        console.print(f"  Retrieval queries: {len(retrieval_queries)}")
        console.print(f"  RAG queries: {len(rag_queries)}")
        console.print(f"  Exa enrichment: {'on' if config.enrich_exa_contents else 'off'}")
        console.print()

        results: dict[str, Any] = {"config": {"limit": config.limit}, "searchers": {}}

        with Progress(
            TextColumn("[cyan]{task.fields[name]:>10}[/cyan]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            for searcher in self.searchers:
                searcher_results: dict[str, Any] = {}

                if retrieval_queries:
                    task_id = progress.add_task(
                        "", name=f"{searcher.name}-ret", total=len(retrieval_queries)
                    )
                    grades = await self._run_retrieval(
                        searcher, retrieval_queries, config, progress, task_id
                    )
                    metrics = compute_retrieval_metrics(grades)
                    searcher_results["retrieval"] = {
                        "metrics": {
                            "match": metrics.match,
                            "recall_at_10": metrics.recall_at_10,
                            "precision": metrics.precision,
                            "num_queries": metrics.num_queries,
                        }
                    }

                if rag_queries:
                    task_id = progress.add_task(
                        "", name=f"{searcher.name}-rag", total=len(rag_queries)
                    )
                    grades = await self._run_rag(
                        searcher, rag_queries, config, progress, task_id
                    )
                    metrics = compute_rag_metrics(grades)
                    searcher_results["rag"] = {
                        "metrics": {
                            "accuracy": metrics.accuracy,
                            "num_queries": metrics.num_queries,
                        }
                    }

                results["searchers"][searcher.name] = searcher_results

        _print_summary(results)

        if config.output_file:
            with open(config.output_file, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"\n[green]Saved to {config.output_file}[/green]")

        return results


def _print_summary(results: dict[str, Any]):
    """Print benchmark results summary."""
    console.print("\n[bold]Results[/bold]\n")
    searchers = results.get("searchers", {})

    if not searchers:
        return

    has_retrieval = any("retrieval" in s for s in searchers.values())
    has_rag = any("rag" in s for s in searchers.values())

    if has_retrieval:
        t = Table(title="Retrieval Track")
        t.add_column("Searcher", style="cyan")
        for col in ["R@1", "R@10", "Precision", "Queries"]:
            t.add_column(col, justify="right")

        for name, data in searchers.items():
            if "retrieval" in data:
                m = data["retrieval"].get("metrics", {})
                t.add_row(
                    name,
                    f"{m.get('match', 0):.1%}",
                    f"{m.get('recall_at_10', 0):.1%}",
                    f"{m.get('precision', 0):.1%}",
                    str(m.get("num_queries", 0)),
                )

        console.print(t)
        console.print()

    if has_rag:
        t = Table(title="RAG Track")
        t.add_column("Searcher", style="cyan")
        for col in ["Accuracy", "Queries"]:
            t.add_column(col, justify="right")

        for name, data in searchers.items():
            if "rag" in data:
                m = data["rag"].get("metrics", {})
                t.add_row(
                    name,
                    f"{m.get('accuracy', 0):.1%}",
                    str(m.get("num_queries", 0)),
                )

        console.print(t)


def _build_searcher(name: str) -> Searcher | None:
    """Build a searcher by name."""
    try:
        if name == "exa":
            from .searchers.exa import ExaSearcher

            return ExaSearcher(category="company")
    except (ValueError, ImportError) as e:
        console.print(f"[yellow]{name}: {e}[/yellow]")
    return None


def main():
    """CLI entry point."""
    retrieval_exists = (DATA_DIR / "company" / "retrieval").exists()
    rag_exists = (DATA_DIR / "company" / "rag").exists()

    if not retrieval_exists and not rag_exists:
        console.print("[red]No benchmark data found![/red]")
        console.print("\nMake sure data/company/ directory contains query files.")
        return

    parser = argparse.ArgumentParser(description="Company Search Benchmark")
    parser.add_argument("--limit", type=int, help="Limit number of queries")
    parser.add_argument("--num-results", type=int, default=10, help="Results per query")
    parser.add_argument("--output", "-o", help="Output file for results JSON")
    parser.add_argument(
        "--enrich-exa-contents", action="store_true", help="Fetch page contents via Exa API"
    )
    parser.add_argument(
        "--track", choices=["retrieval", "rag"], help="Run only specific track"
    )
    parser.add_argument(
        "--split", choices=["static", "dynamic"], help="Run only specific split"
    )
    parser.add_argument(
        "--searchers", nargs="+", help="Searchers to use (default: exa)"
    )
    args = parser.parse_args()

    searcher_names = args.searchers or ["exa"]
    searchers = [s for name in searcher_names if (s := _build_searcher(name))]

    if not searchers:
        console.print("[red]No searchers available![/red]")
        return

    config = BenchmarkConfig(
        limit=args.limit,
        num_results=args.num_results,
        output_file=args.output,
        enrich_exa_contents=args.enrich_exa_contents,
        track=args.track,
        split=args.split,
    )
    asyncio.run(Benchmark(searchers).run(config))


if __name__ == "__main__":
    main()

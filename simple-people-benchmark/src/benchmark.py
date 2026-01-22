import argparse
import asyncio
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table

from benchmarks.shared.graders import PeopleGrader
from benchmarks.shared.searchers import SearchResult, Searcher

from .metrics import compute_retrieval_metrics

console = Console()
logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent / "data"
RUNS_DIR = Path(__file__).parent.parent / "runs"


@dataclass
class Query:
    query_id: str
    text: str
    bucket: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    limit: int | None = None
    num_results: int = 10
    output_file: str | None = None
    enrich_exa_contents: bool = False


def load_queries(limit: int | None = None) -> list[Query]:
    filepath = DATA_DIR / "people" / "simple_people_search.jsonl"
    if not filepath.exists():
        return []

    queries = []
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            queries.append(
                Query(
                    query_id=data.get("query_id", ""),
                    text=data.get("text", ""),
                    bucket=data.get("bucket", ""),
                    metadata=data.get("metadata", {}),
                )
            )

    return queries[:limit] if limit else queries


async def fetch_exa_contents(urls: list[str], api_key: str | None = None) -> dict[str, str]:
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
    try:
        contents = await fetch_exa_contents([r.url for r in results if r.url])
    except Exception as e:
        logger.warning(f"Content fetch failed: {e}")
        return results
    return [SearchResult(r.url, r.title, contents.get(r.url, r.text), r.metadata) for r in results]


@dataclass
class RunLog:
    run_id: str
    timestamp: str
    config: dict
    searchers: list[str]
    grades: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def save(self):
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        filepath = RUNS_DIR / f"{self.run_id}.json"
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return filepath


class Benchmark:
    def __init__(self, searchers: list[Searcher], grading_concurrency: int = 50):
        self.searchers = searchers
        self.grader = PeopleGrader()
        self._grade_semaphore = asyncio.Semaphore(grading_concurrency)
        self._run_log: RunLog | None = None

    async def _grade(self, query: Query, results: list[SearchResult]) -> list[dict]:
        async def grade_one(rank: int, r: SearchResult) -> dict:
            async with self._grade_semaphore:
                g = await self.grader.grade(query.text, r)
            return {
                "query_id": query.query_id,
                "rank": rank,
                "is_match": g.scores.get("is_match", 0),
            }

        return await asyncio.gather(*[grade_one(i, r) for i, r in enumerate(results, 1)])

    async def _run_searcher(
        self,
        searcher: Searcher,
        queries: list[Query],
        config: BenchmarkConfig,
        progress: Progress,
        task_id: TaskID,
    ) -> list[dict]:
        grades = []
        semaphore = asyncio.Semaphore(5)

        async def process(q: Query):
            async with semaphore:
                results = await searcher.search(q.text, config.num_results)
                if config.enrich_exa_contents:
                    results = await enrich_results(results)
                grades.extend(await self._grade(q, results))
                progress.advance(task_id)

        await asyncio.gather(*[process(q) for q in queries])
        return grades

    async def run(self, config: BenchmarkConfig | None = None) -> dict[str, Any]:
        config = config or BenchmarkConfig()
        queries = load_queries(config.limit)

        if not queries:
            console.print("[red]No queries found![/red]")
            console.print("Make sure data/people/simple_people_search.jsonl exists.")
            return {}

        run_id = str(uuid.uuid4())
        self._run_log = RunLog(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            config={
                "limit": config.limit,
                "num_results": config.num_results,
                "enrich_exa_contents": config.enrich_exa_contents,
            },
            searchers=[s.name for s in self.searchers],
        )

        console.print("\n[bold]People Search Benchmark[/bold]")
        console.print(f"  Run ID: {run_id}")
        console.print(f"  Searchers: {[s.name for s in self.searchers]}")
        console.print(f"  Queries: {len(queries)}")
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
            tasks = {
                s.name: progress.add_task("", name=s.name, total=len(queries))
                for s in self.searchers
            }

            async def run_one(searcher: Searcher) -> tuple[str, list[dict]]:
                grades = await self._run_searcher(
                    searcher, queries, config, progress, tasks[searcher.name]
                )
                return searcher.name, grades

            all_grades = await asyncio.gather(*[run_one(s) for s in self.searchers])

        for name, grades in all_grades:
            if grades:
                self._run_log.grades.extend(grades)
                results["searchers"][name] = {
                    "metrics": compute_retrieval_metrics(grades).__dict__,
                    "grades": grades,
                }

        self._run_log.metrics = {
            name: data.get("metrics", {}) for name, data in results.get("searchers", {}).items()
        }
        run_file = self._run_log.save()
        console.print(f"\n[green]Run log saved to {run_file}[/green]")

        _print_summary(results)

        if config.output_file:
            with open(config.output_file, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"\n[green]Saved to {config.output_file}[/green]")

        return results


def _print_summary(results: dict[str, Any]):
    console.print("\n[bold]Results[/bold]\n")
    searchers = results.get("searchers", {})

    if not searchers:
        return

    t = Table(title="People Search")
    t.add_column("Searcher", style="cyan")
    for col in ["R@1", "R@10", "Precision", "Queries"]:
        t.add_column(col, justify="right")

    for name, data in searchers.items():
        m = data.get("metrics", {})
        t.add_row(
            name,
            f"{m.get('match', 0):.1%}",
            f"{m.get('recall_at_10', 0):.1%}",
            f"{m.get('precision', 0):.1%}",
            str(m.get("num_queries", 0)),
        )

    console.print(t)


def _build_searcher(name: str) -> Searcher | None:
    try:
        if name == "exa":
            from .searchers.exa import ExaSearcher

            return ExaSearcher(category="people")
        if name == "brave":
            from .searchers.brave import BraveSearcher

            return BraveSearcher(site_filter="linkedin.com/in")
        if name == "parallel":
            from .searchers.parallel import ParallelSearcher

            return ParallelSearcher(source_policy={"include_domains": ["linkedin.com"]})
    except (ValueError, ImportError) as e:
        console.print(f"[yellow]{name}: {e}[/yellow]")
    return None


def main():
    queries_exist = (DATA_DIR / "people" / "simple_people_search.jsonl").exists()

    if not queries_exist:
        console.print("[red]No benchmark data found![/red]")
        console.print("\nDownload the people dataset:")
        console.print(
            "  curl -L https://github.com/exa-labs/people-benchmark/releases/latest/download/data.tar.gz | tar xz"
        )
        return

    parser = argparse.ArgumentParser(description="People Search Benchmark")
    parser.add_argument("--limit", type=int, help="Limit number of queries")
    parser.add_argument("--num-results", type=int, default=10, help="Results per query")
    parser.add_argument("--output", "-o", help="Output file for results JSON")
    parser.add_argument(
        "--enrich-exa-contents", action="store_true", help="Fetch page contents via Exa API"
    )
    parser.add_argument(
        "--searchers", nargs="+", help="Searchers to use (default: exa, brave, parallel)"
    )
    args = parser.parse_args()

    searcher_names = args.searchers or ["exa", "brave", "parallel"]
    searchers = [s for name in searcher_names if (s := _build_searcher(name))]

    if not searchers:
        console.print("[red]No searchers available![/red]")
        return

    config = BenchmarkConfig(
        limit=args.limit,
        num_results=args.num_results,
        output_file=args.output,
        enrich_exa_contents=args.enrich_exa_contents,
    )
    asyncio.run(Benchmark(searchers).run(config))


if __name__ == "__main__":
    main()

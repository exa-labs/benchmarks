import argparse
import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table
from shared.graders import PaperRetrievalGrader
from shared.searchers import Searcher, SearchResult

from .metrics import compute_retrieval_metrics

console = Console()
logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent / "data"
RUNS_DIR = Path(__file__).parent.parent / "runs"

TRACKS = ("paper", "tot")


@dataclass
class Query:
    query_id: str
    text: str
    track: str = ""
    bucket: str = ""
    gold_paper: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    limit: int | None = None
    num_results: int = 10
    output_file: str | None = None
    track: str | None = None


def load_queries(track: str | None = None, limit: int | None = None) -> list[Query]:
    filepath = DATA_DIR / "publication" / "publication_search.jsonl"
    if not filepath.exists():
        return []

    queries = []
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if track and data.get("track") != track:
                continue
            queries.append(
                Query(
                    query_id=data.get("query_id", ""),
                    text=data.get("text", ""),
                    track=data.get("track", ""),
                    bucket=data.get("bucket", ""),
                    gold_paper=data.get("gold_paper", {}),
                    metadata=data.get("metadata", {}),
                )
            )

    return queries[:limit] if limit else queries


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
    def __init__(self, searchers: list[Searcher]):
        self.searchers = searchers
        self.grader = PaperRetrievalGrader()
        self._run_log: RunLog | None = None

    def _grade(self, query: Query, results: list[SearchResult]) -> list[dict]:
        grades = []
        for rank, r in enumerate(results, 1):
            g = self.grader.grade(r, query.gold_paper)
            grades.append(
                {
                    "query_id": query.query_id,
                    "track": query.track,
                    "rank": rank,
                    "is_match": g.scores.get("is_match", 0),
                    "match_method": g.details.get("match_method"),
                }
            )
        return grades

    async def _run_searcher(
        self,
        searcher: Searcher,
        queries: list[Query],
        config: BenchmarkConfig,
        progress: Progress,
        task_id: TaskID,
    ) -> list[dict]:
        grades: list[dict] = []
        semaphore = asyncio.Semaphore(5)

        async def process(q: Query):
            async with semaphore:
                try:
                    results = await searcher.search(q.text, config.num_results)
                except Exception as e:
                    logger.warning(f"{searcher.name} search failed for {q.query_id}: {e}")
                    results = []
                grades.extend(self._grade(q, results))
                progress.advance(task_id)

        await asyncio.gather(*[process(q) for q in queries])
        return grades

    async def run(self, config: BenchmarkConfig | None = None) -> dict[str, Any]:
        config = config or BenchmarkConfig()
        queries = load_queries(track=config.track, limit=config.limit)

        if not queries:
            console.print("[red]No queries found![/red]")
            console.print("Make sure data/publication/publication_search.jsonl exists.")
            return {}

        run_id = str(uuid.uuid4())
        self._run_log = RunLog(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            config={
                "limit": config.limit,
                "num_results": config.num_results,
                "track": config.track,
            },
            searchers=[s.name for s in self.searchers],
        )

        console.print("\n[bold]Publication Retrieval Benchmark[/bold]")
        console.print(f"  Run ID: {run_id}")
        console.print(f"  Searchers: {[s.name for s in self.searchers]}")
        console.print(f"  Queries: {len(queries)}")
        console.print()

        results: dict[str, Any] = {"config": {"limit": config.limit}, "searchers": {}}

        with Progress(
            TextColumn("[cyan]{task.fields[name]:>12}[/cyan]"),
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
            if not grades:
                continue
            self._run_log.grades.extend(grades)
            searcher_results: dict[str, Any] = {"overall": _metrics_dict(grades)}
            for track in TRACKS:
                track_grades = [g for g in grades if g["track"] == track]
                if track_grades:
                    searcher_results[track] = _metrics_dict(track_grades)
            results["searchers"][name] = searcher_results

        self._run_log.metrics = results["searchers"]
        run_file = self._run_log.save()
        console.print(f"\n[green]Run log saved to {run_file}[/green]")

        _print_summary(results)

        if config.output_file:
            with open(config.output_file, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"\n[green]Saved to {config.output_file}[/green]")

        return results


def _metrics_dict(grades: list[dict]) -> dict:
    return compute_retrieval_metrics(grades).__dict__


def _print_summary(results: dict[str, Any]):
    console.print("\n[bold]Results[/bold]\n")
    searchers = results.get("searchers", {})
    if not searchers:
        return

    section_titles = {
        "overall": "Overall",
        "paper": "Publication (specific-question)",
        "tot": "Tip-of-the-tongue",
    }
    for section, title in section_titles.items():
        if not any(section in data for data in searchers.values()):
            continue
        t = Table(title=title)
        t.add_column("Searcher", style="cyan")
        for col in ["R@1", "R@5", "R@10", "MRR", "Queries"]:
            t.add_column(col, justify="right")
        for name, data in searchers.items():
            if section not in data:
                continue
            m = data[section]
            t.add_row(
                name,
                f"{m.get('recall_at_1', 0):.1%}",
                f"{m.get('recall_at_5', 0):.1%}",
                f"{m.get('recall_at_10', 0):.1%}",
                f"{m.get('mrr', 0):.3f}",
                str(m.get("num_queries", 0)),
            )
        console.print(t)
        console.print()


def _build_searcher(name: str) -> Searcher | None:
    try:
        if name == "exa":
            from shared.searchers import ExaSearcher

            return ExaSearcher(category="publication", include_text=True)
        if name == "brave":
            from shared.searchers import BraveSearcher

            return BraveSearcher()
        if name == "parallel":
            from shared.searchers import ParallelSearcher

            return ParallelSearcher()
        if name == "perplexity":
            from shared.searchers import PerplexitySearcher

            return PerplexitySearcher()
    except (ValueError, ImportError) as e:
        console.print(f"[yellow]{name}: {e}[/yellow]")
    return None


def main():
    if not (DATA_DIR / "publication" / "publication_search.jsonl").exists():
        console.print("[red]No benchmark data found![/red]")
        console.print("Make sure data/publication/publication_search.jsonl exists.")
        return

    parser = argparse.ArgumentParser(description="Publication Retrieval Benchmark")
    parser.add_argument("--limit", type=int, help="Limit number of queries")
    parser.add_argument("--num-results", type=int, default=10, help="Results per query")
    parser.add_argument("--track", choices=list(TRACKS), help="Run only a specific track")
    parser.add_argument("--output", "-o", help="Output file for results JSON")
    parser.add_argument("--searchers", nargs="+", help="Searchers to use (default: exa)")
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
        track=args.track,
    )
    asyncio.run(Benchmark(searchers).run(config))


if __name__ == "__main__":
    main()

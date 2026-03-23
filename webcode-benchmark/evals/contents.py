"""Contents eval — extraction fidelity against golden markdown."""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from shared.graders import ContentsGrader
from shared.searchers import Searcher
from shared.searchers.claude import ClaudeWebFetchSearcher
from shared.searchers.exa import ExaSearcher
from shared.searchers.parallel import ParallelSearcher
from shared.searchers.tavily import TavilySearcher

from src.metrics import compute_contents_metrics

console = Console()
logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent / "data"


def load_queries(limit: int | None = None) -> list[dict]:
    filepath = DATA_DIR / "contents" / "code_contents.jsonl"
    if not filepath.exists():
        return []
    queries = []
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            queries.append(json.loads(line))
    return queries[:limit] if limit else queries


def _build_exa_searcher() -> ExaSearcher:
    return ExaSearcher(max_age_hours=0)


SEARCHER_BUILDERS: dict[str, callable] = {
    "exa": _build_exa_searcher,
    "tavily": TavilySearcher,
    "parallel": ParallelSearcher,
    "claude": ClaudeWebFetchSearcher,
}


def build_searcher(name: str) -> Searcher | None:
    builder = SEARCHER_BUILDERS.get(name)
    if builder is None:
        console.print(f"[yellow]Unknown searcher: {name}[/yellow]")
        return None
    try:
        return builder()
    except (ValueError, ImportError) as e:
        console.print(f"[yellow]{name}: {e}[/yellow]")
        return None


async def run(
    searcher_names: list[str],
    limit: int | None = None,
    output: str | None = None,
    concurrency: int = 5,
    grader_model: str = "gpt-5.4",
):
    queries = load_queries(limit)
    if not queries:
        console.print("[red]No queries found. Ensure data/contents/code_contents.jsonl exists.[/red]")
        return

    searchers = [s for name in searcher_names if (s := build_searcher(name))]
    if not searchers:
        console.print("[red]No searchers available.[/red]")
        return

    grader = ContentsGrader(model=grader_model)
    semaphore = asyncio.Semaphore(concurrency)
    all_results: dict[str, list[dict]] = {}

    console.print("\n[bold]Contents Extraction Eval[/bold]")
    console.print(f"  Queries: {len(queries)}")
    console.print(f"  Searchers: {[s.name for s in searchers]}\n")

    with Progress(
        TextColumn("[cyan]{task.fields[name]:>12}[/cyan]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for searcher in searchers:
            task_id = progress.add_task("", name=searcher.name, total=len(queries))
            grades = []

            async def process(q: dict) -> dict:
                async with semaphore:
                    url = q.get("url", "")
                    golden = q.get("expected_markdown", "")
                    start = time.time()
                    try:
                        results = await searcher.extract(url)
                        extracted = results[0].text if results else ""
                    except Exception as e:
                        logger.warning(f"Extraction failed for {url}: {e}")
                        extracted = ""
                    latency = time.time() - start

                    grade = await grader.grade(url, golden, extracted)
                    progress.advance(task_id)
                    return {
                        "id": q.get("id", ""),
                        "url": url,
                        "latency": round(latency, 2),
                        **grade.scores,
                    }

            grades = await asyncio.gather(*[process(q) for q in queries])
            all_results[searcher.name] = list(grades)

    _print_summary(all_results)

    if output:
        with open(output, "w") as f:
            json.dump(all_results, f, indent=2)
        console.print(f"\n[green]Results saved to {output}[/green]")


def _print_summary(all_results: dict[str, list[dict]]):
    table = Table(title="Contents Extraction Results")
    table.add_column("Searcher", style="cyan")
    for col in ["Completeness", "Accuracy", "Structure", "Signal", "Code Recall", "Table Recall", "ROUGE-L", "Queries"]:
        table.add_column(col, justify="right")

    for name, grades in all_results.items():
        metrics = compute_contents_metrics(grades)
        table.add_row(
            name,
            f"{metrics.completeness:.1%}",
            f"{metrics.accuracy:.1%}",
            f"{metrics.structure:.1%}",
            f"{metrics.signal:.1%}",
            f"{metrics.det_code_block_recall:.1%}",
            f"{metrics.det_table_recall:.1%}",
            f"{metrics.det_rouge_l:.1%}",
            str(metrics.num_queries),
        )

    console.print()
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Contents extraction eval")
    parser.add_argument("--searchers", nargs="+", default=["exa"], help="Searchers to evaluate")
    parser.add_argument("--limit", type=int, help="Limit number of queries")
    parser.add_argument("--output", "-o", help="Output file for results JSON")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--grader-model", default="gpt-5.4")
    args = parser.parse_args()

    asyncio.run(run(
        searcher_names=args.searchers,
        limit=args.limit,
        output=args.output,
        concurrency=args.concurrency,
        grader_model=args.grader_model,
    ))


if __name__ == "__main__":
    main()

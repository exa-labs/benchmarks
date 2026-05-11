"""RAG eval — full-web retrieval + synthesis on long-context code docs."""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from shared.agents import SimpleRAGAgent
from shared.graders import Citation, GroundedRAGGrader
from shared.graders.pplx_llm import PerplexityAgentLLMClient
from shared.searchers import Searcher
from shared.searchers.brave import BraveSearcher
from shared.searchers.exa import ExaSearcher
from shared.searchers.parallel import ParallelSearcher
from shared.searchers.perplexity import PerplexitySearcher
from shared.searchers.perplexity_agent import PerplexityAgentSearcher
from shared.searchers.tavily import TavilySearcher

# API keys for Perplexity variant arms — set via environment variables.
_PPLX_NOCF_KEY = os.environ["PPLX_NOCF_KEY"]
_PPLX_BETA_KEY = os.environ["PPLX_BETA_KEY"]

from src.metrics import compute_grounded_rag_metrics

console = Console()
logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent / "data"


def load_queries(limit: int | None = None) -> list[dict]:
    filepath = DATA_DIR / "rag" / "code_rag.jsonl"
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
    return ExaSearcher(search_type="fast", include_text=True, max_characters=20000, max_age_hours=0)


def _build_brave_searcher() -> BraveSearcher:
    return BraveSearcher(search_type="llm_context")


SEARCHER_BUILDERS: dict[str, callable] = {
    "exa": _build_exa_searcher,
    "brave": _build_brave_searcher,
    "perplexity": PerplexitySearcher,
    "parallel": ParallelSearcher,
    "tavily": TavilySearcher,
    # Perplexity Agent API variants (web_search tool, 4 arms).
    "pplx-nocf-noyt": lambda: PerplexityAgentSearcher(
        api_key=_PPLX_NOCF_KEY, name="pplx-nocf-noyt", excluded_domains=["-youtube.com"],
    ),
    "pplx-nocf": lambda: PerplexityAgentSearcher(
        api_key=_PPLX_NOCF_KEY, name="pplx-nocf", excluded_domains=[],
    ),
    "pplx-beta-noyt": lambda: PerplexityAgentSearcher(
        api_key=_PPLX_BETA_KEY, name="pplx-beta-noyt", excluded_domains=["-youtube.com"],
    ),
    "pplx-beta": lambda: PerplexityAgentSearcher(
        api_key=_PPLX_BETA_KEY, name="pplx-beta", excluded_domains=[],
    ),
    # High-step variant: max_steps=10, max_tokens=10000, max_tokens_per_page=4000.
    "pplx-ms10": lambda: PerplexityAgentSearcher(
        api_key=_PPLX_NOCF_KEY,
        name="pplx-ms10",
        excluded_domains=[],
        max_steps=10,
        max_tokens=10000,
        max_tokens_per_page=4000,
    ),
    "pplx-ms10-noyt": lambda: PerplexityAgentSearcher(
        api_key=_PPLX_NOCF_KEY,
        name="pplx-ms10-noyt",
        excluded_domains=["-youtube.com"],
        max_steps=10,
        max_tokens=10000,
        max_tokens_per_page=4000,
    ),
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
    num_results: int = 5,
    concurrency: int = 5,
    grader_model: str = "gpt-5.4",
    rag_model: str = "gpt-5-mini",
):
    queries = load_queries(limit)
    if not queries:
        console.print("[red]No queries found. Ensure data/rag/code_rag.jsonl exists.[/red]")
        return

    searchers = [s for name in searcher_names if (s := build_searcher(name))]
    if not searchers:
        console.print("[red]No searchers available.[/red]")
        return

    _pplx_llm = PerplexityAgentLLMClient(api_key=_PPLX_NOCF_KEY)
    grader = GroundedRAGGrader(model=grader_model, client=_pplx_llm)
    rag_agent = SimpleRAGAgent(model=rag_model, client=_pplx_llm)
    all_results: dict[str, list[dict]] = {}

    console.print("\n[bold]RAG Eval (Long-Context Code QA)[/bold]")
    console.print(f"  Queries: {len(queries)}")
    console.print(f"  Searchers: {[s.name for s in searchers]}\n")

    async def run_searcher(
        searcher: Searcher,
        task_id: int,
        sem: asyncio.Semaphore,
    ) -> tuple[str, list[dict]]:
        async def process(q: dict) -> dict:
            async with sem:
                query_text = q["query"]
                expected = q["expected_answer"]
                start = time.time()

                try:
                    results = await searcher.search(query_text, num_results=num_results)
                except Exception as e:
                    logger.warning(f"[{searcher.name}] search failed: {e}")
                    results = []

                latency = time.time() - start

                try:
                    rag_result = await rag_agent.synthesize(query_text, results)
                except Exception as e:
                    logger.warning(f"[{searcher.name}] rag synthesis failed: {e}")
                    progress.advance(task_id)
                    return {"id": q.get("id", ""), "query": query_text, "latency": round(latency, 2)}

                citations = [
                    Citation(url=c.url, title=c.title, text=c.text)
                    for c in rag_result.citations
                ]

                try:
                    grade = await grader.grade(
                        question=query_text,
                        expected_answer=expected,
                        predicted_answer=rag_result.answer,
                        citations=citations,
                    )
                except Exception as e:
                    logger.warning(f"[{searcher.name}] grading failed: {e}")
                    progress.advance(task_id)
                    return {"id": q.get("id", ""), "query": query_text, "latency": round(latency, 2)}

                progress.advance(task_id)

                return {
                    "id": q.get("id", ""),
                    "query": query_text,
                    "latency": round(latency, 2),
                    **grade.scores,
                }

        grades = await asyncio.gather(*[process(q) for q in queries])
        return searcher.name, list(grades)

    with Progress(
        TextColumn("[cyan]{task.fields[name]:>12}[/cyan]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # One progress bar + semaphore per searcher; all run in parallel.
        searcher_tasks = [
            run_searcher(
                s,
                progress.add_task("", name=s.name, total=len(queries)),
                asyncio.Semaphore(concurrency),
            )
            for s in searchers
        ]
        pairs = await asyncio.gather(*searcher_tasks)
        all_results = dict(pairs)

    _print_summary(all_results)

    if output:
        with open(output, "w") as f:
            json.dump(all_results, f, indent=2)
        console.print(f"\n[green]Results saved to {output}[/green]")


def _print_summary(all_results: dict[str, list[dict]]):
    table = Table(title="RAG Eval Results")
    table.add_column("Searcher", style="cyan")
    for col in ["Groundedness", "Correctness", "Citation Prec.", "Avg Tokens", "Queries"]:
        table.add_column(col, justify="right")

    for name, grades in all_results.items():
        metrics = compute_grounded_rag_metrics(grades)
        table.add_row(
            name,
            f"{metrics.groundedness:.1%}",
            f"{metrics.correctness:.1%}",
            f"{metrics.citation_precision:.1%}",
            f"{metrics.avg_citation_tokens:.0f}",
            str(metrics.num_queries),
        )

    console.print()
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="RAG eval (long-context code QA)")
    parser.add_argument("--searchers", nargs="+", default=["exa"], help="Searchers to evaluate")
    parser.add_argument("--limit", type=int, help="Limit number of queries")
    parser.add_argument("--num-results", type=int, default=5, help="Results per search query")
    parser.add_argument("--output", "-o", help="Output file for results JSON")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--grader-model", default="gpt-5.4")
    parser.add_argument("--rag-model", default="gpt-5-mini")
    args = parser.parse_args()

    asyncio.run(run(
        searcher_names=args.searchers,
        limit=args.limit,
        output=args.output,
        num_results=args.num_results,
        concurrency=args.concurrency,
        grader_model=args.grader_model,
        rag_model=args.rag_model,
    ))


if __name__ == "__main__":
    main()

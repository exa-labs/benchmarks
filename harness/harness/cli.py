"""``bench``: list systems and suites, download data, run and summarize evaluations.

Examples::

    uv run bench list
    uv run bench download --suite browsecomp
    uv run bench run --system scout-exa-auto --suite browsecomp --limit 5
    uv run bench run --system scout-brave --model anthropic/claude-sonnet-5 --suite dsqa
    uv run bench summary runs/scout-exa-auto-browsecomp-1a2b3c4d5e
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from harness.llm.clients import provider_of
from harness.llm.judge import DEFAULT_JUDGE_MODEL, Judge
from harness.runner import (
    Runner,
    load_summary_outcomes,
    read_json,
    safe_name,
    summarize,
    write_json,
)
from harness.suites.registry import get_suite, list_suites
from harness.systems import DEFAULT_CATALOG, Catalog, SystemSpec

console = Console()

PROVIDER_KEYS = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "exa": ("EXA_API_KEY",),
    "brave": ("BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"),
    "parallel": ("PARALLEL_API_KEY",),
    "perplexity": ("PERPLEXITY_API_KEY",),
}


def missing_credentials(spec: SystemSpec, judge_model: str) -> list[str]:
    """Name each provider whose API key is absent, before any paid call is made."""
    providers = {provider_of(spec.model), provider_of(judge_model)}
    if spec.searcher is not None:
        providers.add(spec.searcher["provider"])
    missing = []
    for provider in sorted(providers):
        names = PROVIDER_KEYS[provider]
        if not any(os.environ.get(name) for name in names):
            missing.append(" or ".join(names))
    return missing


def cmd_list(args: argparse.Namespace) -> int:
    catalog = Catalog.load(args.catalog)
    table = Table(title="Systems")
    for column in ("system", "kind", "model", "search backend"):
        table.add_column(column)
    for name in sorted(catalog.systems):
        spec = catalog.resolve(name)
        backend = (
            f"{spec.searcher['name']} ({spec.searcher['provider']})"
            if spec.searcher
            else f"{provider_of(spec.model)} hosted web search"
        )
        table.add_row(name, spec.kind, spec.model, backend)
    console.print(table)
    suites = Table(title="Suites")
    for column in ("suite", "primary metric", "description"):
        suites.add_column(column)
    for name in list_suites():
        suite = get_suite(name)
        suites.add_row(name, suite.primary_metric, suite.description)
    console.print(suites)
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    failures = 0
    for name in [args.suite] if args.suite else list_suites():
        try:
            tasks = get_suite(name).load()
        except Exception as error:  # report every suite, then fail the command
            failures += 1
            console.print(f"[red]{name}: {error}[/red]")
            continue
        console.print(f"{name}: {len(tasks)} tasks")
    return 1 if failures else 0


def cmd_run(args: argparse.Namespace) -> int:
    catalog = Catalog.load(args.catalog)
    spec = catalog.resolve(args.system, model=args.model)
    missing = missing_credentials(spec, args.judge_model)
    if missing:
        console.print(f"[red]Missing credentials: {', '.join(missing)}[/red]")
        return 2
    suite = get_suite(args.suite)
    tasks = suite.load()
    if args.limit is not None:
        tasks = tasks[: args.limit]
    runner = Runner(
        spec,
        suite,
        judge=Judge(args.judge_model),
        runs_root=Path(args.runs_dir),
        run_suffix=args.run_suffix,
        concurrency=args.concurrency,
    )
    console.print(f"Run directory: {runner.run_dir}")
    with Progress(
        TextColumn(f"[cyan]{spec.name} × {suite.name}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task("", total=len(tasks))
        summary = asyncio.run(runner.run(tasks, on_done=lambda _: progress.advance(bar)))
    print_summary(summary)
    return 0 if summary["failed"] == 0 else 1


def cmd_summary(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    config = read_json(run_dir / "config.json")
    suite = get_suite(config["suite"])
    tasks = [t for t in suite.load() if (run_dir / "tasks" / safe_name(t.id)).exists()]
    summary = summarize(suite, load_summary_outcomes(run_dir, suite, tasks), config)
    write_json(run_dir / "summary.json", summary)
    print_summary(summary)
    return 0


def print_summary(summary: dict) -> None:
    table = Table(title=f"{summary['system']} × {summary['suite']}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in summary["metrics"].items():
        table.add_row(key, f"{value:.4f}")
    primary = summary["primary_metric"]
    table.add_row(f"{primary} (failed as zero)", f"{summary[f'{primary}_failed_as_zero']:.4f}")
    table.add_row("graded / tasks", f"{summary['graded']} / {summary['tasks']}")
    cost = summary["cost"]
    known = "" if cost["system_cost_known"] else " (incomplete)"
    table.add_row("system cost per task", f"${cost['system_usd_per_task']:.4f}{known}")
    table.add_row("judge cost", f"${cost['judge_usd']:.4f}")
    table.add_row("mean searches", f"{summary['mean_searches']:.2f}")
    table.add_row("mean latency", f"{summary['mean_latency_s']:.1f}s")
    table.add_row("stop reasons", json.dumps(summary["stop_reasons"]))
    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__.split("\n")[0])
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG), help="systems.toml path")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="list systems and suites").set_defaults(func=cmd_list)

    download = commands.add_parser("download", help="download and verify suite data")
    download.add_argument("--suite")
    download.set_defaults(func=cmd_download)

    run = commands.add_parser("run", help="evaluate one system on one suite (resumable)")
    run.add_argument("--system", required=True)
    run.add_argument("--suite", required=True)
    run.add_argument("--model", help="override the system's model, e.g. anthropic/claude-sonnet-5")
    run.add_argument("--limit", type=int, help="evaluate only the first N tasks")
    run.add_argument("--concurrency", type=int, default=5)
    run.add_argument("--run-suffix", help="start an independent repeat of the same setup")
    run.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    run.add_argument("--runs-dir", default="runs")
    run.set_defaults(func=cmd_run)

    summary = commands.add_parser("summary", help="recompute a run directory's summary")
    summary.add_argument("run_dir")
    summary.set_defaults(func=cmd_summary)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()

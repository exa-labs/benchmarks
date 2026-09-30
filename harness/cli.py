"""``bench``: list systems and suites, download data, run and summarize evaluations.

Examples::

    uv run bench list
    uv run bench download --suite browsecomp
    uv run bench run --system scout-exa-auto-highlights --suite browsecomp --limit 5
    uv run bench run --system scout-brave-llm-context --model anthropic/claude-sonnet-5 --suite dsqa
    uv run bench summary results/runs/scout-exa-auto-highlights-browsecomp-1a2b3c4d5e
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from benchmarks import get_suite, list_suites
from harness.llm import LLM_DEFAULT
from harness.llm.clients import provider_of
from harness.llm.judge import Judge
from harness.runner import (
    DEFAULT_RUNS_ROOT,
    Runner,
    load_summary_outcomes,
    read_json,
    safe_name,
    summarize,
    write_json,
)
from harness.systems import DEFAULT_CATALOG, Catalog, SystemSpec

console = Console()

PROVIDER_KEYS = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "exa": ("EXA_API_KEY",),
    "brave": ("BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"),
    "parallel": ("PARALLEL_API_KEY", "PARALLELS_API_KEY"),
    "perplexity": ("PERPLEXITY_API_KEY",),
    "claude": ("ANTHROPIC_API_KEY",),
}


def missing_credentials(spec: SystemSpec, judge_model: str | None) -> list[str]:
    """Name each provider whose API key is absent, before any paid call is made."""
    providers = {provider_of(model) for model in (spec.model, judge_model) if model}
    if spec.settings.get("enrich_exa_contents"):
        providers.add("exa")
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
        if spec.searcher:
            backend = f"{spec.searcher['name']} ({spec.searcher['provider']})"
        else:
            assert spec.model is not None
            backend = f"{provider_of(spec.model)} hosted web search"
        table.add_row(name, spec.kind, spec.model or "—", backend)
    console.print(table)
    suites = Table(title="Suites")
    for column in ("suite", "systems", "primary metric", "description"):
        suites.add_column(column)
    for name in list_suites():
        suite = get_suite(name)
        suites.add_row(name, ", ".join(suite.system_kinds), suite.primary_metric, suite.description)
    console.print(suites)
    console.print("WebCode E2E: dataset-only export (33 tasks); no execution harness.")
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


def select_names(requested: list[str], available: list[str], label: str) -> list[str]:
    """Expand an explicit all selector and reject typos before any calls."""
    if requested == ["all"]:
        return sorted(available)
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(sorted(unknown))}")
    return list(dict.fromkeys(requested))


def cmd_run(args: argparse.Namespace) -> int:
    """Validate the entire suite/system matrix before executing compatible pairs."""
    catalog = Catalog.load(args.catalog)
    names = select_names(args.system, list(catalog.systems), "systems")
    suites = [get_suite(name) for name in select_names(args.suite, list_suites(), "suites")]
    plan = []
    errors = []
    for suite in suites:
        tasks = suite.load()
        if args.split:
            tasks = [t for t in tasks if t.metadata.get("split") == args.split]
        if args.limit is not None:
            tasks = tasks[: args.limit]
        if not tasks:
            errors.append(f"{suite.name}: selection contains no tasks")
            continue
        compatible = []
        for name in names:
            spec = catalog.resolve(name)
            if args.model and spec.kind != "search":
                spec = catalog.resolve(name, model=args.model)
            if spec.kind not in suite.system_kinds:
                continue
            settings = dict(spec.settings)
            if args.num_results is not None:
                settings["num_results"] = args.num_results
            if args.enrich_exa_contents:
                if spec.kind not in ("search", "rag"):
                    errors.append("--enrich-exa-contents requires search or rag systems")
                    continue
                settings["enrich_exa_contents"] = True
            spec = replace(spec, settings=settings)
            missing = missing_credentials(spec, args.judge_model if suite.requires_judge else None)
            if missing:
                errors.append(f"{name} × {suite.name}: missing {', '.join(missing)}")
            compatible.append(name)
            plan.append((spec, suite, tasks))
        if not compatible:
            errors.append(f"{suite.name}: select a system of kind {', '.join(suite.system_kinds)}")
    for spec, suite, tasks in plan:
        console.print(f"{spec.name} × {suite.name}: {len(tasks)} tasks")
    if errors:
        for error in dict.fromkeys(errors):
            console.print(f"[red]{error}[/red]")
        return 2
    if args.dry_run:
        console.print("Preflight passed; no paid API calls made.")
        return 0
    summaries = []
    for spec, suite, tasks in plan:
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
        summaries.append(summary)
    if args.output:
        write_json(Path(args.output), summaries)
    return 0 if all(s["failed"] == 0 for s in summaries) else 1


def cmd_summary(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    config = read_json(run_dir / "config.json")
    suite = get_suite(config["suite"])
    tasks = [t for t in suite.load() if (run_dir / "tasks" / safe_name(t.id)).exists()]
    summary = summarize(suite, load_summary_outcomes(run_dir, suite, tasks), config)
    write_json(run_dir / "summary.json", summary)
    if args.output:
        write_json(Path(args.output), [summary])
    print_summary(summary)
    return 0


def print_summary(summary: dict) -> None:
    table = Table(title=f"{summary['system']} × {summary['suite']}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_column("95% CI", justify="right")
    intervals = summary["confidence_intervals"]["metrics"]

    def interval_text(key: str) -> str:
        interval = intervals[key]
        if interval["low"] is None:
            return "— (n < 2)"
        return f"[{interval['low']:.4f}, {interval['high']:.4f}]"

    for key, value in summary["metrics"].items():
        table.add_row(key, f"{value:.4f}", interval_text(key))
    primary = summary["primary_metric"]
    failed_key = f"{primary}_failed_as_zero"
    table.add_row(
        f"{primary} (failed as zero)", f"{summary[failed_key]:.4f}", interval_text(failed_key)
    )
    table.add_row("graded / tasks", f"{summary['graded']} / {summary['tasks']}")
    cost = summary["cost"]
    known = "" if cost["system_cost_known"] else " (incomplete)"
    table.add_row("system cost per task", f"${cost['system_usd_per_task']:.4f}{known}")
    judge_known = "" if cost["judge_cost_known"] else " (incomplete)"
    table.add_row("judge cost", f"${cost['judge_usd']:.4f}{judge_known}")
    table.add_row("mean searches", f"{summary['mean_searches']:.2f}")
    table.add_row("mean latency", f"{summary['mean_latency_s']:.1f}s")
    table.add_row("stop reasons", json.dumps(summary["stop_reasons"]))
    console.print(table)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__.split("\n")[0])
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG), help="systems.toml path")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="list systems and suites").set_defaults(func=cmd_list)

    download = commands.add_parser("download", help="download and verify suite data")
    download.add_argument("--suite")
    download.set_defaults(func=cmd_download)

    run = commands.add_parser("run", help="evaluate compatible system/suite pairs (resumable)")
    run.add_argument("--system", nargs="+", required=True, help="system names or all")
    run.add_argument("--suite", nargs="+", required=True, help="suite names or all")
    run.add_argument("--model", help="override the system's model, e.g. anthropic/claude-sonnet-5")
    run.add_argument("--limit", type=positive_int, help="evaluate only the first N tasks")
    run.add_argument("--concurrency", type=positive_int, default=5)
    run.add_argument("--run-suffix", help="start an independent repeat of the same setup")
    run.add_argument("--judge-model", default=LLM_DEFAULT)
    run.add_argument("--runs-dir", default=str(DEFAULT_RUNS_ROOT))
    run.add_argument("--num-results", type=positive_int)
    run.add_argument("--split", choices=("static", "dynamic"))
    run.add_argument("--enrich-exa-contents", action="store_true")
    run.add_argument("--output", "-o", help="write aggregate summaries as JSON")
    run.add_argument(
        "--dry-run", action="store_true", help="preflight all pairs without paid calls"
    )
    run.set_defaults(func=cmd_run)

    summary = commands.add_parser("summary", help="recompute a run directory's summary")
    summary.add_argument("run_dir")
    summary.add_argument("--output", "-o", help="export a JSON list containing this summary")
    summary.set_defaults(func=cmd_summary)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        status = args.func(args)
    except (ValueError, KeyError, FileNotFoundError) as error:
        console.print(f"[red]{error}[/red]")
        status = 2
    sys.exit(status)


if __name__ == "__main__":
    main()

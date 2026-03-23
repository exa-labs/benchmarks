"""E2E eval — sandboxed coding tasks requiring web search (dataset only)."""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
DATA_DIR = Path(__file__).parent.parent / "data"


def load_tasks(limit: int | None = None) -> list[dict]:
    filepath = DATA_DIR / "e2e" / "code_e2e.jsonl"
    if not filepath.exists():
        return []
    tasks = []
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            tasks.append(json.loads(line))
    return tasks[:limit] if limit else tasks


def print_info():
    tasks = load_tasks()

    console.print("\n[bold]End-to-End Code Tasks Dataset[/bold]")
    console.print(
        "  GitHub release tasks (post-2026-02-01, 100+ star repos)."
    )
    console.print(
        "  Focuses on breaking changes and new functions."
    )
    console.print(
        "  No agent harness included — bring your own (e.g. mini-swe-agent).\n"
    )

    if not tasks:
        console.print("[yellow]No tasks loaded. Ensure data/e2e/code_e2e.jsonl exists.[/yellow]")
        return

    console.print(f"  Total tasks: {len(tasks)}")

    repos = set(t.get("repo", "") for t in tasks)
    console.print(f"  Unique repos: {len(repos)}\n")

    table = Table(title="Sample Tasks (first 5)")
    table.add_column("ID", style="cyan", max_width=20)
    table.add_column("Repo", max_width=30)
    table.add_column("Release Tag", max_width=15)
    table.add_column("Description", max_width=60)

    for task in tasks[:5]:
        desc = task.get("task_description", "")
        if len(desc) > 60:
            desc = desc[:57] + "..."
        table.add_row(
            task.get("id", ""),
            task.get("repo", ""),
            task.get("release_tag", ""),
            desc,
        )

    console.print(table)

    console.print("\n[dim]Schema: id, slug, repo, repo_url, release_tag, task_description, test_patch, metadata[/dim]")
    console.print("[dim]Load with: from evals.e2e import load_tasks[/dim]")


def main():
    parser = argparse.ArgumentParser(description="E2E code tasks dataset (no runner)")
    parser.add_argument("--info", action="store_true", default=True, help="Print dataset info")
    parser.add_argument("--limit", type=int, help="Limit tasks displayed")
    parser.parse_args()

    print_info()


if __name__ == "__main__":
    main()

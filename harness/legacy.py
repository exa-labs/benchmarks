"""Compatibility commands translating legacy flags into common harness runs.

These entry points own no execution, grading, aggregation, or persistence logic.
The systems catalog owns provider settings; suites own dataset and grade contracts.
"""

from __future__ import annotations

import argparse

from harness.cli import main, positive_int
from harness.llm.judge import DEFAULT_JUDGE_MODEL
from harness.runner import DEFAULT_RUNS_ROOT

_DEFAULTS = {
    "people": ["exa", "brave", "parallel"],
    "company": ["exa"],
    "publication": ["exa"],
}


def system_name(family: str, track: str, provider: str) -> str:
    """Map legacy provider aliases to explicit, reproducible catalog presets."""
    base = {
        "exa": "exa-auto-highlights",
        "brave": "brave-llm-context",
        "parallel": "parallel-advanced",
        "perplexity": "perplexity-web",
    }
    if family == "people":
        base.update(exa="exa-people", brave="brave-people", parallel="parallel-people")
    elif family == "company":
        base["exa"] = "exa-company"
    elif family == "publication":
        base["exa"] = "exa-publication"
    kind = "rag" if track == "rag" else "search"
    return f"{kind}-{base.get(provider, provider)}"


def legacy_main(family: str, argv: list[str] | None = None) -> None:
    """Translate the family's provider aliases and track flags into a shared run plan."""
    parser = argparse.ArgumentParser(
        description=f"{family.replace('-', ' ').title()} Benchmark (shared harness)"
    )
    parser.add_argument("--searchers", nargs="+", default=_DEFAULTS[family])
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--num-results", type=positive_int)
    parser.add_argument("--output", "-o")
    parser.add_argument("--concurrency", type=positive_int, default=5)
    parser.add_argument(
        "--grader-model", "--judge-model", dest="judge_model", default=DEFAULT_JUDGE_MODEL
    )
    parser.add_argument("--rag-model", "--model", dest="model")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--run-suffix")
    parser.add_argument("--dry-run", action="store_true")
    if family == "company":
        parser.add_argument("--track", choices=("retrieval", "rag"))
        parser.add_argument("--split", choices=("static", "dynamic"))
    elif family == "publication":
        parser.add_argument("--track", choices=("paper", "tot"))
    if family in ("people", "company"):
        parser.add_argument("--enrich-exa-contents", action="store_true")
    args = parser.parse_args(argv)
    if family == "company":
        tracks = [args.track] if args.track else ["retrieval", "rag"]
        suites = [f"company-{track}" for track in tracks]
    elif family == "publication":
        tracks = [args.track] if args.track else ["paper", "tot"]
        suites = ["publication-tot" if track == "tot" else "publication" for track in tracks]
    else:
        tracks = ["retrieval"]
        suites = [family]
    systems = list(
        dict.fromkeys(
            system_name(family, track, provider) for track in tracks for provider in args.searchers
        )
    )
    forwarded = ["run", "--system", *systems, "--suite", *suites]
    for name in (
        "limit",
        "num_results",
        "output",
        "concurrency",
        "judge_model",
        "model",
        "runs_dir",
        "run_suffix",
        "split",
    ):
        value = getattr(args, name, None)
        if value is not None:
            if name in ("judge_model", "model") and "/" not in value:
                value = f"openai/{value}"
            forwarded.extend(["--" + name.replace("_", "-"), str(value)])
    for name in ("enrich_exa_contents", "dry_run"):
        if getattr(args, name, False):
            forwarded.append("--" + name.replace("_", "-"))
    main(forwarded)


def people_main() -> None:
    legacy_main("people")


def company_main() -> None:
    legacy_main("company")


def publication_main() -> None:
    legacy_main("publication")

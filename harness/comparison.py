"""Compare aligned Company FindAll runs with an explicit, reproducible fleet."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from benchmarks.company_findall import CompanyFindAll
from harness.runner import load_summary_outcomes, read_json, safe_name, summarize
from harness.statistics import bootstrap_mean


def compare_findall(run_dirs: list[Path]) -> dict:
    """Normalize aligned task counts against this fleet, without rewriting task grades.

    A missing artifact is not a measured failure. Require completed grades or
    recorded errors for every selected task before treating failed tasks as zero.
    """
    if len(run_dirs) < 2:
        raise ValueError("Company FindAll comparison requires at least two run directories")
    if len({p.resolve() for p in run_dirs}) != len(run_dirs):
        raise ValueError("Duplicate run directories in fleet")
    suite = CompanyFindAll()
    tasks_by_id = {task.id: task for task in suite.load()}
    runs = []
    contract = None
    names = set()
    for directory in run_dirs:
        config = read_json(directory / "config.json")
        previous = read_json(directory / "summary.json")
        task_ids = previous.get("task_ids")
        if not task_ids or len(task_ids) != len(set(task_ids)):
            raise ValueError(f"{directory}: missing or invalid task selection; rerun bench summary")
        identity = (
            config["suite"],
            config["suite_revision"],
            config["judge_model"],
            frozenset(task_ids),
        )
        if config["suite"] != suite.name or config["suite_revision"] != suite.revision:
            raise ValueError(
                "Compare requires the current Company FindAll dataset and grader revision"
            )
        if contract is not None and contract != identity:
            raise ValueError(
                "Fleet runs must have identical task IDs, suite revisions, and judge models"
            )
        contract = identity
        name = config["system"]["name"]
        if name in names:
            raise ValueError(f"Duplicate system in fleet: {name}")
        names.add(name)
        if set(task_ids) - tasks_by_id.keys():
            raise ValueError(f"{directory}: task selection contains unknown IDs")
        tasks = [tasks_by_id[key] for key in sorted(task_ids)]
        for task in tasks:
            task_dir = directory / "tasks" / safe_name(task.id)
            if (task_dir / "grade.json").exists():
                complete = (task_dir / "result.json").exists()
            else:
                complete = (task_dir / "error.json").exists()
            if not complete:
                raise ValueError(f"Incomplete task artifacts: {task_dir}")
        outcomes = load_summary_outcomes(directory, suite, tasks)
        runs.append((directory, config, outcomes))
    maxima = [
        max(
            outcomes[i].grade.scores["num_passed"] if outcomes[i].grade else 0.0
            for _, _, outcomes in runs
        )
        for i in range(len(runs[0][2]))
    ]
    fleet = [config["system"] for _, config, _ in runs]
    summaries = []
    for directory, config, outcomes in runs:
        summary = summarize(suite, outcomes, config)
        scores = [
            (outcome.grade.scores["num_passed"] / maximum) if outcome.grade and maximum else 0.0
            for outcome, maximum in zip(outcomes, maxima, strict=True)
        ]
        interval = asdict(bootstrap_mean(scores))
        summary["primary_metric"] = "normalized_num_passed"
        summary["metrics"]["normalized_num_passed"] = interval["estimate"]
        summary["normalized_num_passed_failed_as_zero"] = interval["estimate"]
        summary["confidence_intervals"]["metrics"]["normalized_num_passed"] = interval
        summary["confidence_intervals"]["metrics"]["normalized_num_passed_failed_as_zero"] = (
            interval
        )
        summary["run_directory"] = str(directory)
        summary["comparison_fleet"] = fleet
        summaries.append(summary)
    return {
        "suite": suite.name,
        "primary_metric": "normalized_num_passed",
        "fleet": fleet,
        "task_ids": [o.task.id for o in runs[0][2]],
        "normalization": "Per task: passed / fleet maximum; all-zero tasks and failures score zero",
        "summaries": summaries,
    }

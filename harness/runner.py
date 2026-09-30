"""Resumable evaluation runs: one system over one suite.

Layout of a run directory::

    results/runs/{system}-{suite}[-{suffix}]-{hash}/
        config.json              resolved system spec, suite revision, judge model
        tasks/{task_id}/
            result.json          the system's answer, evidence, costs and trajectory
            grade.json           scores, grader details and judge cost
            error.json           the last failure, when every attempt failed
        summary.json             aggregate metrics and cost

The hash covers the resolved system spec, the suite revision and the judge
model, so rerunning the same command resumes: tasks with a ``grade.json`` are
skipped, tasks with only a ``result.json`` are re-graded without re-running the
system, and failed tasks are retried. Changing any hashed input starts a new
directory. ``--run-suffix`` starts an independent repeat of the same setup.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

from harness.llm.judge import Judge, track_judge_usage
from harness.statistics import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    bootstrap_mean,
)
from harness.suite import Grade, Suite, Task
from harness.systems import System, SystemSpec

DEFAULT_RUNS_ROOT = Path("results/runs")
_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(value: str) -> str:
    """Make a system, suite or task id safe to use as one path component."""
    return _UNSAFE_PATH_CHARS.sub("_", value).strip("_") or "_"


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically so an interrupted run never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def config_hash(spec: SystemSpec, suite: Suite, judge_model: str) -> str:
    """Hash every input that changes results into a short, stable run id."""
    payload = {
        "runner_version": 2,
        "system": spec.to_dict(),
        "suite": suite.name,
        "suite_revision": getattr(suite, "revision", ""),
        "judge_model": judge_model,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:10]


def run_directory(
    root: Path, spec: SystemSpec, suite: Suite, judge_model: str, suffix: str | None
) -> Path:
    parts = [safe_name(spec.name), safe_name(suite.name)]
    if suffix:
        parts.append(safe_name(suffix))
    parts.append(config_hash(spec, suite, judge_model))
    return root / "-".join(parts)


@dataclass
class TaskOutcome:
    task: Task
    result: dict[str, Any] | None
    grade: Grade | None
    error: str | None
    judge_cost_usd: float = 0.0
    judge_cost_known: bool = True


def summarize(suite: Suite, outcomes: list[TaskOutcome], config: dict[str, Any]) -> dict[str, Any]:
    """Aggregate grades and costs; failed tasks count as zero in ``*_failed_as_zero``."""
    graded = [o for o in outcomes if o.grade is not None]
    grades = [o.grade for o in graded if o.grade is not None]
    results = [o.result for o in outcomes if o.result is not None]
    total = len(outcomes)
    metrics = suite.aggregate(grades) if grades else {}
    primary = suite.primary_metric
    primary_sum = sum(g.scores.get(primary, 0.0) for g in grades)
    # All registered suites aggregate per-task means. Failures are excluded
    # from ordinary metrics and included as zero in the primary companion metric.
    intervals = {
        key: asdict(bootstrap_mean([g.scores.get(key, 0.0) for g in grades])) for key in metrics
    }
    intervals[f"{primary}_failed_as_zero"] = asdict(
        bootstrap_mean([o.grade.scores.get(primary, 0.0) if o.grade else 0.0 for o in outcomes])
    )
    cost_known = len(results) == total and all(r.get("cost_known", False) for r in results)
    judge_cost_known = all(
        o.judge_cost_known and (o.grade is None or o.grade.details.get("judge_cost_known", True))
        for o in outcomes
    )
    system_cost = sum(r.get("total_cost_usd", 0.0) for r in results)
    return {
        "system": config["system"]["name"],
        "suite": suite.name,
        "suite_revision": config["suite_revision"],
        "judge_model": config["judge_model"],
        "primary_metric": primary,
        "tasks": total,
        "graded": len(graded),
        "failed": total - len(graded),
        "metrics": metrics,
        f"{primary}_failed_as_zero": primary_sum / total if total else 0.0,
        "confidence_intervals": {
            "confidence_level": CONFIDENCE_LEVEL,
            "method": "percentile",
            "n_resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "sampling_unit": "task",
            "metrics": intervals,
        },
        "cost": {
            "system_usd": system_cost,
            "system_usd_per_task": system_cost / total if total else 0.0,
            "model_usd": sum(r.get("model_cost_usd", 0.0) for r in results),
            "search_usd": sum(r.get("search_cost_usd", 0.0) for r in results),
            "system_cost_known": cost_known,
            "judge_usd": sum(
                o.judge_cost_usd
                if o.judge_cost_usd
                else (o.grade.judge_cost_usd if o.grade else 0.0)
                for o in outcomes
            ),
            "judge_cost_known": judge_cost_known,
        },
        "mean_latency_s": mean(r.get("latency_ms", 0.0) for r in results) / 1000
        if results
        else 0.0,
        "mean_searches": mean(r.get("num_searches", 1) for r in results) if results else 0.0,
        "stop_reasons": dict(Counter(r.get("stop_reason", "single_step") for r in results)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


class RunnableSystem(Protocol):
    async def execute(self, task: Task, suite: Suite) -> dict[str, Any]: ...
    async def close(self) -> None: ...


class Runner:
    """Runs one system over one suite into a resumable run directory."""

    def __init__(
        self,
        spec: SystemSpec,
        suite: Suite,
        *,
        judge: Judge,
        runs_root: Path = DEFAULT_RUNS_ROOT,
        run_suffix: str | None = None,
        concurrency: int = 5,
        max_attempts: int = 2,
        system: RunnableSystem | None = None,
    ) -> None:
        if concurrency < 1 or max_attempts < 1:
            raise ValueError("concurrency and max_attempts must be positive")
        if spec.kind not in suite.system_kinds:
            raise ValueError(f"{suite.name} requires {suite.system_kinds}; got {spec.kind}")
        self.spec = spec
        self.suite = suite
        self.judge = judge
        self.concurrency = concurrency
        self.max_attempts = max_attempts
        self.run_dir = run_directory(runs_root, spec, suite, judge.model, run_suffix)
        self._system = system
        self.config = {
            "system": spec.to_dict(),
            "suite": suite.name,
            "suite_revision": getattr(suite, "revision", ""),
            "judge_model": judge.model,
            "run_suffix": run_suffix,
        }

    def task_dir(self, task: Task) -> Path:
        return self.run_dir / "tasks" / safe_name(task.id)

    async def run(self, tasks: list[Task], *, on_done=None) -> dict[str, Any]:
        """Evaluate ``tasks`` (resuming prior work) and write ``summary.json``."""
        ids = [safe_name(task.id) for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task IDs collide after path normalization")
        config_path = self.run_dir / "config.json"
        if config_path.exists():
            if read_json(config_path)["system"] != self.config["system"]:
                raise RuntimeError(f"{config_path} was written by a different system spec")
        else:
            write_json(
                config_path, {**self.config, "created_at": datetime.now(timezone.utc).isoformat()}
            )

        system = self._system or System(self.spec)
        semaphore = asyncio.Semaphore(self.concurrency)

        async def bounded(task: Task) -> TaskOutcome:
            async with semaphore:
                outcome = await self._run_task(system, task)
            if on_done is not None:
                on_done(outcome)
            return outcome

        try:
            outcomes = await asyncio.gather(*(bounded(task) for task in tasks))
        finally:
            await system.close()
        summary = summarize(self.suite, list(outcomes), self.config)
        write_json(self.run_dir / "summary.json", summary)
        return summary

    async def _run_task(self, system: RunnableSystem, task: Task) -> TaskOutcome:
        directory = self.task_dir(task)
        cost_path = directory / "cost.json"
        costs: dict[str, Any] = (
            read_json(cost_path)
            if cost_path.exists()
            else {
                "judge_cost_usd": 0.0,
                "judge_cost_known": True,
                "system_cost_known": True,
            }
        )
        grade_path, result_path, error_path = (
            directory / "grade.json",
            directory / "result.json",
            directory / "error.json",
        )
        if grade_path.exists() and result_path.exists():
            stored = read_json(grade_path)
            grade = Grade(
                stored["scores"], stored.get("details", {}), stored.get("judge_cost_usd", 0.0)
            )
            return TaskOutcome(
                task,
                read_json(result_path),
                grade,
                None,
                costs["judge_cost_usd"],
                costs["judge_cost_known"],
            )

        prompt = self.suite.prompt(task)
        result = read_json(result_path) if result_path.exists() else None
        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                if result is None:
                    try:
                        result = await system.execute(task, self.suite)
                    except Exception:
                        costs["system_cost_known"] = False
                        write_json(cost_path, costs)
                        raise
                    result = {"task_id": task.id, "prompt": prompt, "attempt": attempt, **result}
                    result["cost_known"] = (
                        result.get("cost_known", False) and costs["system_cost_known"]
                    )
                    write_json(result_path, result)
                with track_judge_usage() as spend:
                    try:
                        grade = await self.suite.grade_result(task, result, self.judge)
                    finally:
                        costs["judge_cost_usd"] += spend.total_usd
                        costs["judge_cost_known"] &= spend.known
                        write_json(cost_path, costs)
            except Exception as error:  # recorded per task; the run continues
                last_error = "".join(traceback.format_exception(error))
                write_json(
                    error_path, {"task_id": task.id, "attempt": attempt, "error": last_error}
                )
                continue
            grade.judge_cost_usd = costs["judge_cost_usd"] or grade.judge_cost_usd
            grade.details["judge_cost_known"] = costs["judge_cost_known"] and grade.details.get(
                "judge_cost_known", True
            )
            write_json(
                grade_path,
                {
                    "task_id": task.id,
                    "scores": grade.scores,
                    "details": grade.details,
                    "judge_cost_usd": grade.judge_cost_usd,
                },
            )
            error_path.unlink(missing_ok=True)
            return TaskOutcome(
                task, result, grade, None, costs["judge_cost_usd"], costs["judge_cost_known"]
            )
        return TaskOutcome(
            task, result, None, last_error, costs["judge_cost_usd"], costs["judge_cost_known"]
        )


def load_summary_outcomes(run_dir: Path, suite: Suite, tasks: list[Task]) -> list[TaskOutcome]:
    """Rebuild task outcomes from a run directory without running anything."""
    outcomes = []
    for task in tasks:
        directory = run_dir / "tasks" / safe_name(task.id)
        result = (
            read_json(directory / "result.json") if (directory / "result.json").exists() else None
        )
        grade = None
        if (directory / "grade.json").exists():
            stored = read_json(directory / "grade.json")
            grade = Grade(
                stored["scores"], stored.get("details", {}), stored.get("judge_cost_usd", 0.0)
            )
        costs = read_json(directory / "cost.json") if (directory / "cost.json").exists() else {}
        outcomes.append(
            TaskOutcome(
                task,
                result,
                grade,
                None if grade else "not graded",
                costs.get("judge_cost_usd", 0.0),
                costs.get("judge_cost_known", True),
            )
        )
    return outcomes

"""The suite contract: what a benchmark provides to the runner.

A suite loads its tasks from a pinned upstream source, turns each task into the
prompt a system receives, and grades the system's final answer. Everything a
suite needs from an LLM judge goes through ``harness.llm.judge.Judge``, so
judge spend is accounted separately from system spend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from harness.llm.judge import Judge


@dataclass(frozen=True)
class Task:
    """One benchmark item.

    ``problem`` is the task text before suite formatting; ``answer`` is the gold
    reference in whatever shape the suite's grader expects.
    """

    id: str
    problem: str
    answer: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Grade:
    """Scores, reasoning, and judge cost for an answer or an individual search result."""

    scores: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)
    judge_cost_usd: float = 0.0


class Suite(ABC):
    """Base class for every benchmark suite."""

    name: str
    description: str
    primary_metric: str
    # Identifies the pinned data and grading contract (source pin + grader version).
    # Bump it whenever the data pin, a prompt, or grading logic changes.
    revision: str = ""
    system_kinds = ("scout", "rag", "agent")
    requires_judge = True
    # Generation settings applied by the CLI; changes belong in the suite revision.
    judge_settings: dict[str, Any] = {}
    # Ratio-of-sums metrics need paired task resampling, not a mean of task ratios.
    ratio_metrics: dict[str, tuple[str, str]] = {}

    def failure_scores(self, result: dict[str, Any] | None) -> dict[str, float] | None:
        """Optional zero-score treatment for failed tasks in ordinary metrics."""
        return None

    def agent_request(self) -> dict[str, Any]:
        """Public instructions/output_schema/output_spec for agents; never gold answers."""
        return {}

    @abstractmethod
    def load(self) -> list[Task]:
        """Return every task in the pinned dataset revision, in a stable order."""

    def prompt(self, task: Task) -> str:
        """Return the exact text sent to the system under test."""
        return task.problem

    async def grade(self, task: Task, response: str, judge: Judge) -> Grade:
        """Grade one final answer against the task's gold reference."""
        raise NotImplementedError(f"{self.name} requires the full result")

    async def grade_result(self, task: Task, result: dict[str, Any], judge: Judge) -> Grade:
        """Grade answer-only suites; retrieval suites override to inspect ranked results."""
        return await self.grade(task, result["answer"], judge)

    def aggregate(self, grades: list[Grade]) -> dict[str, float]:
        """Average every score key across grades; suites override for set metrics."""
        keys = sorted({key for grade in grades for key in grade.scores})
        return {key: mean(grade.scores.get(key, 0.0) for grade in grades) for key in keys}

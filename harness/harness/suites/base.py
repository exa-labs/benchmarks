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
    """A graded answer. ``scores`` always contains the suite's primary metric."""

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

    @abstractmethod
    def load(self) -> list[Task]:
        """Return every task in the pinned dataset revision, in a stable order."""

    def prompt(self, task: Task) -> str:
        """Return the exact text sent to the system under test."""
        return task.problem

    @abstractmethod
    async def grade(self, task: Task, response: str, judge: Judge) -> Grade:
        """Grade one final answer against the task's gold reference."""

    def aggregate(self, grades: list[Grade]) -> dict[str, float]:
        """Average every score key across grades; suites override for set metrics."""
        keys = sorted({key for grade in grades for key in grade.scores})
        return {
            key: mean(grade.scores[key] for grade in grades if key in grade.scores) for key in keys
        }

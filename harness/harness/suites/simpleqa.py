"""SimpleQA: short fact-seeking questions with a single indisputable answer.

Source: OpenAI simple-evals' test set (4,326 questions), pinned by the SHA-256 of
the downloaded CSV. The system receives the bare question, as in simple-evals.

Grading matches the grader Exa uses for its published comparisons: the simple-evals
SimpleQA judge (see ``expected_answer`` for how its message layout and reply format
differ from ``simpleqa_eval.py``). Per-task scores are one-hot ``correct`` /
``incorrect`` / ``not_attempted``; the aggregate adds simple-evals' official
``accuracy_given_attempted`` and ``f1``. An empty answer is NOT_ATTEMPTED without a
judge call.

Paper: https://cdn.openai.com/papers/simpleqa.pdf
"""

from __future__ import annotations

from harness.llm.judge import Judge
from harness.suites.base import Grade, Suite, Task
from harness.suites.data import csv_rows, fetch_verified, require_count
from harness.suites.expected_answer import SIMPLEQA_PROMPTS, judge_correctness, label_scores
from harness.suites.grading import JudgeSpend, is_blank

SOURCE_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
SOURCE_SHA256 = "feee3f7e7db3617e94e8fcf1977b756ec420ef8568f4e0fcbbe0e92e9d5fc032"
ROW_COUNT = 4326

_LABEL_KEYS = ("correct", "incorrect", "not_attempted")


def tasks_from_rows(rows: list[dict[str, str]]) -> list[Task]:
    """Build tasks; ids are the row index at the pinned hash."""
    return [
        Task(
            id=f"simpleqa-{index:04d}",
            problem=row["problem"],
            answer=row["answer"],
            metadata={"upstream_metadata": row["metadata"]},
        )
        for index, row in enumerate(rows)
    ]


def simpleqa_aggregate(correct: float, incorrect: float, not_attempted: float) -> dict[str, float]:
    """simple-evals' SimpleQA summary from the three label rates.

    ``f1`` is the harmonic mean of overall accuracy and accuracy on attempted items.
    """
    attempted = correct + incorrect
    given_attempted = correct / attempted if attempted > 0 else 0.0
    denominator = given_attempted + correct
    f1 = 2 * given_attempted * correct / denominator if denominator > 0 else 0.0
    return {
        "correct": correct,
        "incorrect": incorrect,
        "not_attempted": not_attempted,
        "accuracy_given_attempted": given_attempted,
        "f1": f1,
    }


class SimpleQA(Suite):
    """OpenAI SimpleQA with its three-way judge."""

    name = "simpleqa"
    description = "SimpleQA: 4,326 short fact-seeking questions"
    primary_metric = "correct"
    revision = f"sha256:{SOURCE_SHA256}+grader-v1"

    def load(self) -> list[Task]:
        """Download (or reuse) the pinned CSV."""
        data = fetch_verified(SOURCE_URL, SOURCE_SHA256, "simple_qa_test_set.csv")
        rows = csv_rows(data)
        require_count(self.name, len(rows), ROW_COUNT)
        return tasks_from_rows(rows)

    async def grade(self, task: Task, response: str, judge: Judge) -> Grade:
        """Label the answer CORRECT / INCORRECT / NOT_ATTEMPTED."""
        spend = JudgeSpend()
        if is_blank(response):
            scores = label_scores("NOT_ATTEMPTED")
            return spend.grade({k: scores[k] for k in _LABEL_KEYS}, {"empty_response": True})
        result = await judge_correctness(
            judge,
            SIMPLEQA_PROMPTS,
            question=task.problem,
            target=str(task.answer),
            predicted_answer=response,
            spend=spend,
        )
        scores = label_scores(result.correctness)
        return spend.grade(
            {k: scores[k] for k in _LABEL_KEYS},
            {"label": result.correctness, "reasoning": result.reasoning},
        )

    def aggregate(self, grades: list[Grade]) -> dict[str, float]:
        """Label rates plus ``accuracy_given_attempted`` and the official ``f1``."""
        if not grades:
            return {}
        rates = {
            key: sum(grade.scores[key] for grade in grades) / len(grades) for key in _LABEL_KEYS
        }
        return simpleqa_aggregate(**rates)

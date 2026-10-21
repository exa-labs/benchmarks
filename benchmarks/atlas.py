"""ATLAS: wide research tasks answered as one Markdown table.

Tasks load from the gated Hugging Face dataset ``exa-ai/ATLAS`` at
``data.loaders.ATLAS_REVISION``. ``benchmarks/graders/atlas/README.md`` documents the task
format, the grading steps and the metrics; ``row_f1`` is primary.

Blog: https://exa.ai/blog/atlas-benchmark
"""

from __future__ import annotations

from benchmarks.graders.atlas import TableJudge, TableTask, grade_table
from benchmarks.graders.base import JudgeSpend
from data import loaders
from harness.llm.judge import Judge
from harness.suite import Grade, Suite, Task


class Atlas(Suite):
    """ATLAS with its table grader."""

    name = "atlas"
    description = "ATLAS: 547 wide research tasks answered as a table"
    primary_metric = "row_f1"
    revision = f"{loaders.ATLAS_REVISION}+grader-v1"

    def __init__(self) -> None:
        self._table_judge: TableJudge | None = None

    def load(self) -> list[Task]:
        return loaders.load_atlas()

    async def grade(self, task: Task, response: str, judge: Judge) -> Grade:
        """Score the response table against the task's gold table."""
        # One TableJudge per run, so its verdict cache spans every task.
        if self._table_judge is None or self._table_judge.judge is not judge:
            self._table_judge = TableJudge(judge)
        spend = JudgeSpend()
        scores, details = await grade_table(
            TableTask.from_record(task.answer), response, self._table_judge, spend
        )
        return spend.grade(scores, details)

"""FRAMES: multi-hop factual questions that combine several Wikipedia articles.

Source: Hugging Face ``google/frames-benchmark`` (824 rows, ``test.tsv``) at a pinned
commit. The system receives the ``Prompt`` column verbatim; ``Answer`` is the gold.

Grading matches the grader Exa uses for its published comparisons: the
expected-answer judge with the simple-evals SimpleQA prompt pair (see
``expected_answer``). ``score`` is 1.0 for CORRECT and 0.0 otherwise. The FRAMES
paper instead used its own single-prompt LLM auto-rater; numbers here follow the
SimpleQA grading rules. An empty answer is NOT_ATTEMPTED without a judge call.

Paper: https://arxiv.org/abs/2409.12941
"""

from __future__ import annotations

from benchmarks.base import Grade, Suite, Task
from benchmarks.data import csv_rows, hf_file, require_count
from benchmarks.expected_answer import SIMPLEQA_PROMPTS, judge_correctness, label_scores
from benchmarks.grading import JudgeSpend, is_blank
from harness.llm.judge import Judge

REPO_ID = "google/frames-benchmark"
REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"
FILENAME = "test.tsv"
ROW_COUNT = 824


def tasks_from_rows(rows: list[dict[str, str]]) -> list[Task]:
    """Build tasks; ids come from the upstream row index (the TSV's unnamed first column)."""
    return [
        Task(
            id=f"frames-{int(row['']):03d}",
            problem=row["Prompt"],
            answer=row["Answer"],
            metadata={
                "reasoning_types": row.get("reasoning_types", ""),
                "wiki_links": row.get("wiki_links", ""),
            },
        )
        for row in rows
    ]


class Frames(Suite):
    """Google FRAMES graded by the expected-answer judge."""

    name = "frames"
    description = "FRAMES: 824 multi-hop questions over Wikipedia"
    primary_metric = "score"
    revision = f"{REVISION}+grader-v1"

    def load(self) -> list[Task]:
        """Fetch the pinned TSV through the Hugging Face cache."""
        rows = csv_rows(hf_file(REPO_ID, FILENAME, REVISION).read_bytes(), delimiter="\t")
        require_count(self.name, len(rows), ROW_COUNT)
        return tasks_from_rows(rows)

    async def grade(self, task: Task, response: str, judge: Judge) -> Grade:
        """Label the answer CORRECT / INCORRECT / NOT_ATTEMPTED; ``score`` is CORRECT."""
        spend = JudgeSpend()
        if is_blank(response):
            return spend.grade(label_scores("NOT_ATTEMPTED"), {"empty_response": True})
        result = await judge_correctness(
            judge,
            SIMPLEQA_PROMPTS,
            question=task.problem,
            target=str(task.answer),
            predicted_answer=response,
            spend=spend,
        )
        return spend.grade(
            label_scores(result.correctness),
            {"label": result.correctness, "reasoning": result.reasoning},
        )

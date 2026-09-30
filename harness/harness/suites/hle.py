"""Humanity's Last Exam (HLE): expert-written closed-ended questions, text-only subset.

Source: Hugging Face ``cais/hle`` (gated; accept the terms on the dataset page and
set ``HF_TOKEN`` or run ``hf auth login``) at a pinned commit. Rows with an image are
dropped, leaving the text-only questions. The system receives the question followed
by HLE's official answer-format instructions.

Grading matches the grader Exa uses for its published comparisons: the
expected-answer judge with the agentic prompt pair (see ``expected_answer``), which
is built for long, cited answers from search agents. This differs from HLE's
official ``run_judge_results.py`` judge (extract the final answer, compare yes/no,
report confidence and calibration error): numbers here are accuracy under the
expected-answer rules and no calibration error is computed. An empty answer is
NOT_ATTEMPTED without a judge call.

Paper: https://arxiv.org/abs/2501.14249
"""

from __future__ import annotations

from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import get_token
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from harness.llm.judge import Judge
from harness.suites.base import Grade, Suite, Task
from harness.suites.data import SourceError, hf_file, require_count
from harness.suites.expected_answer import AGENTIC_PROMPTS, judge_correctness, label_scores
from harness.suites.grading import JudgeSpend, is_blank

REPO_ID = "cais/hle"
REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
FILENAME = "data/test-00000-of-00001.parquet"
TEXT_ONLY_ROW_COUNT = 2158
_COLUMNS = ["id", "question", "image", "answer", "answer_type", "category", "raw_subject"]

_ACCESS_HELP = (
    "HLE is a gated Hugging Face dataset: accept its terms at "
    "https://huggingface.co/datasets/cais/hle, then set HF_TOKEN (or run `hf auth login`)."
)

# Source: https://github.com/centerforaisafety/hle/blob/main/hle_eval/run_model_predictions.py
ANSWER_FORMAT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)


def tasks_from_rows(rows: list[dict[str, Any]]) -> list[Task]:
    """Keep text-only rows (empty ``image``) and build tasks keyed by the upstream id."""
    return [
        Task(
            id=row["id"],
            problem=row["question"],
            answer=row["answer"],
            metadata={
                "answer_type": row["answer_type"],
                "category": row["category"],
                "raw_subject": row["raw_subject"],
            },
        )
        for row in rows
        if not row["image"]
    ]


class HLE(Suite):
    """Humanity's Last Exam, text-only, graded by the agentic expected-answer judge."""

    name = "hle"
    description = "Humanity's Last Exam: 2,158 text-only expert questions (gated dataset)"
    primary_metric = "score"
    revision = f"{REVISION}+grader-v1"

    def load(self) -> list[Task]:
        """Fetch the pinned parquet with the caller's Hugging Face token."""
        token = get_token()
        if not token:
            raise SourceError(f"no Hugging Face token found. {_ACCESS_HELP}")
        try:
            path = hf_file(REPO_ID, FILENAME, REVISION, token=token)
        except (GatedRepoError, RepositoryNotFoundError) as error:
            raise SourceError(f"access to {REPO_ID} was refused. {_ACCESS_HELP}") from error
        rows = pq.read_table(path, columns=_COLUMNS).to_pylist()
        tasks = tasks_from_rows(rows)
        require_count(f"{self.name} (text-only)", len(tasks), TEXT_ONLY_ROW_COUNT)
        return tasks

    def prompt(self, task: Task) -> str:
        """The question followed by HLE's Explanation / Answer / Confidence format."""
        return f"{task.problem}\n\n{ANSWER_FORMAT}"

    async def grade(self, task: Task, response: str, judge: Judge) -> Grade:
        """Label the answer CORRECT / INCORRECT / NOT_ATTEMPTED; ``score`` is CORRECT."""
        spend = JudgeSpend()
        if is_blank(response):
            return spend.grade(label_scores("NOT_ATTEMPTED"), {"empty_response": True})
        result = await judge_correctness(
            judge,
            AGENTIC_PROMPTS,
            question=task.problem,
            target=str(task.answer),
            predicted_answer=response,
            spend=spend,
        )
        return spend.grade(
            label_scores(result.correctness),
            {"label": result.correctness, "reasoning": result.reasoning},
        )

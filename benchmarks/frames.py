"""FRAMES: multi-hop factual questions that combine several Wikipedia articles.

Source: Hugging Face ``google/frames-benchmark`` (824 rows, ``test.tsv``) at a pinned
commit. The system receives the ``Prompt`` column verbatim; ``Answer`` is the gold.

Grading matches the grader Exa uses for its published comparisons: the
expected-answer judge with the simple-evals SimpleQA prompt pair. ``score`` is 1.0
for CORRECT and 0.0 otherwise. The FRAMES paper instead used its own single-prompt
LLM auto-rater; numbers here follow the SimpleQA grading rules. An empty answer is
NOT_ATTEMPTED without a judge call.

Paper: https://arxiv.org/abs/2409.12941
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.graders.base import JudgeSpend, is_blank
from benchmarks.graders.rag import CORRECTNESS_SYSTEM_PROMPT
from data import loaders
from harness.llm.judge import Judge
from harness.suite import Grade, Suite, Task

Label = Literal["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]

# Source: https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py (MIT).
SIMPLEQA_GRADER_USER_TEMPLATE = """
Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```


Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()


class CorrectnessResult(BaseModel):
    """The judge's verdict; field order puts the rationale before the label."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        description=(
            "Step-by-step analysis of whether the predicted answer is correct, incorrect, "
            "or not attempted relative to the gold target. Explain your judgment before "
            "assigning the label."
        ),
    )
    correctness: Label = Field(
        description="Whether the predicted answer is correct, incorrect, or not attempted.",
    )


def label_scores(label: Label) -> dict[str, float]:
    """One-hot scores for a label; ``score`` is 1.0 only for CORRECT."""
    return {
        "score": 1.0 if label == "CORRECT" else 0.0,
        "correct": 1.0 if label == "CORRECT" else 0.0,
        "incorrect": 1.0 if label == "INCORRECT" else 0.0,
        "not_attempted": 1.0 if label == "NOT_ATTEMPTED" else 0.0,
    }


class Frames(Suite):
    """Google FRAMES graded by the expected-answer judge."""

    name = "frames"
    description = "FRAMES: 824 multi-hop questions over Wikipedia"
    primary_metric = "score"
    revision = f"{loaders.FRAMES_REVISION}+grader-v1"

    def load(self) -> list[Task]:
        return loaders.load_frames()

    async def grade(self, task: Task, response: str, judge: Judge) -> Grade:
        """Label the answer CORRECT / INCORRECT / NOT_ATTEMPTED; ``score`` is CORRECT."""
        spend = JudgeSpend()
        if is_blank(response):
            return spend.grade(label_scores("NOT_ATTEMPTED"), {"empty_response": True})
        prompt = SIMPLEQA_GRADER_USER_TEMPLATE.format(
            question=task.problem, target=str(task.answer), predicted_answer=response
        )
        result, reply = await judge.complete_json(
            prompt, CorrectnessResult, system=CORRECTNESS_SYSTEM_PROMPT
        )
        spend.add(reply)
        return spend.grade(
            label_scores(result.correctness),
            {"label": result.correctness, "reasoning": result.reasoning},
        )

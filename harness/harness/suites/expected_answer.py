"""Expected-answer grading: CORRECT / INCORRECT / NOT_ATTEMPTED against a gold target.

This is the SimpleQA grading scheme: a judge sorts each answer into one of three
labels using the simple-evals SimpleQA grader, split into a system message (rules
and examples) and a user message (the item). FRAMES uses it. It matches the grader
Exa uses for its published comparisons. Differences from
simple-evals' ``simpleqa_eval.py``: the prompt is split into system and user
messages, and the judge returns JSON with a short rationale before the label
instead of a bare letter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from shared.graders.rag import CORRECTNESS_SYSTEM_PROMPT

from harness.llm.judge import Judge
from harness.suites.grading import JudgeSpend

Label = Literal["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]

# Source: https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py (MIT)
SIMPLEQA_GRADER_SYSTEM_PROMPT = CORRECTNESS_SYSTEM_PROMPT

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


@dataclass(frozen=True)
class PromptPair:
    """A judge system prompt and a user template with {question}/{target}/{predicted_answer}."""

    system: str
    user_template: str


SIMPLEQA_PROMPTS = PromptPair(SIMPLEQA_GRADER_SYSTEM_PROMPT, SIMPLEQA_GRADER_USER_TEMPLATE)


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


async def judge_correctness(
    judge: Judge,
    prompts: PromptPair,
    *,
    question: str,
    target: str,
    predicted_answer: str,
    spend: JudgeSpend,
) -> CorrectnessResult:
    """Ask the judge to label one predicted answer against its gold target."""
    user = prompts.user_template.format(
        question=question, target=target, predicted_answer=predicted_answer
    )
    result, response = await judge.complete_json(user, CorrectnessResult, system=prompts.system)
    spend.add(response)
    return result


def label_scores(label: Label) -> dict[str, float]:
    """One-hot scores for a label; ``score`` is 1.0 only for CORRECT."""
    return {
        "score": 1.0 if label == "CORRECT" else 0.0,
        "correct": 1.0 if label == "CORRECT" else 0.0,
        "incorrect": 1.0 if label == "INCORRECT" else 0.0,
        "not_attempted": 1.0 if label == "NOT_ATTEMPTED" else 0.0,
    }

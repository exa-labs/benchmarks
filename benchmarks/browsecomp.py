"""BrowseComp: short-answer questions that need persistent web browsing.

Source: OpenAI simple-evals' encrypted test set (1,266 questions), pinned by the
SHA-256 of the downloaded CSV. Each row's ``problem`` and ``answer`` are XOR-encrypted
with a key derived from the row's ``canary``; they are decrypted in memory only.

Grading matches the grader Exa uses for its published comparisons, which is the
simple-evals / HLE judge template. The one difference from simple-evals'
``browsecomp_eval.py``: the judge replies with a JSON object holding the same four
fields (``extracted_final_answer``, ``reasoning``, ``correct``, ``confidence``)
instead of free text scraped with a regex. An empty answer is scored 0 without a
judge call.

Paper: https://openai.com/index/browsecomp/
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from benchmarks.graders.base import JudgeSpend, is_blank
from data import loaders
from harness.llm.judge import Judge
from harness.suite import Grade, Suite, Task

# Source: https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py (MIT)
QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

# Source: https://github.com/centerforaisafety/hle/blob/main/hle_eval/run_judge_results.py,
# as used by https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.
""".strip()  # noqa: E501


class ExtractedAnswer(BaseModel):
    """The judge's structured verdict, in the template's field order."""

    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int


class BrowseComp(Suite):
    """OpenAI BrowseComp, graded by the HLE-style extract-and-compare judge."""

    name = "browsecomp"
    description = "BrowseComp: 1,266 hard-to-find short answers that require browsing"
    primary_metric = "score"
    revision = f"sha256:{loaders.BROWSECOMP_SHA256}+grader-v1"

    def load(self) -> list[Task]:
        return loaders.load_browsecomp()

    def prompt(self, task: Task) -> str:
        """The question wrapped in the official Explanation / Exact Answer / Confidence format."""
        return QUERY_TEMPLATE.format(Question=task.problem)

    async def grade(self, task: Task, response: str, judge: Judge) -> Grade:
        """Extract the final answer from the response and compare it with the gold answer."""
        spend = JudgeSpend()
        if is_blank(response):
            return spend.grade({"score": 0.0}, {"empty_response": True})
        prompt = GRADER_TEMPLATE.format(
            question=task.problem, response=response, correct_answer=task.answer
        )
        verdict, reply = await judge.complete_json(prompt, ExtractedAnswer)
        spend.add(reply)
        return spend.grade(
            {
                "score": 1.0 if verdict.correct == "yes" else 0.0,
                "confidence": float(verdict.confidence),
            },
            verdict.model_dump(),
        )

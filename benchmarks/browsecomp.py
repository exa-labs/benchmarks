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

import base64
import hashlib
from typing import Literal

from pydantic import BaseModel

from benchmarks.base import Grade, Suite, Task
from benchmarks.data import csv_rows, fetch_verified, require_count
from benchmarks.grading import JudgeSpend, is_blank
from harness.llm.judge import Judge

SOURCE_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
SOURCE_SHA256 = "7b24471cd5b3eb2a46830a14802b5c029ea62f488ff75a0f88af7923d1454abf"
ROW_COUNT = 1266

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


def derive_key(password: str, length: int) -> bytes:
    """Repeat SHA-256(password) to ``length`` bytes, as simple-evals does."""
    digest = hashlib.sha256(password.encode()).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt one base64 XOR-encrypted BrowseComp field."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key, strict=True)).decode()


def encrypt(plaintext: str, password: str) -> str:
    """Inverse of ``decrypt``; used to build synthetic rows in tests."""
    raw = plaintext.encode()
    key = derive_key(password, len(raw))
    return base64.b64encode(bytes(a ^ b for a, b in zip(raw, key, strict=True))).decode()


def tasks_from_rows(rows: list[dict[str, str]]) -> list[Task]:
    """Decrypt source rows into tasks; ids are the row index at the pinned hash."""
    return [
        Task(
            id=f"browsecomp-{index:04d}",
            problem=decrypt(row["problem"], row["canary"]),
            answer=decrypt(row["answer"], row["canary"]),
            metadata={"problem_topic": row["problem_topic"]},
        )
        for index, row in enumerate(rows)
    ]


class BrowseComp(Suite):
    """OpenAI BrowseComp, graded by the HLE-style extract-and-compare judge."""

    name = "browsecomp"
    description = "BrowseComp: 1,266 hard-to-find short answers that require browsing"
    primary_metric = "score"
    revision = f"sha256:{SOURCE_SHA256}+grader-v1"

    def load(self) -> list[Task]:
        """Download (or reuse) the pinned CSV and decrypt every row in memory."""
        data = fetch_verified(SOURCE_URL, SOURCE_SHA256, "browse_comp_test_set.csv")
        rows = csv_rows(data)
        require_count(self.name, len(rows), ROW_COUNT)
        return tasks_from_rows(rows)

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

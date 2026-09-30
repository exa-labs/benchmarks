"""DeepSearchQA (DSQA): research prompts whose answers are single values or exhaustive sets.

Source: Hugging Face ``google/deepsearchqa`` (900 rows, ``DSQA-full.csv``) at a pinned
commit. The system receives the ``problem`` verbatim.

Grading follows the paper's LLM-as-a-judge method and matches the grader Exa uses
for its published comparisons: the judge marks each expected answer part found or
missing and lists excessive answers, and precision / recall / F1 are computed from
those counts. ``f1`` is the primary metric (for single answers it is 1.0 only when
the answer is found with nothing extra). The paper's categorical outcomes
(``fully_correct``, ``fully_incorrect``, ``partially_correct``,
``correct_with_extraneous``) are reported alongside.

Notes on the data: four rows have the literal gold answer "None"; the CSV is read
as text so that answer is kept rather than turned into a missing value. A judge
reply that is not the expected JSON is retried, then fails the grade. An empty
answer is scored 0 on every metric without a judge call.

Dataset: https://huggingface.co/datasets/google/deepsearchqa
Paper: https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from harness.llm.judge import Judge
from harness.suites.base import Grade, Suite, Task
from harness.suites.data import csv_rows, hf_file, require_count
from harness.suites.grading import JudgeSpend, complete_parsed, is_blank

REPO_ID = "google/deepsearchqa"
REVISION = "b2623f8653065c2672de6d941fc5434cd652376c"
FILENAME = "DSQA-full.csv"
ROW_COUNT = 900

# Source: DeepSearchQA paper, Appendix A (link in the module docstring)
JUDGE_PROMPT = """Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**
* **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
* **Process:**
  * Identify the "Prompt Type": "{prompt_type}".
  * Refer to the "Correct Answer": "{answer}".
  * Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
  * **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
  * **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
* **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
* **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
  * For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
* **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.

**Output Format:**
Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: "Answer Correctness". Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for "Answer Correctness" should be a dictionary containing "Explanation" (a string), "Correctness Details" (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and "Excessive Answers" (a list of strings indicating the excessive answers).
Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.

**Example (Partial):**
```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true
    }},
    "Excessive Answers": ["Italy"]
  }}
}}
```

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>

-------------------

** Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>

-------------------

AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

-------------------
Rating:"""  # noqa: E501


@dataclass(frozen=True)
class JudgeRating:
    """The parsed "Answer Correctness" object of one judge reply."""

    explanation: str
    correctness_details: dict[str, bool]
    excessive_answers: list[str]


def parse_rating(text: str) -> JudgeRating:
    """Parse a judge reply, raising ``ValueError`` with context when it is malformed."""
    fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    bare = None if fenced else re.search(r"\{.*\}", text, re.DOTALL)
    if fenced is None and bare is None:
        raise ValueError(f"no JSON in judge reply: {text[:200]!r}")
    raw = fenced.group(1) if fenced else bare.group(0)  # type: ignore[union-attr]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"judge JSON did not parse ({error}): {raw[:200]!r}") from error
    if not isinstance(parsed, dict) or not isinstance(parsed.get("Answer Correctness"), dict):
        raise ValueError(f"judge JSON lacks an 'Answer Correctness' object: {raw[:200]!r}")
    body = parsed["Answer Correctness"]
    for key in ("Explanation", "Correctness Details", "Excessive Answers"):
        if key not in body:
            raise ValueError(f"judge JSON lacks {key!r}: {raw[:200]!r}")
    explanation = body["Explanation"]
    details = body["Correctness Details"]
    excessive = body["Excessive Answers"]
    if not isinstance(explanation, str):
        raise ValueError(f"judge Explanation is not a string: {raw[:200]!r}")
    if not isinstance(details, dict) or not details:
        raise ValueError(f"judge Correctness Details is not a non-empty object: {raw[:200]!r}")
    if any(not isinstance(found, bool) for found in details.values()):
        raise ValueError(f"judge Correctness Details values must be booleans: {raw[:200]!r}")
    if not isinstance(excessive, list) or any(not isinstance(a, str) for a in excessive):
        raise ValueError(f"judge Excessive Answers is not a list of strings: {raw[:200]!r}")
    return JudgeRating(explanation, details, excessive)


def rating_scores(rating: JudgeRating, answer_type: str) -> dict[str, float]:
    """Precision / recall / F1 and the paper's categorical outcomes for one rating."""
    num_expected = len(rating.correctness_details)
    num_found = sum(1 for found in rating.correctness_details.values() if found is True)
    num_excessive = len(rating.excessive_answers)

    precision = num_found / (num_found + num_excessive) if num_found + num_excessive else 0.0
    if num_expected > 0:
        recall = num_found / num_expected
    else:
        recall = 1.0 if num_found == 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    all_found = num_found == num_expected
    fully_correct = 1.0 if all_found and num_excessive == 0 else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "exact_match": fully_correct if answer_type == "Single Answer" else 0.0,
        "fully_correct": fully_correct,
        "fully_incorrect": 1.0 if num_found == 0 else 0.0,
        "partially_correct": 1.0 if 0 < num_found < num_expected else 0.0,
        "correct_with_extraneous": 1.0 if all_found and num_excessive > 0 else 0.0,
        "num_expected": float(num_expected),
        "num_found": float(num_found),
        "num_excessive": float(num_excessive),
    }


def empty_scores() -> dict[str, float]:
    """Scores for an empty answer: nothing found, nothing extra."""
    return {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "exact_match": 0.0,
        "fully_correct": 0.0,
        "fully_incorrect": 1.0,
        "partially_correct": 0.0,
        "correct_with_extraneous": 0.0,
        "num_found": 0.0,
        "num_excessive": 0.0,
    }


def tasks_from_rows(rows: list[dict[str, str]]) -> list[Task]:
    """Build tasks; ids are the row index at the pinned revision."""
    return [
        Task(
            id=f"dsqa-{index:03d}",
            problem=row["problem"],
            answer=row["answer"],
            metadata={
                "answer_type": row["answer_type"],
                "problem_category": row["problem_category"],
            },
        )
        for index, row in enumerate(rows)
    ]


class DeepSearchQA(Suite):
    """Google DeepSearchQA with the paper's set-answer judge."""

    name = "dsqa"
    description = "DeepSearchQA: 900 research prompts with single or exhaustive set answers"
    primary_metric = "f1"
    revision = f"{REVISION}+grader-v1"

    def load(self) -> list[Task]:
        """Fetch the pinned CSV through the Hugging Face cache."""
        rows = csv_rows(hf_file(REPO_ID, FILENAME, REVISION).read_bytes())
        require_count(self.name, len(rows), ROW_COUNT)
        return tasks_from_rows(rows)

    async def grade(self, task: Task, response: str, judge: Judge) -> Grade:
        """Rate which expected parts the response contains and score set overlap."""
        answer_type = str(task.metadata["answer_type"])
        spend = JudgeSpend()
        if is_blank(response):
            return spend.grade(empty_scores(), {"empty_response": True, "answer_type": answer_type})
        prompt = JUDGE_PROMPT.format(
            prompt=task.problem,
            prompt_type=answer_type,
            answer=task.answer,
            response=response,
        )
        rating, _ = await complete_parsed(
            judge, prompt, parse_rating, spend, what=f"dsqa rating for {task.id}"
        )
        details: dict[str, Any] = {
            "answer_type": answer_type,
            "explanation": rating.explanation,
            "correctness_details": rating.correctness_details,
            "excessive_answers": rating.excessive_answers,
        }
        return spend.grade(rating_scores(rating, answer_type), details)

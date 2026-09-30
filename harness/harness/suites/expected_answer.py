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

from harness.llm.judge import Judge
from harness.suites.grading import JudgeSpend

Label = Literal["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]

# Source: https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py (MIT)
SIMPLEQA_GRADER_SYSTEM_PROMPT = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k".
    - Predicted answers "120k", "124k", and 115k" are all CORRECT.
    - Predicted answers "100k" and "113k" are INCORRECT.
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".
""".strip()

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

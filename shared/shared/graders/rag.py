import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

import tiktoken
from pydantic import BaseModel, Field, model_validator

from .base import BaseLLMGrader, GradeResult

logger = logging.getLogger(__name__)

_TIKTOKEN_ENC = tiktoken.get_encoding("o200k_base")


RAG_GRADING_SYSTEM = """You are evaluating if an extracted answer matches the expected answer for a company fact query.
This is BINARY - score 1 if the answer is correct, score 0 if it's wrong.

AUTOMATIC SCORE 0 (no exceptions):
1. Answer is "unknown", "not found", "N/A", or empty -> Score 0
2. Answer contradicts the expected answer -> Score 0

For text answers (founding year, location, industry, YC batch, founders):
- Score 1 if the extracted answer is semantically equivalent to the expected answer
- Accept reasonable variations (e.g., "San Francisco" = "SF" = "San Francisco, CA")
- For founders, accept if all expected founders are mentioned (order doesn't matter)

For numeric answers (employees, funding):
- Score 1 if within 20% of expected value
- Accept different formats (e.g., "$10M" = "10 million" = "10,000,000")

For yes/no verification questions:
- Score 1 only if the answer matches exactly

When genuinely uncertain about a close match, lean toward score 1 if the core answer aligns."""

RAG_GRADING_USER = """Question: {query}
Expected answer: {expected}

Extracted answer: {actual}"""


class RAGGradeResult(BaseModel):
    explanation: str
    score: float = Field(..., ge=0.0, le=1.0)


class RAGGrader(BaseLLMGrader):
    async def grade(
        self,
        query: str,
        expected_answer: str,
        actual_answer: str,
        bucket: str = "",
    ) -> GradeResult:
        if not actual_answer or actual_answer.lower() in [
            "unknown",
            "not found",
            "n/a",
        ]:
            return GradeResult(scores={"is_correct": 0.0})

        try:
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": RAG_GRADING_SYSTEM},
                    {
                        "role": "user",
                        "content": RAG_GRADING_USER.format(
                            query=query,
                            expected=expected_answer,
                            actual=actual_answer,
                        ),
                    },
                ],
                response_format=RAGGradeResult,
            )
            parsed = response.choices[0].message.parsed
            assert parsed is not None
            return GradeResult(
                scores={"is_correct": 1.0 if parsed.score >= 0.5 else 0.0}
            )
        except Exception as e:
            logger.warning(f"RAG grading failed: {e}")
            return GradeResult(scores={"is_correct": 0.0})


# ---------------------------------------------------------------------------
# GroundedRAGGrader — two-axis evaluation (correctness + groundedness)
# ---------------------------------------------------------------------------

CORRECTNESS_SYSTEM_PROMPT = """
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


GROUNDEDNESS_SYSTEM_PROMPT = """You are a retrieval quality auditor. Given a question, a gold target answer, and a set of retrieved citations, determine whether the retrieved citations contain sufficient information to derive the gold target answer.

This is NOT about evaluating a synthesized answer — you are evaluating whether the search engine retrieved sources that contain the answer.

The retrieved citations are numbered [1], [2], etc. You MUST reason step-by-step using the exact structure below.

STEP 1 — EVIDENCE
Quote the specific passage(s) from the retrieved citations that contain information relevant to the gold target answer.
Copy the exact text. If you cannot find any supporting passage, write "NO EVIDENCE FOUND".

STEP 2 — GROUNDEDNESS
Based on Step 1, do the retrieved citations contain ALL the non-trivial information in the gold target answer?

The bar is HIGH. The retrieved sources must contain ALL the non-trivial factual claims from the gold target. Apply these definitions strictly:

- GROUNDED: every non-trivial factual claim in the gold target answer is present in the retrieved sources
- PARTIAL: every non-trivial factual claim from the gold target is present in the retrieved sources, AND the gold target also includes trivial supplementary details that don't appear verbatim — things like basic programming syntax, standard formatting, universally-known facts (e.g., "water is H2O"), or explanatory scaffolding. The key: PARTIAL is ONLY for trivial additions alongside a fully-grounded core.
- UNGROUNDED: ANY non-trivial factual claim from the gold target answer is missing from the retrieved sources. If even ONE specific fact, name, number, or entity from the gold target is absent from the sources, the verdict is UNGROUNDED.

Examples of trivial (ok for PARTIAL): basic syntax, formatting, "Python uses indentation", country names for well-known cities, unit conversions.
Examples of non-trivial (makes it UNGROUNDED if missing from sources): specific entity names, measurements, rankings, dates, API function signatures, statistics.

STEP 3 — SOURCE ATTRIBUTION
Which citation(s) contain information sufficient to derive the gold target answer?
Consider the citations as a set — for multi-hop questions, the answer may require combining information from multiple sources. List the 1-indexed citation numbers.
If none contain the gold target answer, return an empty list."""


@dataclass
class Citation:
    url: str
    title: str
    text: str


class CorrectnessResult(BaseModel):
    reasoning: str = Field(
        ...,
        description="Step-by-step analysis before assigning the label.",
    )
    correctness: Literal["CORRECT", "INCORRECT", "NOT_ATTEMPTED"] = Field(
        ...,
        description="Whether the predicted answer is correct, incorrect, or not attempted.",
    )


class GroundednessResult(BaseModel):
    evidence: str = Field(
        ...,
        description=(
            "Direct quote(s) from retrieved citations relevant to the gold target. "
            "If no supporting passage exists, write 'NO EVIDENCE FOUND'."
        ),
    )
    reasoning: str = Field(
        ...,
        description="Step-by-step analysis of groundedness.",
    )
    groundedness: Literal["GROUNDED", "PARTIAL", "UNGROUNDED"] = Field(
        ...,
        description="Whether the citations ground the gold target answer.",
    )
    source_indices: list[int] = Field(
        ...,
        description="1-indexed citation numbers that contain the answer.",
    )

    @model_validator(mode="after")
    def _enforce_source_consistency(self) -> "GroundednessResult":
        if self.groundedness != "UNGROUNDED" and not self.source_indices:
            self.groundedness = "UNGROUNDED"
        return self


class GroundedRAGGrader(BaseLLMGrader):
    def __init__(
        self,
        model: str = "gpt-5.4",
        temperature: float = 0.0,
        api_key: str | None = None,
    ):
        super().__init__(model=model, temperature=temperature, api_key=api_key)

    async def grade(
        self,
        question: str,
        expected_answer: str,
        predicted_answer: str,
        citations: list[Citation],
    ) -> GradeResult:
        if not predicted_answer:
            return GradeResult(scores={"score": 0.0, "grounded": 0.0})
        if not expected_answer:
            return GradeResult(scores={"score": 0.0, "grounded": 0.0})

        (corr_result, grnd_result) = await asyncio.gather(
            self._call_correctness(question, expected_answer, predicted_answer),
            self._call_groundedness(question, expected_answer, citations),
        )

        score = 1.0 if corr_result.correctness == "CORRECT" else 0.0
        grounded = 0.0 if grnd_result.groundedness == "UNGROUNDED" else 1.0

        matching_urls = self._extract_source_urls(grnd_result.source_indices, citations)
        num_citations = len(citations)
        citation_precision = (
            float(len(matching_urls)) / num_citations if num_citations > 0 else 0.0
        )

        citation_token_counts = [len(_TIKTOKEN_ENC.encode(c.text)) for c in citations]
        avg_citation_tokens = (
            sum(citation_token_counts) / len(citation_token_counts)
            if citation_token_counts
            else 0.0
        )
        total_citation_tokens = float(sum(citation_token_counts))

        return GradeResult(
            scores={
                "score": score,
                "grounded": grounded,
                "citation_precision": round(citation_precision, 3),
                "avg_citation_tokens": round(avg_citation_tokens, 1),
                "total_citation_tokens": total_citation_tokens,
            },
            details={
                "correctness": corr_result.correctness,
                "correctness_reasoning": corr_result.reasoning,
                "groundedness": grnd_result.groundedness,
                "groundedness_reasoning": grnd_result.reasoning,
                "evidence": grnd_result.evidence,
                "answer_source_urls": matching_urls,
            },
        )

    async def _call_correctness(
        self, question: str, expected_answer: str, predicted_answer: str
    ) -> CorrectnessResult:
        user_content = (
            f"Question: {question}\n\n"
            f"Gold target: {expected_answer}\n\n"
            f"Predicted answer: {predicted_answer}"
        )
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": CORRECTNESS_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format=CorrectnessResult,
        )
        parsed = response.choices[0].message.parsed
        assert parsed is not None
        return parsed

    async def _call_groundedness(
        self, question: str, expected_answer: str, citations: list[Citation]
    ) -> GroundednessResult:
        parts = []
        for i, c in enumerate(citations, start=1):
            parts.append(f"[{i}] URL: {c.url}\nTitle: {c.title}\n{c.text}")
        contents = "\n\n---\n\n".join(parts)

        user_content = (
            f"Question: {question}\n\n"
            f"Retrieved citations:\n{contents}\n\n"
            f"Gold target answer: {expected_answer}"
        )
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": GROUNDEDNESS_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format=GroundednessResult,
        )
        parsed = response.choices[0].message.parsed
        assert parsed is not None
        return parsed

    @staticmethod
    def _extract_source_urls(source_indices: list[int], citations: list[Citation]) -> list[str]:
        valid = [idx - 1 for idx in source_indices if 1 <= idx <= len(citations)]
        return [citations[i].url for i in sorted(set(valid))]

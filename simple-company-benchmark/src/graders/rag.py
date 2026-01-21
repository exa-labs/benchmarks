"""
RAG grader for company search benchmark.

Evaluates fact extraction accuracy for company attributes:
- Static facts (founding year, location, industry): exact match
- Dynamic facts (employees, funding): 20% tolerance

Header author: Devin
"""

import logging
import re

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from .base import GradeResult

logger = logging.getLogger(__name__)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    return re.sub(r"\s+", " ", answer.lower().strip())


def extract_number(text: str) -> float | None:
    """Extract numeric value from text."""
    if not text:
        return None
    numbers = re.findall(r"[\d,]+\.?\d*", text.replace(",", ""))
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            return None
    return None


def is_numeric_match(expected: str, actual: str, tolerance: float = 0.2) -> bool:
    """Check if two numeric values match within tolerance."""
    expected_num = extract_number(expected)
    actual_num = extract_number(actual)

    if expected_num is None or actual_num is None:
        return False

    if expected_num == 0:
        return actual_num == 0

    diff = abs(expected_num - actual_num) / expected_num
    return diff <= tolerance


RAG_GRADING_SYSTEM = """You are evaluating if an extracted answer matches the expected answer for a company fact query.

For text answers (founding year, location, industry, YC batch, founders):
- Score 1 if the extracted answer is semantically equivalent to the expected answer
- Accept reasonable variations (e.g., "San Francisco" = "SF" = "San Francisco, CA")
- For founders, accept if all expected founders are mentioned (order doesn't matter)

For numeric answers (employees, funding):
- Score 1 if within 20% of expected value
- Accept different formats (e.g., "$10M" = "10 million" = "10,000,000")

For yes/no verification questions:
- Score 1 only if the answer matches exactly

Score 0 if:
- The answer is wrong or contradicts the expected answer
- The answer is missing or says "unknown"/"not found"
- Cannot determine the answer from the response"""

RAG_GRADING_USER = """Question: {query}
Expected answer: {expected}

Extracted answer: {actual}"""


class RAGGradeResult(BaseModel):
    explanation: str
    score: float = Field(..., ge=0.0, le=1.0)


class RAGGrader:
    """Grader for RAG track queries.

    Evaluates extracted answers against expected answers using:
    - Exact matching for simple facts
    - Numeric tolerance for dynamic facts
    - LLM-based semantic matching for complex answers
    """

    def __init__(
        self, model: str = "gpt-4.1", temperature: float = 0.0, api_key: str | None = None
    ):
        self.model = model
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=api_key)

    async def grade(
        self,
        query: str,
        expected_answer: str,
        actual_answer: str,
        bucket: str = "",
    ) -> GradeResult:
        expected_norm = normalize_answer(expected_answer)
        actual_norm = normalize_answer(actual_answer)

        if not actual_norm or actual_norm in ["unknown", "not found", "n/a"]:
            return GradeResult(
                scores={"is_correct": 0.0}, details={"method": "empty_answer"}
            )

        if expected_norm == actual_norm:
            return GradeResult(
                scores={"is_correct": 1.0}, details={"method": "exact_match"}
            )

        if bucket in ["employees", "funding", "funding_amount"]:
            if is_numeric_match(expected_answer, actual_answer, tolerance=0.2):
                return GradeResult(
                    scores={"is_correct": 1.0}, details={"method": "numeric_tolerance"}
                )

        return await self._grade_with_llm(query, expected_answer, actual_answer)

    async def _grade_with_llm(
        self, query: str, expected: str, actual: str
    ) -> GradeResult:
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
                            expected=expected,
                            actual=actual,
                        ),
                    },
                ],
                response_format=RAGGradeResult,
            )
            parsed = response.choices[0].message.parsed
            assert parsed is not None
            return GradeResult(
                scores={"is_correct": 1.0 if parsed.score >= 0.5 else 0.0},
                details={"method": "llm_semantic", "explanation": parsed.explanation},
            )
        except Exception as e:
            logger.warning(f"RAG grading failed: {e}")
            return GradeResult(scores={"is_correct": 0.0}, details={"error": str(e)})

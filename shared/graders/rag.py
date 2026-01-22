import logging

from pydantic import BaseModel, Field

from .base import BaseLLMGrader, GradeResult

logger = logging.getLogger(__name__)


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

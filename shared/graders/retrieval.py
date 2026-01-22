import logging

from pydantic import BaseModel, Field

from .base import BaseLLMGrader, GradeResult
from .utils import url_matches
from ..searchers import SearchResult

logger = logging.getLogger(__name__)


RETRIEVAL_GRADING_SYSTEM = """You are evaluating if a search result matches a company search query.
This is BINARY - score 1 if the result matches, score 0 if it doesn't.

AUTOMATIC SCORE 0 (no exceptions):
1. Job listing pages (URLs with /jobs/, /careers/) -> Score 0
2. News articles about the company (not the company's own page) -> Score 0
3. If content is empty/missing and cannot verify the company -> Score 0

For queries with constraints (industry, geography, founding year, etc.):
- Score 1 if the result is about a company that matches ALL query constraints
- For industry/geo queries: company must be in the specified industry AND location
- For founded_year queries: company must be founded in the specified year
- For employee_count queries: company has approximately the specified count (within 20% tolerance)
- For funding queries: company matches the funding stage or amount criteria

Score 0 if:
- The result doesn't match ANY of the constraints
- The result is not about a company
- Cannot verify the company matches from available content

Be strict about matching ALL constraints. Partial matches = 0.
When genuinely uncertain about a close match, lean toward score 1 if the core criteria align."""

RETRIEVAL_GRADING_USER = """Query: {query}
Constraints: {constraints}

Result URL: {url}
Title: {title}

{text}"""


class RetrievalGradeResult(BaseModel):
    explanation: str
    score: float = Field(..., ge=0.0, le=1.0)


class RetrievalGrader(BaseLLMGrader):
    async def grade(
        self,
        query: str,
        result: SearchResult,
        gold_homepage: str | None = None,
        constraints: dict | None = None,
    ) -> GradeResult:
        if gold_homepage:
            is_match = url_matches(result.url, gold_homepage)
            return GradeResult(scores={"is_match": 1.0 if is_match else 0.0})

        if constraints:
            return await self._grade_with_llm(query, result, constraints)

        return GradeResult(scores={"is_match": 0.0})

    async def _grade_with_llm(
        self, query: str, result: SearchResult, constraints: dict
    ) -> GradeResult:
        try:
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": RETRIEVAL_GRADING_SYSTEM},
                    {
                        "role": "user",
                        "content": RETRIEVAL_GRADING_USER.format(
                            query=query,
                            constraints=constraints,
                            url=result.url,
                            title=result.title,
                            text=result.text[:30000] if result.text else "(no content)",
                        ),
                    },
                ],
                response_format=RetrievalGradeResult,
            )
            parsed = response.choices[0].message.parsed
            assert parsed is not None
            return GradeResult(scores={"is_match": 1.0 if parsed.score >= 0.5 else 0.0})
        except Exception as e:
            logger.warning(f"Retrieval grading failed: {e}")
            return GradeResult(scores={"is_match": 0.0})

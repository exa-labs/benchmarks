from pydantic import BaseModel, Field

from harness.searchers import SearchResult
from harness.suite import Grade

from .base import BaseLLMGrader
from .utils import url_matches

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
    ) -> Grade:
        """Match a known homepage deterministically, or judge company constraints."""
        if gold_homepage:
            return Grade(scores={"is_match": float(url_matches(result.url, gold_homepage))})
        if not constraints:
            return Grade(scores={"is_match": 0.0})
        parsed = await self.parse(
            RETRIEVAL_GRADING_SYSTEM,
            RETRIEVAL_GRADING_USER.format(
                query=query,
                constraints=constraints,
                url=result.url,
                title=result.title,
                text=result.content[:30000] or "(no content)",
            ),
            RetrievalGradeResult,
        )
        return Grade(
            scores={"is_match": float(parsed.score >= 0.5)},
            details={"explanation": parsed.explanation},
        )

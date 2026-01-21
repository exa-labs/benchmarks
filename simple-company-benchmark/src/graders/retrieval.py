"""
Retrieval grader for company search benchmark.

Evaluates whether search results contain the expected company based on:
- gold_company_homepage: Check if the homepage URL appears in results
- constraints: Validate returned companies against structured filters (rule-based)

Header author: Devin
"""

import logging
import re
from urllib.parse import urlparse

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from ..searchers import SearchResult
from .base import GradeResult

logger = logging.getLogger(__name__)


def normalize_url(url: str) -> str:
    """Normalize URL for comparison by removing protocol, www, and trailing slashes."""
    if not url:
        return ""
    parsed = urlparse(url.lower())
    domain = parsed.netloc or parsed.path
    domain = re.sub(r"^www\.", "", domain)
    path = parsed.path.rstrip("/") if parsed.netloc else ""
    return f"{domain}{path}".rstrip("/")


def url_matches(result_url: str, gold_url: str) -> bool:
    """Check if result URL matches or contains the gold company homepage."""
    result_normalized = normalize_url(result_url)
    gold_normalized = normalize_url(gold_url)

    if not result_normalized or not gold_normalized:
        return False

    return gold_normalized in result_normalized or result_normalized in gold_normalized


RETRIEVAL_GRADING_SYSTEM = """You are evaluating if a search result matches a company search query.

For queries with constraints (industry, geography, founding year, etc.), evaluate if the result satisfies ALL specified constraints.

Score 1 if:
- The result is about a company that matches ALL query constraints
- For industry/geo queries: company is in the specified industry AND location
- For founded_year queries: company was founded in the specified year
- For employee_count queries: company has approximately the specified employee count (within 20% tolerance)
- For funding queries: company matches the funding stage or amount criteria

Score 0 if:
- The result doesn't match ANY of the constraints
- The result is not about a company
- Cannot verify the company matches from available content

Be strict about matching ALL constraints. Partial matches = 0."""

RETRIEVAL_GRADING_USER = """Query: {query}
Constraints: {constraints}

Result URL: {url}
Title: {title}
Content: {text}"""


class RetrievalGradeResult(BaseModel):
    explanation: str
    score: float = Field(..., ge=0.0, le=1.0)


class RetrievalGrader:
    """Grader for retrieval track queries.

    For queries with gold_company_homepage, uses URL matching.
    For queries with constraints, uses LLM-based evaluation.
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
        result: SearchResult,
        gold_homepage: str | None = None,
        constraints: dict | None = None,
    ) -> GradeResult:
        if gold_homepage:
            is_match = url_matches(result.url, gold_homepage)
            return GradeResult(
                scores={"is_match": 1.0 if is_match else 0.0},
                details={"method": "url_match", "gold_homepage": gold_homepage},
            )

        if constraints:
            return await self._grade_with_constraints(query, result, constraints)

        return GradeResult(scores={"is_match": 0.0}, details={"error": "no_grading_criteria"})

    async def _grade_with_constraints(
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
                            text=result.text[:2000] if result.text else "(no content)",
                        ),
                    },
                ],
                response_format=RetrievalGradeResult,
            )
            parsed = response.choices[0].message.parsed
            assert parsed is not None
            return GradeResult(
                scores={"is_match": 1.0 if parsed.score >= 0.5 else 0.0},
                details={"method": "llm_constraints", "explanation": parsed.explanation},
            )
        except Exception as e:
            logger.warning(f"Retrieval grading failed: {e}")
            return GradeResult(scores={"is_match": 0.0}, details={"error": str(e)})

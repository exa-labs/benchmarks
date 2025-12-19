import asyncio
import logging

from openai import AsyncOpenAI, RateLimitError
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ..searchers import SearchResult
from .base import GradeResult

logger = logging.getLogger(__name__)

PEOPLE_ROLE_GRADING_SYSTEM = """You are evaluating if a LinkedIn profile satisfies a job role search query.
This is BINARY - score 1 if the profile matches, score 0 if it doesn't.
AUTOMATIC SCORE 0 (no exceptions):
1. Job listing pages (URLs with /jobs/, titles like "500 Jobs in NYC") → Score 0
2. Company pages (not a personal profile) → Score 0
3. If content is empty/missing and title doesn't clearly show the role → Score 0
4. Role abbreviations are job titles, NOT names:
   - "TAM" = Technical Account Manager, NOT a person named "Tam"
   - "Tax Manager" does NOT match "TAM" query
ROLE EQUIVALENCE RULES:
- Accept reasonable role variations within the SAME function:
  - "Security Engineer" ≈ "System Security Engineer" ≈ "Application Security Engineer" ✓
  - "Head of X" ≈ "Director of X" ≈ "VP of X" (leadership equivalence) ✓
  - "ML Engineer" ≈ "Machine Learning Engineer" ✓
- Do NOT accept different functions:
  - "UX Designer" ≠ "Head of UX" (IC vs leadership)
  - "Data Analyst" ≠ "Data Engineer" (different function)
  - "Project Manager" ≠ "Product Manager" (different function)
Score 1 if:
- Job function matches (with equivalences above)
- Location matches if specified (same metro area is fine)
- The person currently holds this role in this location, or if they hold this role and their primary profile is in this location
- It's a real personal LinkedIn profile
Score 0 if:
- Wrong job function
- Wrong location
- Cannot verify the role from available content
- Job listing or company page
When genuinely uncertain about a close match, lean toward score 1 if the core job function aligns."""

PEOPLE_ROLE_GRADING_USER = """Query: {query}
Result: URL: {url}
Title: {title}

{text}"""


class PeopleGradeResult(BaseModel):
    explanation: str
    score: float = Field(..., ge=0.0, le=1.0)


class PeopleGrader:
    def __init__(
        self, model: str = "gpt-4.1", temperature: float = 0.0, api_key: str | None = None
    ):
        self.model = model
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=api_key)

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=60),
    )
    async def _grade_with_retry(self, query: str, result: SearchResult) -> GradeResult:
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": PEOPLE_ROLE_GRADING_SYSTEM},
                {
                    "role": "user",
                    "content": PEOPLE_ROLE_GRADING_USER.format(
                        query=query,
                        url=result.url,
                        title=result.title,
                        text=result.text or "(no content)",
                    ),
                },
            ],
            response_format=PeopleGradeResult,
        )
        parsed = response.choices[0].message.parsed
        assert parsed is not None
        return GradeResult(scores={"is_match": 1.0 if parsed.score >= 0.5 else 0.0})

    async def grade(self, query: str, result: SearchResult) -> GradeResult:
        try:
            return await self._grade_with_retry(query, result)
        except Exception as e:
            # Only log non-rate-limit errors, rate limits are handled by retry
            if not isinstance(e, RateLimitError):
                logger.warning(f"People grading failed: {e}")
            # Return 0 for any persistent failures after retries
            return GradeResult(scores={"is_match": 0.0})

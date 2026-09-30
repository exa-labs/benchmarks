"""Shared judge contracts, parsing retries, and per-grade cost accounting."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast

from pydantic import BaseModel

from harness.llm.judge import Judge, JudgeOutputError, JudgeResponse
from harness.suite import Grade

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")
PARSE_ATTEMPTS = 3


async def gather_judgments(*calls: Awaitable[R]) -> list[R]:
    """Finish every submitted call before raising, so failed batches retain all spend."""
    results = await asyncio.gather(*calls, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return cast(list[R], results)


class StructuredJudge(Protocol):
    async def complete_json(
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> tuple[T, Any]: ...


class BaseLLMGrader:
    def __init__(self, judge: StructuredJudge):
        self.judge = judge

    async def parse(self, system: str, prompt: str, schema: type[T]) -> T:
        """Use the run's judge; errors propagate to the resumable task runner."""
        parsed, _ = await self.judge.complete_json(prompt, schema, system=system)
        return parsed


@dataclass
class JudgeSpend:
    """Running total of judge cost for one grade.

    A response with an unknown price (``cost_usd is None``) adds nothing to the
    total but marks the whole grade's cost as unknown.
    """

    total_usd: float = 0.0
    known: bool = True
    calls: int = 0

    def add(self, response: JudgeResponse) -> None:
        """Account for one judge response."""
        self.calls += 1
        if response.cost_usd is None:
            self.known = False
        else:
            self.total_usd += response.cost_usd

    def grade(self, scores: dict[str, float], details: dict[str, Any] | None = None) -> Grade:
        """Build a ``Grade`` carrying this spend."""
        merged = dict(details or {})
        merged["judge_calls"] = self.calls
        if not self.known:
            merged["judge_cost_known"] = False
        return Grade(scores=scores, details=merged, judge_cost_usd=self.total_usd)


def is_blank(response: str) -> bool:
    """Whether a system produced no answer text at all."""
    return not response or not response.strip()


async def complete_parsed(
    judge: Judge,
    prompt: str,
    parse: Callable[[str], R],
    spend: JudgeSpend,
    *,
    what: str,
    attempts: int = PARSE_ATTEMPTS,
) -> tuple[R, str]:
    """Ask the judge for free text and parse it, retrying unparseable replies.

    ``parse`` raises ``ValueError`` on a malformed reply. Returns the parsed value
    and the raw reply text; raises ``JudgeOutputError`` naming ``what`` once every
    attempt has failed.
    """
    last_error: ValueError | None = None
    for _ in range(attempts):
        response = await judge.complete(prompt)
        spend.add(response)
        try:
            return parse(response.text), response.text
        except ValueError as error:
            last_error = error
    raise JudgeOutputError(
        f"{what}: no parseable judge reply after {attempts} attempts: {last_error}"
    )

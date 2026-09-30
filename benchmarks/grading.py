"""Grading helpers shared by every suite: judge spend accounting and parse retries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from benchmarks.base import Grade
from harness.llm.judge import Judge, JudgeOutputError, JudgeResponse

_T = TypeVar("_T")

# Free-text judge replies that must parse (DeepSearchQA ratings, WideSearch
# alignments) are retried this many times in total before the grade fails.
PARSE_ATTEMPTS = 3


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
    parse: Callable[[str], _T],
    spend: JudgeSpend,
    *,
    what: str,
    attempts: int = PARSE_ATTEMPTS,
) -> tuple[_T, str]:
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

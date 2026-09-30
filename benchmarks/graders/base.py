"""Shared grade types and a provider-independent structured judge contract."""

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, cast

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")


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


@dataclass
class GradeResult:
    scores: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)


class BaseGradeOutput(BaseModel):
    explanation: str
    score: float = Field(..., ge=0.0, le=1.0)


class BaseLLMGrader:
    def __init__(self, judge: StructuredJudge):
        self.judge = judge

    async def parse(self, system: str, prompt: str, schema: type[T]) -> T:
        """Use the run's judge; errors propagate to the resumable task runner."""
        parsed, _ = await self.judge.complete_json(prompt, schema, system=system)
        return parsed

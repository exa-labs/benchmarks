from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field


@dataclass
class GradeResult:
    scores: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)


class BaseGradeOutput(BaseModel):
    explanation: str
    score: float = Field(..., ge=0.0, le=1.0)


class BaseLLMGrader:
    def __init__(
        self,
        model: str = "gpt-4.1",
        temperature: float = 0.0,
        api_key: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=api_key)

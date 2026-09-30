"""LLM judge used by suite graders.

A judge is a plain model client with a fixed model and generation settings, plus
two conveniences graders need: free-text completion and schema-validated JSON
completion. Its spend is reported on every response so the runner can keep
judge cost apart from the cost of the system under test.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from harness.llm import LLM_DEFAULT
from harness.llm.clients import ModelClient, create_client
from harness.llm.types import Usage

_T = TypeVar("_T", bound=BaseModel)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeResponse:
    """One judge call's raw text, token usage, and USD cost (``None`` if unpriced)."""

    text: str
    usage: Usage
    cost_usd: float | None


@dataclass
class JudgeUsage:
    total_usd: float = 0.0
    known: bool = True
    calls: int = 0


_ACTIVE_USAGE: ContextVar[JudgeUsage | None] = ContextVar("judge_usage", default=None)


@contextmanager
def track_judge_usage():
    """Track all judge calls for a task, including parsing retries and failed grades."""
    usage = JudgeUsage()
    token = _ACTIVE_USAGE.set(usage)
    try:
        yield usage
    finally:
        _ACTIVE_USAGE.reset(token)


class JudgeOutputError(ValueError):
    """The judge did not return JSON matching the requested schema."""


class Judge:
    """A fixed-model grader client."""

    def __init__(
        self,
        model: str = LLM_DEFAULT,
        *,
        reasoning_effort: str | None = None,
        max_output_tokens: int = 16_000,
        json_attempts: int = 2,
        client: ModelClient | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.json_attempts = json_attempts
        self._client = client

    @property
    def client(self) -> ModelClient:
        if self._client is None:
            self._client = create_client(self.model)
        return self._client

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        response_schema: dict | None = None,
    ) -> JudgeResponse:
        """Return the judge's free-text reply to one prompt."""
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        usage = _ACTIVE_USAGE.get()
        try:
            generation = await self.client.generate(
                messages,
                max_output_tokens=self.max_output_tokens,
                reasoning_effort=self.reasoning_effort,
                response_schema=response_schema,
            )
        except Exception:
            if usage is not None:
                usage.known = False
            raise
        if usage is not None:
            usage.calls += 1
            usage.total_usd += generation.cost_usd or 0.0
            usage.known &= generation.cost_usd is not None
        return JudgeResponse(generation.text, generation.usage, generation.cost_usd)

    async def complete_json(
        self,
        prompt: str,
        schema: type[_T],
        *,
        system: str | None = None,
    ) -> tuple[_T, JudgeResponse]:
        """Return the judge's reply parsed into ``schema``.

        Clients with structured outputs constrain the reply to the schema; others get
        the JSON schema appended to the prompt. A reply that does not validate is
        retried up to ``json_attempts`` times in total; the returned response's usage
        and cost cover every attempt.
        """
        json_schema = schema.model_json_schema()
        if self.client.supports_response_schema:
            instruction = prompt
            response_schema = {"name": schema.__name__, "schema": strict_schema(json_schema)}
        else:
            instruction = (
                f"{prompt}\n\nRespond with only a JSON object matching this JSON schema; it "
                f"replaces any other reply format requested above:\n{json.dumps(json_schema)}"
            )
            response_schema = None
        usage = Usage()
        cost: float | None = 0.0
        last_error: Exception | None = None
        for _ in range(self.json_attempts):
            response = await self.complete(
                instruction, system=system, response_schema=response_schema
            )
            usage = usage + response.usage
            cost = None if cost is None or response.cost_usd is None else cost + response.cost_usd
            try:
                parsed = schema.model_validate(_extract_json(response.text))
            except (ValueError, ValidationError) as error:
                last_error = error
                continue
            return parsed, JudgeResponse(response.text, usage, cost)
        raise JudgeOutputError(f"judge returned no valid {schema.__name__}: {last_error}")


def strict_schema(schema: dict) -> dict:
    """Make a Pydantic JSON schema acceptable to strict structured outputs.

    Strict mode requires every object to list all of its properties as required and
    to forbid additional properties. Without strict mode the model may ignore the
    schema and follow a reply format written in the prompt instead.
    """

    def walk(node: object) -> object:
        if isinstance(node, dict):
            node = {key: walk(value) for key, value in node.items()}
            properties = node.get("properties")
            if node.get("type") == "object" and isinstance(properties, dict):
                node["additionalProperties"] = False
                node["required"] = list(properties)
            return node
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    result = walk(schema)
    assert isinstance(result, dict)
    return result


def _extract_json(text: str) -> object:
    """Parse the first JSON object in a reply, tolerating surrounding prose or fences."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(text)
        if match is None:
            raise ValueError("no JSON object in judge reply") from None
        return json.loads(match.group(0))

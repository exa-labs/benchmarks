"""Drop-in AsyncOpenAI-compatible client backed by Perplexity Agent API.

Routes completions through /v1/agent with max_steps=1 (no search loop).
Implements the subset of the AsyncOpenAI interface used by BaseLLMGrader
and SimpleRAGAgent:
  - client.beta.chat.completions.parse(model, messages, response_format)
  - client.chat.completions.create(model, messages)

Model names without a provider prefix (e.g. "gpt-5.4") are auto-prefixed
with "openai/" when forwarded to the Agent API.
"""

import asyncio
import json
import re
from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

PPLX_AGENT_URL = "https://api.perplexity.ai/v1/agent"


class _Message:
    def __init__(self, content: str, parsed: Any = None):
        self.content = content
        self.parsed = parsed


class _Choice:
    def __init__(self, message: _Message):
        self.message = message


class _Response:
    def __init__(self, choices: list[_Choice]):
        self.choices = choices


class _BetaChatCompletions:
    def __init__(self, parent: "PerplexityAgentLLMClient"):
        self._p = parent

    async def parse(
        self,
        model: str,
        messages: list[dict],
        response_format: Type[T],
        temperature: float = 0.0,
        **kwargs,
    ) -> _Response:
        schema = response_format.model_json_schema()
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "schema": schema,
                "strict": True,
            },
        }
        text = await self._p._call(model, messages, response_format=rf)

        parsed = None
        try:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            raw = m.group(1) if m else text
            parsed = response_format.model_validate_json(raw)
        except Exception:
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                parsed = response_format.model_validate_json(text[start:end])
            except Exception:
                pass

        return _Response(choices=[_Choice(_Message(content=text, parsed=parsed))])


class _ChatCompletions:
    def __init__(self, parent: "PerplexityAgentLLMClient"):
        self._p = parent

    async def create(self, model: str, messages: list[dict], **kwargs) -> _Response:
        text = await self._p._call(model, messages)
        return _Response(choices=[_Choice(_Message(content=text))])


class _BetaChat:
    def __init__(self, parent: "PerplexityAgentLLMClient"):
        self.completions = _BetaChatCompletions(parent)


class _Beta:
    def __init__(self, parent: "PerplexityAgentLLMClient"):
        self.chat = _BetaChat(parent)


class _Chat:
    def __init__(self, parent: "PerplexityAgentLLMClient"):
        self.completions = _ChatCompletions(parent)


class PerplexityAgentLLMClient:
    """AsyncOpenAI-compatible wrapper backed by Perplexity /v1/agent."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._http = httpx.AsyncClient(timeout=120.0)
        self.beta = _Beta(self)
        self.chat = _Chat(self)

    async def _call(
        self,
        model: str,
        messages: list[dict],
        response_format: dict | None = None,
    ) -> str:
        # Auto-prefix bare model names for Agent API.
        if model and "/" not in model:
            model = f"openai/{model}"

        input_items = [
            {"type": "message", "role": m["role"], "content": m["content"]}
            for m in messages
        ]

        payload: dict[str, Any] = {
            "model": model,
            "max_steps": 1,
            "input": input_items,
        }
        if response_format:
            payload["response_format"] = response_format

        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = await self._http.post(
                    PPLX_AGENT_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503) and attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError):
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise

        for item in (data.get("output") or []):
            if item.get("type") == "message" and item.get("role") == "assistant":
                for block in item.get("content") or []:
                    if block.get("type") == "output_text":
                        return block.get("text", "")
        return ""

    async def close(self):
        await self._http.aclose()

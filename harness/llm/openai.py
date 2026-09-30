"""OpenAI Responses API client.

Every call re-sends the whole conversation (no server-side state): prior
function calls and outputs are replayed as native ``function_call`` /
``function_call_output`` items. Provider-hosted web search is the Responses
``web_search`` tool; its calls, sources and citations are normalized into
``HostedSearch`` so the Scout loop can account for them like its own searches.
"""

from __future__ import annotations

import re
from typing import Any

import openai
from openai import AsyncOpenAI

from harness.llm.clients import ModelClient
from harness.llm.pricing import token_cost
from harness.llm.types import ContextWindowExceeded, Generation, HostedSearch, ToolCall, Usage

_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5", "gpt-6")
_CONTEXT_ERROR = re.compile(r"context_length_exceeded|maximum context length|context window", re.I)


def supports_reasoning(model_name: str) -> bool:
    """Reasoning models accept ``reasoning.effort`` and reject ``temperature``."""
    return model_name.startswith(_REASONING_MODEL_PREFIXES)


def web_search_tool(options: dict[str, Any]) -> dict[str, Any]:
    """Encode the Responses ``web_search`` tool from harness options.

    Recognized options: ``search_context_size`` (default ``medium``),
    ``allowed_domains``, ``excluded_domains``, ``user_location``.
    """
    tool: dict[str, Any] = {
        "type": "web_search",
        "search_context_size": options.get("search_context_size") or "medium",
    }
    filters = {}
    if options.get("allowed_domains"):
        filters["allowed_domains"] = options["allowed_domains"]
    if options.get("excluded_domains"):
        filters["blocked_domains"] = options["excluded_domains"]
    if filters:
        tool["filters"] = filters
    if options.get("user_location"):
        tool["user_location"] = options["user_location"]
    return tool


def to_responses_input(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Convert a Chat Completions conversation to Responses ``instructions`` + ``input``."""
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "system":
            instructions.append(message["content"])
        elif role == "user":
            items.append({"role": "user", "content": message["content"]})
        elif role == "assistant":
            if message.get("content"):
                items.append({"role": "assistant", "content": message["content"]})
            for call in message.get("tool_calls") or []:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    }
                )
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
        else:
            raise ValueError(f"unsupported message role {role!r}")
    return "\n\n".join(instructions), items


def parse_response(response: dict[str, Any]) -> tuple[str, list[ToolCall], HostedSearch, list[str]]:
    """Extract final text, function calls, hosted-search activity and reasoning summaries.

    The final text is the ``output_text`` of message items after the last
    ``web_search_call``, so narration the model emitted between searches is not
    mistaken for the answer.
    """
    output = response.get("output") or []
    last_search = max(
        (index for index, item in enumerate(output) if item.get("type") == "web_search_call"),
        default=-1,
    )
    texts: list[str] = []
    all_texts: list[str] = []
    tool_calls: list[ToolCall] = []
    search = HostedSearch()
    seen_urls: dict[str, dict[str, str]] = {}
    reasoning: list[str] = []

    def add_source(url: str | None, title: str = "", text: str = "") -> None:
        if not url:
            return
        source = seen_urls.get(url)
        if source is None:
            source = {"url": url, "title": title or "", "text": text or ""}
            seen_urls[url] = source
            search.sources.append(source)
        else:
            source["title"] = source["title"] or title or ""
            source["text"] = source["text"] or text or ""

    for index, item in enumerate(output):
        kind = item.get("type")
        if kind == "function_call":
            tool_calls.append(ToolCall(item["call_id"], item["name"], item.get("arguments") or ""))
        elif kind == "web_search_call":
            search.num_searches += 1
            action = item.get("action") or {}
            for query in [action.get("query"), *(action.get("queries") or [])]:
                if query and query not in search.queries:
                    search.queries.append(query)
            for source in action.get("sources") or []:
                add_source(source.get("url"), source.get("title", ""))
            if action.get("type") in ("open_page", "find_in_page"):
                add_source(action.get("url"))
        elif kind == "message":
            for part in item.get("content") or []:
                if part.get("type") != "output_text":
                    continue
                all_texts.append(part.get("text") or "")
                if index > last_search:
                    texts.append(part.get("text") or "")
                for annotation in part.get("annotations") or []:
                    if annotation.get("type") == "url_citation":
                        add_source(annotation.get("url"), annotation.get("title", ""))
        elif kind == "reasoning":
            for summary in item.get("summary") or []:
                if summary.get("text"):
                    reasoning.append(summary["text"])

    text = "".join(texts)
    if not text.strip():
        text = "".join(all_texts)
    return text, tool_calls, search, reasoning


def parse_usage(response: dict[str, Any]) -> Usage:
    """Read Responses usage; cached tokens are a subset of input tokens."""
    usage = response.get("usage") or {}
    return Usage(
        input_tokens=usage.get("input_tokens") or 0,
        output_tokens=usage.get("output_tokens") or 0,
        cached_input_tokens=(usage.get("input_tokens_details") or {}).get("cached_tokens") or 0,
        reasoning_tokens=(usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0,
    )


class OpenAIClient(ModelClient):
    """``openai/<model>`` through the Responses API."""

    provider = "openai"
    supports_response_schema = True

    def __init__(
        self,
        model: str,
        *,
        timeout: float = 900.0,
        max_retries: int = 3,
        client: AsyncOpenAI | None = None,
    ) -> None:
        super().__init__(model)
        self._client = client or AsyncOpenAI(timeout=timeout, max_retries=max_retries)

    def build_request(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        hosted_web_search: dict[str, Any] | None,
        max_output_tokens: int | None,
        temperature: float | None,
        reasoning_effort: str | None,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the ``responses.create`` keyword arguments for one call."""
        instructions, items = to_responses_input(messages)
        api_tools = [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["parameters"],
                **({"strict": tool["strict"]} if "strict" in tool else {}),
            }
            for tool in tools or []
        ]
        if hosted_web_search is not None:
            api_tools.append(web_search_tool(hosted_web_search))
        request: dict[str, Any] = {"model": self.model_name, "input": items}
        if instructions:
            request["instructions"] = instructions
        if api_tools:
            request["tools"] = api_tools
        if hosted_web_search is not None:
            request["include"] = ["web_search_call.action.sources"]
        if max_output_tokens is not None:
            request["max_output_tokens"] = max_output_tokens
        if supports_reasoning(self.model_name):
            if reasoning_effort:
                request["reasoning"] = {"effort": reasoning_effort}
        elif temperature is not None:
            request["temperature"] = temperature
        if response_schema is not None:
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": response_schema["name"],
                    "schema": response_schema["schema"],
                    "strict": True,
                }
            }
        return request

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        hosted_web_search: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> Generation:
        request = self.build_request(
            messages,
            tools=tools,
            hosted_web_search=hosted_web_search,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            response_schema=response_schema,
        )
        try:
            response = (await self._client.responses.create(**request)).model_dump(mode="json")
        except openai.BadRequestError as error:
            if _CONTEXT_ERROR.search(str(error)):
                raise ContextWindowExceeded(str(error)) from error
            raise
        if response.get("status") == "failed":
            raise RuntimeError(f"OpenAI response failed: {response.get('error')}")
        text, tool_calls, search, reasoning = parse_response(response)
        usage = parse_usage(response)
        return Generation(
            text=text,
            tool_calls=tool_calls,
            hosted_search=search if hosted_web_search is not None else None,
            usage=usage,
            cost_usd=token_cost(self.model, usage, cached_included_in_input=True),
            reasoning=reasoning,
            raw=response,
        )

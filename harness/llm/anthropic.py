"""Anthropic Messages API client.

Conversations arrive Chat Completions-shaped and are converted on every call:
assistant ``tool_calls`` become ``tool_use`` blocks and consecutive ``tool``
messages become one user message of ``tool_result`` blocks. Provider-hosted web
search is Anthropic's server-side ``web_search`` tool; a ``pause_turn`` stop is
continued until the model finishes its turn, and every continuation's searches
and tokens are folded into one ``Generation``.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from harness.llm.clients import ModelClient
from harness.llm.pricing import token_cost
from harness.llm.types import ContextWindowExceeded, Generation, HostedSearch, ToolCall, Usage

WEB_SEARCH_TOOL_TYPE = "web_search_20260318"
DEFAULT_MAX_TOKENS = 32_000
MAX_PAUSE_TURN_CONTINUATIONS = 20
_CONTEXT_ERROR = re.compile(r"prompt is too long|context window|too many tokens", re.I)
_SERVER_TOOL_BLOCKS = ("server_tool_use", "web_search_tool_result")


def web_search_tool(options: dict[str, Any]) -> dict[str, Any]:
    """Encode the server-side ``web_search`` tool from harness options.

    Recognized options: ``max_uses`` (default 15 searches per model turn),
    ``allowed_domains``, ``excluded_domains``, ``user_location``.
    """
    tool: dict[str, Any] = {
        "type": options.get("tool_type") or WEB_SEARCH_TOOL_TYPE,
        "name": "web_search",
        # Newer tool revisions default to code-execution callers; the harness
        # calls web search directly from the model turn.
        "allowed_callers": ["direct"],
        "max_uses": options.get("max_uses", 15),
    }
    if options.get("allowed_domains"):
        tool["allowed_domains"] = options["allowed_domains"]
    if options.get("excluded_domains"):
        tool["blocked_domains"] = options["excluded_domains"]
    if options.get("user_location"):
        tool["user_location"] = options["user_location"]
    return tool


def sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt array bounds Anthropic rejects (``minItems`` > 1) into the description."""
    sanitized = copy.deepcopy(schema)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "array" and value.get("minItems", 0) > 1:
                minimum = value["minItems"]
                maximum = value.pop("maxItems", None)
                if maximum == minimum:
                    bound = f"Exactly {minimum} items must be provided."
                elif maximum is not None:
                    bound = f"Between {minimum} and {maximum} items must be provided."
                else:
                    bound = f"At least {minimum} items must be provided."
                description = value.get("description", "")
                if bound.lower() not in description.lower():
                    value["description"] = f"{description.rstrip()} {bound}".strip()
                value["minItems"] = 1
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(sanitized)
    return sanitized


def to_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Convert a Chat Completions conversation to Anthropic ``system`` + ``messages``."""
    system: list[str] = []
    converted: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message["role"]
        if role == "system":
            system.append(message["content"])
            index += 1
        elif role == "assistant" and message.get("tool_calls"):
            blocks: list[dict[str, Any]] = []
            if message.get("content"):
                blocks.append({"type": "text", "text": message["content"]})
            for call in message["tool_calls"]:
                try:
                    arguments = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "input": arguments if isinstance(arguments, dict) else {},
                    }
                )
            converted.append({"role": "assistant", "content": blocks})
            index += 1
        elif role == "tool":
            results = []
            while index < len(messages) and messages[index]["role"] == "tool":
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": messages[index]["tool_call_id"],
                        "content": messages[index]["content"],
                    }
                )
                index += 1
            converted.append({"role": "user", "content": results})
        elif role in ("user", "assistant"):
            converted.append({"role": role, "content": message["content"]})
            index += 1
        else:
            raise ValueError(f"unsupported message role {role!r}")
    return "\n\n".join(system), converted


def final_text(content: list[dict[str, Any]]) -> str:
    """Return the text after the last server-tool block, or all text if there is none.

    Anthropic splits one answer across several text blocks (citation-bearing and
    connective prose), so every trailing text block is concatenated.
    """
    last_tool = max(
        (index for index, block in enumerate(content) if block.get("type") in _SERVER_TOOL_BLOCKS),
        default=-1,
    )
    tail = "".join(b.get("text") or "" for b in content[last_tool + 1 :] if b.get("type") == "text")
    if tail.strip():
        return tail
    return "".join(block.get("text") or "" for block in content if block.get("type") == "text")


def parse_search(content: list[dict[str, Any]], search: HostedSearch) -> None:
    """Fold one response's server-side searches, results and citations into ``search``."""
    by_url = {source["url"]: source for source in search.sources}

    def add_source(url: str | None, title: str = "", text: str = "") -> None:
        if not url:
            return
        source = by_url.get(url)
        if source is None:
            source = {"url": url, "title": title or "", "text": text or ""}
            by_url[url] = source
            search.sources.append(source)
        else:
            source["title"] = source["title"] or title or ""
            source["text"] = source["text"] or text or ""

    for block in content:
        kind = block.get("type")
        if kind == "server_tool_use" and block.get("name") == "web_search":
            query = (block.get("input") or {}).get("query")
            if query:
                search.queries.append(query)
        elif kind == "web_search_tool_result":
            results = block.get("content")
            if isinstance(results, list):
                for entry in results:
                    add_source(entry.get("url"), entry.get("title", ""))
        elif kind == "text":
            for citation in block.get("citations") or []:
                add_source(
                    citation.get("url"), citation.get("title", ""), citation.get("cited_text", "")
                )


def replayable_content(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepare a paused turn's blocks for replay as the next request's assistant message.

    Some gateways flatten a block's ``caller`` object to the string ``"direct"``;
    the API only accepts the object form on input.
    """
    blocks = copy.deepcopy(content)
    for block in blocks:
        if block.get("caller") == "direct":
            block["caller"] = {"type": "direct"}
    return blocks


def parse_usage(response: dict[str, Any]) -> tuple[Usage, int]:
    """Read token usage and the executed web-search count from one response."""
    usage = response.get("usage") or {}
    server = usage.get("server_tool_use") or {}
    return (
        Usage(
            input_tokens=usage.get("input_tokens") or 0,
            output_tokens=usage.get("output_tokens") or 0,
            cached_input_tokens=usage.get("cache_read_input_tokens") or 0,
            cache_creation_tokens=usage.get("cache_creation_input_tokens") or 0,
        ),
        server.get("web_search_requests") or 0,
    )


class AnthropicClient(ModelClient):
    """``anthropic/<model>`` through the Messages API (streamed, to allow long turns)."""

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        timeout: float = 900.0,
        max_retries: int = 3,
        client: AsyncAnthropic | None = None,
    ) -> None:
        super().__init__(model)
        self._client = client or AsyncAnthropic(timeout=timeout, max_retries=max_retries)

    def build_request(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        hosted_web_search: dict[str, Any] | None,
        max_output_tokens: int | None,
        temperature: float | None,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        """Build the ``messages.stream`` keyword arguments for one call."""
        system, converted = to_messages(messages)
        api_tools = [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": sanitize_schema(tool["parameters"]),
                **({"strict": tool["strict"]} if "strict" in tool else {}),
            }
            for tool in tools or []
        ]
        if hosted_web_search is not None:
            api_tools.append(web_search_tool(hosted_web_search))
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": converted,
            "max_tokens": max_output_tokens or DEFAULT_MAX_TOKENS,
        }
        if system:
            request["system"] = system
        if api_tools:
            request["tools"] = api_tools
        if temperature is not None:
            request["temperature"] = temperature
        if reasoning_effort:
            request["output_config"] = {"effort": reasoning_effort}
        return request

    async def _stream(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            async with self._client.messages.stream(**request) as stream:
                message = await stream.get_final_message()
        except anthropic.BadRequestError as error:
            if _CONTEXT_ERROR.search(str(error)):
                raise ContextWindowExceeded(str(error)) from error
            raise
        return message.model_dump(mode="json", exclude_none=True)

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
        del response_schema  # not enforced here; the judge falls back to prompting
        request = self.build_request(
            messages,
            tools=tools,
            hosted_web_search=hosted_web_search,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        responses: list[dict[str, Any]] = []
        conversation = list(request["messages"])
        for _ in range(MAX_PAUSE_TURN_CONTINUATIONS + 1):
            response = await self._stream({**request, "messages": conversation})
            responses.append(response)
            if response.get("stop_reason") != "pause_turn":
                break
            conversation = [
                *conversation,
                {"role": "assistant", "content": replayable_content(response["content"])},
            ]
        else:
            raise RuntimeError(
                f"Anthropic server-tool turn exceeded {MAX_PAUSE_TURN_CONTINUATIONS} continuations"
            )

        usage = Usage()
        search = HostedSearch()
        tool_calls: list[ToolCall] = []
        reasoning: list[str] = []
        for response in responses:
            response_usage, searches = parse_usage(response)
            usage = usage + response_usage
            search.num_searches += searches
            parse_search(response.get("content") or [], search)
            for block in response.get("content") or []:
                if block.get("type") == "tool_use":
                    tool_calls.append(
                        ToolCall(block["id"], block["name"], json.dumps(block.get("input") or {}))
                    )
                elif block.get("type") == "thinking" and block.get("thinking"):
                    reasoning.append(block["thinking"])
        search.num_searches = max(search.num_searches, len(search.queries))
        return Generation(
            text=final_text(responses[-1].get("content") or []),
            tool_calls=tool_calls,
            hosted_search=search if hosted_web_search is not None else None,
            usage=usage,
            cost_usd=token_cost(self.model, usage, cached_included_in_input=False),
            reasoning=reasoning,
            raw=responses if len(responses) > 1 else responses[0],
        )

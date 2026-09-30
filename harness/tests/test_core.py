"""Contract tests for the harness core: searchers, model clients, Scout, catalog, runner.

Everything runs offline. Provider HTTP is served by ``httpx.MockTransport`` and
models by scripted fake clients, so these tests pin request shapes, loop
semantics and run-directory behavior without API keys.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from harness.llm import anthropic as anthropic_client
from harness.llm import openai as openai_client
from harness.llm.clients import ModelClient
from harness.llm.judge import Judge
from harness.llm.pricing import model_price, token_cost
from harness.llm.types import ContextWindowExceeded, Generation, HostedSearch, ToolCall, Usage
from harness.runner import Runner
from harness.scout import (
    BUDGET_WARNING,
    FINAL_SYNTHESIS_PROMPT,
    HOSTED_SEARCH_CONTINUATION,
    NO_TOOL_NUDGE,
    Scout,
    ScoutConfig,
    StopReason,
    truncate_oldest_exchange,
)
from harness.suites.base import Grade, Suite, Task
from harness.systems import Catalog, System
from harness.tools import OBJECTIVE_SEARCH_TOOL, QUERY_SEARCH_TOOL, SearchTool
from pydantic import BaseModel
from shared.searchers import (
    BraveSearcher,
    ExaSearcher,
    ParallelSearcher,
    PerplexitySearcher,
    Searcher,
    SearchResponse,
    SearchResult,
)

# --------------------------------------------------------------------------- helpers


def mock_client(handler, **kwargs) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)


class FakeSearcher(Searcher):
    """Returns one canned result per query and records every call."""

    name = "fake"

    def __init__(self, *, batches: bool = False, delay: float = 0.0) -> None:
        self.batches_queries = batches
        self.delay = delay
        self.calls: list[dict[str, Any]] = []

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        return (await self.run(query, num_results)).results

    async def run(self, query, num_results=10, *, objective=None, search_queries=None):
        self.calls.append(
            {"query": query, "objective": objective, "search_queries": search_queries}
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        return SearchResponse(
            results=[
                SearchResult(
                    url=f"https://example.com/{query}", title=query, highlights=[f"about {query}"]
                )
            ],
            cost_usd=0.005,
            queries=search_queries or [query],
        )


class ScriptedClient(ModelClient):
    """Plays back scripted generations (or exceptions) and records every request."""

    provider = "openai"
    supports_response_schema = True

    def __init__(
        self, script: list[Generation | Exception], model: str = "openai/gpt-5.6-luna"
    ) -> None:
        super().__init__(model)
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    async def generate(self, messages, **kwargs) -> Generation:
        self.requests.append({"messages": [dict(m) for m in messages], **kwargs})
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def call(name: str, arguments: dict[str, Any], call_id: str = "c1") -> ToolCall:
    return ToolCall(call_id, name, json.dumps(arguments))


def gen(
    text: str = "", calls: list[ToolCall] | None = None, cost: float = 0.001, **kw
) -> Generation:
    return Generation(text=text, tool_calls=calls or [], usage=Usage(10, 5), cost_usd=cost, **kw)


def submit(answer: str, call_id: str = "s1") -> ToolCall:
    return call("submit_final_result", {"final_result": answer}, call_id)


# --------------------------------------------------------------------------- searchers


async def test_exa_request_shape_and_reported_cost():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "requestId": "r1",
                "costDollars": {"total": 0.012},
                "results": [{"url": "https://a", "title": "A", "highlights": ["h1", "h2"]}],
            },
        )

    searcher = ExaSearcher(api_key="k", search_type="fast", contents={"highlights": True})
    searcher._client = mock_client(handler)
    response = await searcher.run("who", 10)
    assert seen["url"] == "https://api.exa.ai/search"
    assert seen["body"] == {
        "query": "who",
        "numResults": 10,
        "type": "fast",
        "contents": {"highlights": True},
    }
    assert response.cost_usd == 0.012 and response.request_id == "r1"
    assert response.results[0].evidence == "h1\nh2"


async def test_exa_list_price_when_cost_not_reported():
    searcher = ExaSearcher(api_key="k", contents={"highlights": True})
    searcher._client = mock_client(lambda r: httpx.Response(200, json={"results": []}))
    assert (await searcher.run("q", 12)).cost_usd == pytest.approx(0.007 + 2 * 0.001)


async def test_exa_retries_server_errors():
    attempts = []

    def handler(request):
        attempts.append(1)
        return (
            httpx.Response(503) if len(attempts) == 1 else httpx.Response(200, json={"results": []})
        )

    searcher = ExaSearcher(api_key="k", max_attempts=2)
    searcher._client = mock_client(handler)
    asyncio_sleep = asyncio.sleep

    async def no_sleep(_):
        await asyncio_sleep(0)

    import shared.searchers.exa as exa_module

    original = exa_module.asyncio.sleep
    exa_module.asyncio.sleep = no_sleep
    try:
        await searcher.run("q")
    finally:
        exa_module.asyncio.sleep = original
    assert len(attempts) == 2


async def test_brave_llm_context_request_and_parse():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url.copy_with(query=None))
        seen["params"] = dict(request.url.params)
        seen["token"] = request.headers["X-Subscription-Token"]
        return httpx.Response(
            200,
            json={
                "grounding": {
                    "generic": [{"url": "https://a", "title": "A", "snippets": ["s1", "s2"]}]
                },
                "sources": {"https://a": {"age": ["x", "2025-01-01", None, "2025-01-01T10:00:00"]}},
            },
        )

    searcher = BraveSearcher(api_key="k", search_type="llm_context")
    searcher._client = mock_client(handler)
    response = await searcher.run("What's C++20's \"modules\"?", 10)
    assert seen["url"] == "https://api.search.brave.com/res/v1/llm/context"
    assert seen["params"] == {
        "q": "What s C 20 s modules",
        "count": "10",
        "maximum_number_of_tokens_per_url": "4096",
    }
    assert seen["token"] == "k"
    assert response.results[0].text == "s1\n\ns2"
    assert response.results[0].metadata["published_date"] == "2025-01-01T10:00:00"
    assert response.cost_usd == 0.005


async def test_brave_retries_422_with_shorter_query():
    queries = []

    def handler(request):
        queries.append(request.url.params["q"])
        if len(queries) == 1:
            return httpx.Response(422, json={"error": {"code": "VALIDATION"}})
        return httpx.Response(200, json={"web": {"results": []}})

    searcher = BraveSearcher(api_key="k")
    searcher._client = mock_client(handler)
    await searcher.run(" ".join(f"word{i}" for i in range(40)))
    assert len(queries) == 2 and len(queries[1].split()) == 25


async def test_brave_auth_error_is_not_retried():
    searcher = BraveSearcher(api_key="k")
    searcher._client = mock_client(
        lambda r: httpx.Response(401, json={"error": {"code": "SUBSCRIPTION_TOKEN_INVALID"}})
    )
    with pytest.raises(Exception, match="SUBSCRIPTION_TOKEN_INVALID"):
        await searcher.run("q")


async def test_parallel_batches_objective_and_queries():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "search_id": "s",
                "results": [{"url": "https://a", "title": "A", "excerpts": ["e1"]}],
            },
        )

    searcher = ParallelSearcher(api_key="k", mode="fast")
    searcher._client = mock_client(handler)
    response = await searcher.run(
        "goal", 12, objective="goal", search_queries=["a b", "c d", "e f"]
    )
    assert seen["url"] == "https://api.parallel.ai/v1/search"
    assert seen["body"] == {
        "search_queries": ["a b", "c d", "e f"],
        "mode": "fast",
        "advanced_settings": {"max_results": 12},
        "objective": "goal",
    }
    assert response.cost_usd == pytest.approx(0.001 + 2 * 0.001)
    assert response.results[0].highlights == ["e1"]


async def test_perplexity_search_api_shape():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(
            200, json={"id": "p", "results": [{"url": "https://a", "title": "A", "snippet": "s"}]}
        )

    searcher = PerplexitySearcher(api_key="k", search_type="fast")
    searcher._client = mock_client(handler)
    response = await searcher.run("q", 7)
    assert seen["body"] == {"query": "q", "max_results": 7, "search_type": "fast"}
    assert seen["auth"] == "Bearer k"
    assert response.results[0].text == "s" and response.cost_usd == 0.001


# --------------------------------------------------------------------------- tools


def test_search_tool_schema_follows_provider():
    assert SearchTool(FakeSearcher()).schema["parameters"] == QUERY_SEARCH_TOOL["parameters"]
    batched = SearchTool(FakeSearcher(batches=True))
    assert batched.schema["parameters"] == OBJECTIVE_SEARCH_TOOL["parameters"]
    assert batched.requested_queries(
        {"objective": "o", "search_queries": ["a", "b", "c", "d"]}
    ) == (
        "o",
        ["a", "b", "c"],
    )


async def test_batched_tool_sends_one_request():
    searcher = FakeSearcher(batches=True)
    outcome = await SearchTool(searcher)({"objective": "o", "search_queries": ["a", "b", "c"]})
    assert searcher.calls == [{"query": "o", "objective": "o", "search_queries": ["a", "b", "c"]}]
    assert outcome.num_searches == 1 and outcome.queries_used == 3


# --------------------------------------------------------------------------- Scout


def scout(script, *, searcher=None, **config) -> tuple[Scout, ScriptedClient, FakeSearcher]:
    searcher = searcher or FakeSearcher()
    client = ScriptedClient(script)
    loop = Scout(
        ScoutConfig(model=client.model, **config), tools=[SearchTool(searcher)], client=client
    )
    return loop, client, searcher


async def test_search_then_submit():
    loop, client, searcher = scout(
        [gen(calls=[call("search", {"query": "x"})]), gen(calls=[submit("done")])]
    )
    result = await loop.run("question?")
    assert result.stop_reason is StopReason.SUBMITTED and result.answer == "done"
    assert result.num_searches == 1 and result.rounds == 2 and result.tool_calls == 2
    assert result.search_cost_usd == pytest.approx(
        0.005
    ) and result.model_cost_usd == pytest.approx(0.002)
    assert result.citations[0]["url"] == "https://example.com/x"
    tool_message = client.requests[1]["messages"][-1]
    assert (
        tool_message["role"] == "tool" and "URL: https://example.com/x" in tool_message["content"]
    )
    assert [t["name"] for t in client.requests[0]["tools"]] == ["search", "submit_final_result"]


async def test_prose_turns_are_nudged_then_forced_to_synthesize():
    loop, client, _ = scout([gen("thinking"), gen("still"), gen("prose"), gen("final answer")])
    result = await loop.run("q")
    assert result.stop_reason is StopReason.MAX_NO_TOOL_NUDGES and result.degraded
    assert result.answer == "final answer"
    assert client.requests[1]["messages"][-1]["content"] == NO_TOOL_NUDGE
    synthesis = client.requests[-1]
    assert (
        synthesis["tools"] is None
        and synthesis["messages"][-1]["content"] == FINAL_SYNTHESIS_PROMPT
    )


async def test_max_rounds_forces_synthesis():
    script = [gen(calls=[call("search", {"query": f"q{i}"}, f"c{i}")]) for i in range(2)] + [
        gen("best effort")
    ]
    loop, _, _ = scout(script, max_rounds=2)
    result = await loop.run("q")
    assert result.stop_reason is StopReason.MAX_ROUNDS and result.answer == "best effort"


async def test_calls_after_accepted_submission_are_skipped():
    loop, _, searcher = scout(
        [
            gen(
                calls=[
                    call("search", {"query": "a"}, "c1"),
                    submit("ans", "s1"),
                    call("search", {"query": "b"}, "c2"),
                ]
            )
        ]
    )
    result = await loop.run("q")
    assert result.answer == "ans" and [c["query"] for c in searcher.calls] == ["a"]
    assert result.messages[-1]["content"] == "Skipped: final result already submitted"


async def test_invalid_arguments_are_returned_to_the_model():
    bad = ToolCall("c1", "search", "{not json")
    loop, client, _ = scout([gen(calls=[bad]), gen(calls=[submit("ok")])])
    result = await loop.run("q")
    assert result.tool_errors and client.requests[1]["messages"][-1]["content"].startswith(
        "Error: Invalid JSON"
    )


async def test_context_overflow_drops_oldest_exchange():
    script = [
        gen(calls=[call("search", {"query": "a"}, "c1")]),
        ContextWindowExceeded("too long"),
        gen(calls=[submit("ok")]),
    ]
    loop, client, _ = scout(script)
    result = await loop.run("q")
    assert result.context_truncations == 1 and result.answer == "ok"
    retried = client.requests[2]["messages"]
    assert [m["role"] for m in retried] == ["system", "user"]


def test_truncation_keeps_tool_pairs_together():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "tool_call_id": "a", "content": "r"},
        {"role": "assistant", "content": "x"},
    ]
    assert truncate_oldest_exchange(messages) == [messages[0], messages[1], messages[4]]
    with pytest.raises(ValueError):
        truncate_oldest_exchange(messages[:2])


async def test_search_budget_caps_queries():
    loop, _, searcher = scout(
        [
            gen(calls=[call("search", {"query": "a"}, "c1"), call("search", {"query": "b"}, "c2")]),
            gen("final"),
        ],
        max_searches=1,
    )
    result = await loop.run("q")
    assert [c["query"] for c in searcher.calls] == ["a"]
    assert result.stop_reason is StopReason.MAX_SEARCHES


async def test_timeout_forces_synthesis():
    loop, _, _ = scout(
        [gen(calls=[call("search", {"query": "slow"})]), gen("partial")],
        searcher=FakeSearcher(delay=5),
        timeout_s=0.2,
    )
    result = await loop.run("q")
    assert result.stop_reason is StopReason.TIMEOUT and result.answer == "partial"


async def test_budget_warning_is_sent_once():
    script = [gen(calls=[call("search", {"query": f"q{i}"}, f"c{i}")]) for i in range(3)] + [
        gen(calls=[submit("a")])
    ]
    loop, client, _ = scout(script, max_rounds=4)
    await loop.run("q")
    warnings = [
        m
        for m in client.requests[-1]["messages"]
        if m["role"] == "user" and m["content"].startswith("Research budget")
    ]
    assert len(warnings) == 1 and "3 of 4 turns used" in warnings[0]["content"]
    assert BUDGET_WARNING.split("{")[0] in warnings[0]["content"]


async def test_hosted_search_is_accounted_and_continued():
    hosted = Generation(
        text="interim",
        hosted_search=HostedSearch(
            2, ["q1", "q2"], [{"url": "https://h", "title": "H", "text": ""}]
        ),
        usage=Usage(1, 1),
        cost_usd=0.01,
    )
    client = ScriptedClient([hosted, gen(calls=[submit("done")])])
    loop = Scout(ScoutConfig(model=client.model), hosted_web_search={}, client=client)
    result = await loop.run("q")
    assert result.num_searches == 2 and result.search_cost_usd == pytest.approx(0.02)
    assert client.requests[1]["messages"][-1]["content"] == HOSTED_SEARCH_CONTINUATION
    assert (
        client.requests[0]["hosted_web_search"] == {} and result.citations[0]["url"] == "https://h"
    )


async def test_unknown_model_price_marks_cost_unknown():
    loop, _, _ = scout([gen(calls=[submit("a")], cost=None)])
    assert (await loop.run("q")).cost_known is False


# --------------------------------------------------------------------------- OpenAI client


def test_responses_input_replays_tool_history():
    instructions, items = openai_client.to_responses_input(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ]
    )
    assert instructions == "sys"
    assert items == [
        {"role": "user", "content": "u"},
        {"type": "function_call", "call_id": "c1", "name": "search", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "r"},
    ]


def test_responses_request_for_reasoning_model_with_hosted_search():
    client = openai_client.OpenAIClient("openai/gpt-5.6-luna", client=object())
    request = client.build_request(
        [{"role": "user", "content": "u"}],
        tools=[QUERY_SEARCH_TOOL],
        hosted_web_search={},
        max_output_tokens=100,
        temperature=1.0,
        reasoning_effort="medium",
    )
    assert "temperature" not in request and request["reasoning"] == {"effort": "medium"}
    assert request["tools"][-1] == {"type": "web_search", "search_context_size": "medium"}
    assert request["include"] == ["web_search_call.action.sources"]


def test_responses_parse_takes_text_after_last_search():
    text, calls, search, _ = openai_client.parse_response(
        {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "let me look"}]},
                {
                    "type": "web_search_call",
                    "action": {
                        "type": "search",
                        "query": "q",
                        "sources": [{"url": "https://s", "title": "S"}],
                    },
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "answer",
                            "annotations": [
                                {"type": "url_citation", "url": "https://c", "title": "C"}
                            ],
                        }
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "f1",
                    "name": "submit_final_result",
                    "arguments": "{}",
                },
            ]
        }
    )
    assert text == "answer" and calls == [ToolCall("f1", "submit_final_result", "{}")]
    assert search.num_searches == 1 and search.queries == ["q"]
    assert [s["url"] for s in search.sources] == ["https://s", "https://c"]


# --------------------------------------------------------------------------- Anthropic client


def test_anthropic_groups_tool_results():
    system, messages = anthropic_client.to_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "content": "plan",
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"query": "x"}'},
                    },
                    {
                        "id": "b",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"query": "y"}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "ra"},
            {"role": "tool", "tool_call_id": "b", "content": "rb"},
        ]
    )
    assert system == "sys"
    assert messages[1]["content"][1] == {
        "type": "tool_use",
        "id": "a",
        "name": "search",
        "input": {"query": "x"},
    }
    assert messages[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "ra"},
            {"type": "tool_result", "tool_use_id": "b", "content": "rb"},
        ],
    }


def test_anthropic_schema_moves_array_bounds_into_description():
    schema = anthropic_client.sanitize_schema(OBJECTIVE_SEARCH_TOOL["parameters"])
    queries = schema["properties"]["search_queries"]
    assert queries["minItems"] == 1 and "maxItems" not in queries
    assert queries["description"].endswith("Exactly 3 items must be provided.")


class _FakeStream:
    def __init__(self, message):
        self.message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self.message


class _FakeMessage:
    def __init__(self, data):
        self.data = data

    def model_dump(self, **_):
        return self.data


class _FakeAnthropic:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def stream(self, **request):
        self.requests.append(request)
        return _FakeStream(_FakeMessage(self.responses.pop(0)))


async def test_anthropic_continues_paused_server_tool_turns():
    paused = {
        "stop_reason": "pause_turn",
        "content": [
            {
                "type": "server_tool_use",
                "name": "web_search",
                "id": "t1",
                "input": {"query": "q1"},
                "caller": "direct",
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "t1",
                "content": [{"url": "https://a", "title": "A"}],
            },
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 10,
            "server_tool_use": {"web_search_requests": 1},
        },
    }
    final = {
        "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "Based on it, ", "citations": None},
            {
                "type": "text",
                "text": "A is right",
                "citations": [{"url": "https://a", "title": "A", "cited_text": "c"}],
            },
            {
                "type": "tool_use",
                "id": "u1",
                "name": "submit_final_result",
                "input": {"final_result": "A"},
            },
        ],
        "usage": {"input_tokens": 120, "output_tokens": 20, "cache_read_input_tokens": 50},
    }
    fake = _FakeAnthropic([paused, final])
    client = anthropic_client.AnthropicClient("anthropic/claude-sonnet-5", client=fake)
    generation = await client.generate(
        [{"role": "user", "content": "u"}],
        hosted_web_search={"max_uses": 15},
        reasoning_effort="medium",
    )
    assert len(fake.requests) == 2
    assert fake.requests[1]["messages"][-1]["content"][0]["caller"] == {"type": "direct"}
    assert fake.requests[0]["tools"][-1]["max_uses"] == 15
    assert fake.requests[0]["output_config"] == {"effort": "medium"}
    assert generation.text == "Based on it, A is right"
    assert generation.tool_calls == [ToolCall("u1", "submit_final_result", '{"final_result": "A"}')]
    assert generation.hosted_search.num_searches == 1 and generation.hosted_search.queries == ["q1"]
    assert generation.usage.input_tokens == 220 and generation.usage.cached_input_tokens == 50
    assert generation.cost_usd == pytest.approx((220 * 2.0 + 50 * 0.2 + 30 * 10.0) / 1e6)


# --------------------------------------------------------------------------- pricing


def test_token_cost_conventions():
    usage = Usage(input_tokens=1000, output_tokens=100, cached_input_tokens=400)
    assert token_cost("openai/gpt-5.6-luna", usage, cached_included_in_input=True) == pytest.approx(
        (600 * 0.20 + 400 * 0.02 + 100 * 1.20) / 1e6
    )
    assert model_price("openai/gpt-5.4-mini-2026-03-17") == model_price("openai/gpt-5.4-mini")
    assert token_cost("openai/unknown-model", usage, cached_included_in_input=True) is None


# --------------------------------------------------------------------------- judge


class _Verdict(BaseModel):
    label: str


async def test_judge_uses_structured_outputs_when_available():
    client = ScriptedClient([gen('{"label": "yes"}', cost=0.002)])
    parsed, response = await Judge(client.model, client=client).complete_json("grade it", _Verdict)
    assert parsed.label == "yes" and response.cost_usd == 0.002
    request = client.requests[0]
    assert request["response_schema"]["name"] == "_Verdict"
    assert request["response_schema"]["schema"]["additionalProperties"] is False
    assert request["response_schema"]["schema"]["required"] == ["label"]
    assert request["messages"][-1]["content"] == "grade it"


def test_openai_structured_outputs_are_strict():
    client = openai_client.OpenAIClient("openai/gpt-5.6-luna", client=object())
    request = client.build_request(
        [{"role": "user", "content": "u"}],
        tools=None,
        hosted_web_search=None,
        max_output_tokens=None,
        temperature=None,
        reasoning_effort=None,
        response_schema={"name": "V", "schema": {"type": "object"}},
    )
    assert request["text"]["format"] == {
        "type": "json_schema",
        "name": "V",
        "schema": {"type": "object"},
        "strict": True,
    }


async def test_judge_prompts_for_json_and_retries_without_structured_outputs():
    client = ScriptedClient([gen("not json"), gen('Sure: {"label": "no"}')])
    client.supports_response_schema = False
    parsed, response = await Judge(client.model, client=client).complete_json("grade it", _Verdict)
    assert parsed.label == "no" and response.usage.input_tokens == 20
    assert "JSON schema" in client.requests[0]["messages"][-1]["content"]
    assert client.requests[0]["response_schema"] is None


# --------------------------------------------------------------------------- catalog


def test_every_catalog_system_builds(monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "EXA_API_KEY",
        "BRAVE_SEARCH_API_KEY",
        "PARALLEL_API_KEY",
        "PERPLEXITY_API_KEY",
    ):
        monkeypatch.setenv(name, "test")
    catalog = Catalog.load()
    assert catalog.systems
    for name in catalog.systems:
        spec = catalog.resolve(name)
        System(spec)
        assert spec.to_dict()["name"] == name


def test_scout_systems_differ_only_in_search_backend():
    catalog = Catalog.load()
    specs = [catalog.resolve(n) for n in catalog.systems if n.startswith("scout-")]
    assert len({json.dumps(s.settings, sort_keys=True) for s in specs}) == 1
    assert len({s.model for s in specs}) == 1


def test_model_override_and_hosted_provider_guard():
    catalog = Catalog.load()
    assert (
        catalog.resolve("scout-brave", model="anthropic/claude-sonnet-5").model
        == "anthropic/claude-sonnet-5"
    )
    with pytest.raises(ValueError, match="hosted search"):
        catalog.resolve("openai-native-search", model="anthropic/claude-sonnet-5")


# --------------------------------------------------------------------------- runner


class EchoSuite(Suite):
    name = "echo"
    description = "test suite"
    primary_metric = "score"
    revision = "v1"

    def load(self):
        return [Task(id=f"t/{i}", problem=f"p{i}", answer=f"p{i}") for i in range(3)]

    async def grade(self, task, response, judge):
        return Grade(scores={"score": float(response == task.answer)}, judge_cost_usd=0.001)


class EchoSystem:
    def __init__(self, fail: set[str] = frozenset()):
        self.fail = set(fail)
        self.calls = 0

    async def answer(self, prompt):
        self.calls += 1
        if prompt in self.fail:
            raise RuntimeError("boom")
        return {
            "answer": prompt,
            "total_cost_usd": 0.01,
            "model_cost_usd": 0.01,
            "cost_known": True,
        }

    async def close(self):
        pass


class NoJudge:
    model = "openai/gpt-5.6-luna"


def make_runner(tmp_path, system):
    catalog = Catalog.load()
    return Runner(
        catalog.resolve("scout-exa-auto"),
        EchoSuite(),
        judge=NoJudge(),
        runs_root=tmp_path,
        system=system,
    )


async def test_runner_resumes_and_counts_failures_as_zero(tmp_path):
    flaky = EchoSystem(fail={"p1"})
    runner = make_runner(tmp_path, flaky)
    tasks = EchoSuite().load()
    summary = await runner.run(tasks)
    assert summary["graded"] == 2 and summary["failed"] == 1
    assert summary["metrics"]["score"] == 1.0 and summary["score_failed_as_zero"] == pytest.approx(
        2 / 3
    )
    assert (runner.run_dir / "tasks" / "t_1" / "error.json").exists()

    healthy = EchoSystem()
    summary = await make_runner(tmp_path, healthy).run(tasks)
    assert healthy.calls == 1 and summary["graded"] == 3
    assert not (runner.run_dir / "tasks" / "t_1" / "error.json").exists()


async def test_runner_regrades_without_rerunning_the_system(tmp_path):
    system = EchoSystem()
    runner = make_runner(tmp_path, system)
    tasks = EchoSuite().load()
    await runner.run(tasks)
    (runner.run_dir / "tasks" / "t_0" / "grade.json").unlink()
    again = EchoSystem()
    await make_runner(tmp_path, again).run(tasks)
    assert again.calls == 0 and (runner.run_dir / "tasks" / "t_0" / "grade.json").exists()


def test_run_directory_changes_with_the_spec(tmp_path):
    catalog = Catalog.load()
    a = Runner(
        catalog.resolve("scout-exa-auto"),
        EchoSuite(),
        judge=NoJudge(),
        runs_root=tmp_path,
        system=EchoSystem(),
    )
    b = Runner(
        catalog.resolve("scout-exa-fast"),
        EchoSuite(),
        judge=NoJudge(),
        runs_root=tmp_path,
        system=EchoSystem(),
    )
    c = Runner(
        catalog.resolve("scout-exa-auto"),
        EchoSuite(),
        judge=NoJudge(),
        runs_root=tmp_path,
        run_suffix="rep2",
        system=EchoSystem(),
    )
    assert len({a.run_dir, b.run_dir, c.run_dir}) == 3

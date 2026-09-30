"""Scout: a fixed, model-agnostic web-research loop.

Scout gives one model a search tool and a terminal ``submit_final_result`` tool
and lets it research until it submits. The loop, prompts and budgets stay fixed
while the search backend changes, so differences between systems measure the
search API, not the agent around it. The same loop also drives a provider's own
hosted web search (OpenAI, Anthropic) in place of a caller-owned search tool.

One trajectory:

1. The model sees the system prompt, the task, and its tools.
2. Each research turn may call any number of tools; sibling calls run
   concurrently, and calls after an accepted submission are skipped.
3. A turn with no tool call gets a nudge; ``max_no_tool_nudges`` nudges end the
   research phase.
4. The run ends when the model calls ``submit_final_result``. Exhausting any
   budget (turns, wall clock, searches, USD) instead forces one tools-disabled
   synthesis turn over the evidence gathered so far, and the result is marked
   degraded with its stop reason.
5. When a prompt overflows the context window, the oldest completed exchange is
   dropped and the turn is retried.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, TypeVar

from harness.llm import SCOUT_DEFAULT
from harness.llm.clients import ModelClient, create_client
from harness.llm.pricing import hosted_search_cost
from harness.llm.types import ContextWindowExceeded, Generation, ToolCall, Usage
from harness.tools import SearchTool, ToolOutcome

SCOUT_SYSTEM_PROMPT = """You are a web research agent. You answer questions by searching the web, reading the results, and synthesizing what you find into a direct answer.

Today's date: {current_date}

You have a search tool. Use it strategically:
- Craft precise queries — include specific names, model numbers, technical terms.
- Search multiple times when a single query won't cover it. Refine based on what you learn.
- Cross-reference across sources when accuracy matters (prices, specs, availability).

When answering:
- Ground every claim in search results. Include URLs, prices, specs, dates — whatever the question demands.
- Be specific and actionable. Names, links, numbers — not vague suggestions.
- If the results are insufficient or conflicting, say so clearly and explain what's missing.
- Don't pad. Answer the question, then stop.
- Use today's date to resolve relative time references (e.g. "yesterday", "last week", "3 days ago")."""

SUBMISSION_INSTRUCTION = (
    "When tools are available, finish only by calling `submit_final_result`; do not "
    "return the final answer as ordinary prose. If tools are unavailable during a "
    "forced synthesis turn, return the final answer directly."
)
NO_TOOL_NUDGE = (
    "Your last response produced no tool call. Call `submit_final_result` if you "
    "have a complete answer; otherwise continue with the next search tool call."
)
HOSTED_SEARCH_CONTINUATION = (
    "You used provider-hosted search in the last turn. Continue researching if "
    "needed; otherwise call `submit_final_result` with the complete answer."
)
FINAL_SYNTHESIS_PROMPT = (
    "The research budget is exhausted and no more tool calls are available. "
    "Synthesize the best final answer now using only the evidence already in the "
    "conversation. Be direct, include important source URLs, and clearly identify "
    "any missing or conflicting evidence."
)
BUDGET_WARNING = (
    "Research budget notice: {usage}. Wrap up soon and call `submit_final_result` "
    "before the budget is exhausted."
)
EMPTY_SYNTHESIS_ANSWER = "I don't have enough information to answer this question."

SUBMIT_TOOL: dict[str, Any] = {
    "name": "submit_final_result",
    "description": (
        "Submit the final answer and end the research loop. Include the answer, "
        "supporting explanation, and important source URLs. The run does not finish "
        "successfully until this tool is called."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "final_result": {
                "type": "string",
                "description": "The complete final answer to return to the user.",
            }
        },
        "required": ["final_result"],
    },
}

_T = TypeVar("_T")


class StopReason(StrEnum):
    """Why a trajectory ended."""

    SUBMITTED = "submitted"
    MAX_ROUNDS = "max_rounds"
    TIMEOUT = "timeout"
    MAX_NO_TOOL_NUDGES = "max_no_tool_nudges"
    MAX_SEARCHES = "max_searches"
    MAX_COST = "max_cost"


@dataclass
class ScoutConfig:
    """Model, generation settings and research budgets for one Scout system.

    Every budget is optional except ``timeout_s``. ``max_output_tokens`` applies
    to each model call separately, not to the whole trajectory.
    """

    model: str = SCOUT_DEFAULT
    reasoning_effort: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = 32_000
    max_rounds: int | None = 25
    max_tool_calls_per_round: int | None = None
    max_searches: int | None = None
    max_cost_usd: float | None = None
    max_no_tool_nudges: int | None = 3
    budget_warning_fraction: float = 0.75
    timeout_s: float = 3600.0
    system_prompt: str = SCOUT_SYSTEM_PROMPT


@dataclass
class ScoutResult:
    """One finished trajectory: the answer, its evidence, accounting and full transcript."""

    answer: str
    stop_reason: StopReason
    citations: list[dict[str, Any]]
    rounds: int
    tool_calls: int
    num_searches: int
    search_queries: list[str]
    model_cost_usd: float
    search_cost_usd: float
    cost_known: bool
    usage: Usage
    latency_ms: float
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    tool_errors: list[dict[str, str]] = field(default_factory=list)
    context_truncations: int = 0
    no_tool_nudges: int = 0

    @property
    def total_cost_usd(self) -> float:
        return self.model_cost_usd + self.search_cost_usd

    @property
    def degraded(self) -> bool:
        """True when a budget, not the model, ended the research phase."""
        return self.stop_reason is not StopReason.SUBMITTED

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stop_reason"] = self.stop_reason.value
        data["total_cost_usd"] = self.total_cost_usd
        data["degraded"] = self.degraded
        return data


class _ResearchTimeout(Exception):
    """The trajectory's wall-clock deadline passed."""


async def _before_deadline(operation: Callable[[], Awaitable[_T]], deadline: float) -> _T:
    """Run one research operation, cancelling it at the trajectory deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _ResearchTimeout
    try:
        async with asyncio.timeout(remaining):
            return await operation()
    except TimeoutError as error:
        if time.monotonic() >= deadline:
            raise _ResearchTimeout from error
        raise


def truncate_oldest_exchange(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the first assistant message and everything up to the next one.

    Removing a whole exchange keeps every tool call paired with its result.
    Raises ``ValueError`` when there is no completed exchange left to drop.
    """
    first = next((i for i, m in enumerate(messages) if m["role"] == "assistant"), None)
    if first is None:
        raise ValueError("no completed exchange left to drop")
    following = next(
        (i for i in range(first + 1, len(messages)) if messages[i]["role"] == "assistant"),
        len(messages),
    )
    return [*messages[:first], *messages[following:]]


def budget_usage(
    config: ScoutConfig,
    *,
    rounds: int,
    elapsed_s: float,
    searches: int,
    max_searches: int | None,
    cost: float,
) -> tuple[float, str]:
    """Return the largest used fraction across bounded budgets and a readable summary."""
    ratios: list[float] = []
    parts: list[str] = []
    if config.max_rounds is not None:
        ratios.append(rounds / config.max_rounds)
        parts.append(f"{rounds} of {config.max_rounds} turns used")
    ratios.append(elapsed_s / config.timeout_s)
    parts.append(f"{elapsed_s:.0f} of {config.timeout_s:.0f} seconds elapsed")
    if max_searches is not None:
        ratios.append(searches / max_searches)
        parts.append(f"{searches} of {max_searches} searches used")
    if config.max_cost_usd is not None:
        ratios.append(cost / config.max_cost_usd)
        parts.append(f"{cost:.2f} of {config.max_cost_usd:.2f} USD spent")
    return max(ratios), "; ".join(parts)


def parse_arguments(call: ToolCall) -> tuple[dict[str, Any], str | None]:
    """Parse a tool call's JSON arguments; a parse failure is returned to the model."""
    if not call.arguments:
        return {}, None
    try:
        arguments = json.loads(call.arguments)
    except json.JSONDecodeError as error:
        return {}, f"Invalid JSON arguments for {call.name}: {error.msg}"
    if not isinstance(arguments, dict):
        return {}, f"Arguments for {call.name} must be a JSON object"
    return arguments, None


def submit(arguments: dict[str, Any]) -> ToolOutcome:
    """Accept a non-empty final answer from ``submit_final_result``."""
    final = arguments.get("final_result")
    if isinstance(final, (dict, list)):
        final = json.dumps(final)
    if not isinstance(final, str) or not final.strip():
        return ToolOutcome(
            content="Error: final_result must be a non-empty string", error="Empty final result"
        )
    return ToolOutcome(content=final.strip(), final_answer=final.strip())


def assistant_message(generation: Generation) -> dict[str, Any]:
    """Record one model turn in Chat Completions shape."""
    message: dict[str, Any] = {"role": "assistant", "content": generation.text or ""}
    if generation.tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in generation.tool_calls
        ]
    if generation.reasoning:
        message["reasoning"] = generation.reasoning
    return message


class _Trajectory:
    """Mutable accounting for one Scout run."""

    def __init__(self, question: str, config: ScoutConfig, system_prompt: str) -> None:
        self.started = time.monotonic()
        self.deadline = self.started + config.timeout_s
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        self.rounds = 0
        self.tool_calls = 0
        self.num_searches = 0
        self.queries_used = 0
        self.model_cost = 0.0
        self.search_cost = 0.0
        self.cost_known = True
        self.usage = Usage()
        self.tool_history: list[dict[str, Any]] = []
        self.evidence: list[dict[str, Any]] = []
        self.tool_errors: list[dict[str, str]] = []
        self.context_truncations = 0
        self.no_tool_nudges = 0
        self.budget_warning_sent = False

    @property
    def cost(self) -> float:
        return self.model_cost + self.search_cost

    def add_generation(self, generation: Generation) -> None:
        self.usage = self.usage + generation.usage
        if generation.cost_usd is None:
            self.cost_known = False
        else:
            self.model_cost += generation.cost_usd

    def add_outcome(self, call: ToolCall, outcome: ToolOutcome) -> None:
        self.tool_calls += 1
        self.num_searches += outcome.num_searches
        self.queries_used += outcome.queries_used
        self.search_cost += outcome.cost_usd
        self.cost_known = self.cost_known and outcome.cost_known
        self.tool_history.extend(outcome.records)
        self.evidence.extend(outcome.evidence)
        self.messages.append({"role": "tool", "tool_call_id": call.id, "content": outcome.content})
        if outcome.error:
            self.tool_errors.append({"id": call.id, "name": call.name, "error": outcome.error})


class Scout:
    """The Scout loop bound to one model and one search backend."""

    def __init__(
        self,
        config: ScoutConfig,
        *,
        tools: list[SearchTool] | None = None,
        hosted_web_search: dict[str, Any] | None = None,
        client: ModelClient | None = None,
    ) -> None:
        self.config = config
        self.tools = {tool.name: tool for tool in tools or []}
        if "submit_final_result" in self.tools:
            raise ValueError("'submit_final_result' is reserved for the terminal tool")
        self.hosted_web_search = hosted_web_search
        self.client = client or create_client(config.model)
        self.tool_schemas = [tool.schema for tool in self.tools.values()] + [SUBMIT_TOOL]
        self._required = {
            schema["name"]: list(schema["parameters"].get("required", []))
            for schema in self.tool_schemas
        }

    async def run(self, question: str) -> ScoutResult:
        """Research one question to a final answer."""
        current_date = datetime.now(timezone.utc).strftime("%B %d, %Y")
        system_prompt = f"{self.config.system_prompt.replace('{current_date}', current_date)}\n\n{SUBMISSION_INSTRUCTION}"
        run = _Trajectory(question, self.config, system_prompt)
        stop = await self._research(run)
        if not isinstance(stop, StopReason):
            return self._result(run, stop, StopReason.SUBMITTED)
        answer = await self._synthesize(run)
        return self._result(run, answer, stop)

    async def _research(self, run: _Trajectory) -> StopReason | str:
        """Run research turns; return the submitted answer or the reason research stopped."""
        config = self.config
        max_searches = config.max_searches
        while True:
            if config.max_rounds is not None and run.rounds >= config.max_rounds:
                return StopReason.MAX_ROUNDS
            if max_searches is not None and run.queries_used >= max_searches:
                return StopReason.MAX_SEARCHES
            if config.max_cost_usd is not None and run.cost >= config.max_cost_usd:
                return StopReason.MAX_COST
            try:
                generation = await _before_deadline(
                    lambda: self._generate(
                        run, tools=self.tool_schemas, hosted_web_search=self.hosted_web_search
                    ),
                    run.deadline,
                )
            except _ResearchTimeout:
                return StopReason.TIMEOUT
            run.rounds += 1
            run.add_generation(generation)
            hosted_activity = self._record_hosted_search(run, generation)
            run.messages.append(assistant_message(generation))

            if not generation.tool_calls:
                if hosted_activity:
                    run.messages.append({"role": "user", "content": HOSTED_SEARCH_CONTINUATION})
                else:
                    run.messages.append({"role": "user", "content": NO_TOOL_NUDGE})
                    run.no_tool_nudges += 1
                    if (
                        config.max_no_tool_nudges is not None
                        and run.no_tool_nudges >= config.max_no_tool_nudges
                    ):
                        return StopReason.MAX_NO_TOOL_NUDGES
                self._maybe_warn(run)
                continue

            outcomes, timed_out, cost_exhausted = await self._execute_tools(
                run, generation.tool_calls
            )
            answer: str | None = None
            for call, outcome in outcomes:
                run.add_outcome(call, outcome)
                if outcome.done:
                    answer = outcome.final_answer
            self._maybe_warn(run)
            if answer is not None:
                return answer
            if timed_out:
                return StopReason.TIMEOUT
            if cost_exhausted:
                return StopReason.MAX_COST

    async def _synthesize(self, run: _Trajectory) -> str:
        """Force one tools-disabled answer over the evidence gathered so far."""
        run.messages.append({"role": "user", "content": FINAL_SYNTHESIS_PROMPT})
        generation = await self._generate(
            run, tools=None, hosted_web_search=None, preserve_last_user=True
        )
        run.add_generation(generation)
        answer = generation.text.strip() or EMPTY_SYNTHESIS_ANSWER
        run.messages.append({"role": "assistant", "content": answer})
        return answer

    async def _generate(
        self,
        run: _Trajectory,
        *,
        tools: list[dict[str, Any]] | None,
        hosted_web_search: dict[str, Any] | None,
        preserve_last_user: bool = False,
    ) -> Generation:
        """Generate one turn, dropping the oldest exchange whenever the context overflows."""
        while True:
            try:
                return await self.client.generate(
                    run.messages,
                    tools=tools,
                    hosted_web_search=hosted_web_search,
                    max_output_tokens=self.config.max_output_tokens,
                    temperature=self.config.temperature,
                    reasoning_effort=self.config.reasoning_effort,
                )
            except ContextWindowExceeded as error:
                try:
                    truncated = truncate_oldest_exchange(run.messages)
                except ValueError:
                    raise error from None
                last = run.messages[-1]
                if (
                    preserve_last_user
                    and last["role"] == "user"
                    and (not truncated or truncated[-1] is not last)
                ):
                    truncated.append(last)
                run.messages[:] = truncated
                run.context_truncations += 1

    def _record_hosted_search(self, run: _Trajectory, generation: Generation) -> bool:
        """Account for provider-executed searches; return whether any happened."""
        search = generation.hosted_search
        if search is None or (search.num_searches == 0 and not search.sources):
            return False
        cost = hosted_search_cost(self.client.provider, search.num_searches)
        run.tool_calls += search.num_searches
        run.num_searches += search.num_searches
        run.search_cost += cost
        run.tool_history.append(
            {
                "tool": "web_search",
                "hosted": True,
                "query": search.queries[0] if search.queries else "",
                "search_queries": list(search.queries),
                "results": list(search.sources),
                "num_results": len(search.sources),
                "num_searches": search.num_searches,
                "cost_usd": cost,
            }
        )
        run.evidence.extend(search.sources)
        return True

    def _maybe_warn(self, run: _Trajectory) -> None:
        """Append the one-time budget notice once any bounded budget crosses the threshold."""
        if run.budget_warning_sent:
            return
        ratio, usage = budget_usage(
            self.config,
            rounds=run.rounds,
            elapsed_s=time.monotonic() - run.started,
            searches=run.queries_used,
            max_searches=self.config.max_searches,
            cost=run.cost,
        )
        if ratio >= self.config.budget_warning_fraction:
            run.budget_warning_sent = True
            run.messages.append({"role": "user", "content": BUDGET_WARNING.format(usage=usage)})

    async def _call_tool(
        self, index: int, call: ToolCall, allowance: int | None, run: _Trajectory
    ) -> ToolOutcome:
        """Validate and execute one tool call; errors go back to the model as tool output."""
        cap = self.config.max_tool_calls_per_round
        if cap is not None and index >= cap:
            return ToolOutcome(
                content="Skipped: tool call limit exceeded", error="Tool call limit exceeded"
            )
        arguments, parse_error = parse_arguments(call)
        if parse_error:
            return ToolOutcome(content=f"Error: {parse_error}", error=parse_error)
        if call.name not in self._required:
            return ToolOutcome(
                content=f"Error: Unknown tool: {call.name}", error=f"Unknown tool: {call.name}"
            )
        missing = [name for name in self._required[call.name] if name not in arguments]
        if missing:
            error = f"Missing required arguments for {call.name}: {missing}"
            return ToolOutcome(content=f"Error: {error}", error=error)
        if call.name == SUBMIT_TOOL["name"]:
            return submit(arguments)
        try:
            return await _before_deadline(
                lambda: self.tools[call.name](arguments, max_queries=allowance), run.deadline
            )
        except _ResearchTimeout:
            return ToolOutcome(
                content="Skipped: Research trajectory timeout reached",
                error="Research trajectory timeout reached",
            )

    def _allowances(self, run: _Trajectory, calls: list[ToolCall]) -> dict[int, int | None]:
        """Split the remaining search budget across this turn's search calls in order."""
        if self.config.max_searches is None:
            return {}
        remaining = max(0, self.config.max_searches - run.queries_used)
        allowances: dict[int, int | None] = {}
        for index, call in enumerate(calls):
            tool = self.tools.get(call.name)
            if tool is None:
                continue
            arguments, error = parse_arguments(call)
            requested = 0 if error else len(tool.requested_queries(arguments)[1])
            allowances[index] = min(requested, remaining)
            remaining -= allowances[index]
        return allowances

    async def _execute_tools(
        self, run: _Trajectory, calls: list[ToolCall]
    ) -> tuple[list[tuple[ToolCall, ToolOutcome]], bool, bool]:
        """Execute one turn's calls in declared order.

        Without a USD budget, calls up to and including each ``submit_final_result``
        run concurrently, and everything after an accepted submission is skipped.
        With a USD budget, calls run one at a time so observed spend can stop the
        rest; the overshoot is bounded by one search call.

        Returns the outcomes plus whether the deadline or the cost budget expired.
        """
        allowances = self._allowances(run, calls)
        timed_out = False

        async def one(index: int) -> tuple[ToolCall, ToolOutcome]:
            nonlocal timed_out
            outcome = await self._call_tool(index, calls[index], allowances.get(index), run)
            if outcome.error == "Research trajectory timeout reached":
                timed_out = True
            return calls[index], outcome

        skipped = ToolOutcome(content="Skipped: final result already submitted")
        outcomes: list[tuple[ToolCall, ToolOutcome]] = []

        if self.config.max_cost_usd is not None:
            spent = run.cost
            exhausted = False
            submitted = False
            for index, call in enumerate(calls):
                if submitted:
                    outcomes.append((call, skipped))
                    continue
                if call.name in self.tools and spent >= self.config.max_cost_usd:
                    exhausted = True
                    outcomes.append(
                        (
                            call,
                            ToolOutcome(
                                content="Skipped: trajectory cost budget exhausted",
                                error="Trajectory cost budget exhausted",
                            ),
                        )
                    )
                    continue
                pair = await one(index)
                spent += pair[1].cost_usd
                submitted = pair[1].done
                outcomes.append(pair)
            return outcomes, timed_out, exhausted or spent >= self.config.max_cost_usd

        start = 0
        while start < len(calls):
            terminal = next(
                (i for i in range(start, len(calls)) if calls[i].name == SUBMIT_TOOL["name"]), None
            )
            end = len(calls) if terminal is None else terminal + 1
            outcomes.extend(await asyncio.gather(*(one(i) for i in range(start, end))))
            if terminal is None:
                break
            if outcomes[-1][1].done:
                outcomes.extend((calls[i], skipped) for i in range(end, len(calls)))
                break
            start = end
        return outcomes, timed_out, False

    def _result(self, run: _Trajectory, answer: str, stop: StopReason) -> ScoutResult:
        queries = [q for record in run.tool_history for q in record.get("search_queries", []) if q]
        return ScoutResult(
            answer=answer,
            stop_reason=stop,
            citations=[
                {"url": e["url"], "title": e.get("title") or e["url"], "text": e.get("text", "")}
                for e in run.evidence
                if e.get("url")
            ],
            rounds=run.rounds,
            tool_calls=run.tool_calls,
            num_searches=run.num_searches,
            search_queries=queries,
            model_cost_usd=run.model_cost,
            search_cost_usd=run.search_cost,
            cost_known=run.cost_known,
            usage=run.usage,
            latency_ms=(time.monotonic() - run.started) * 1000,
            messages=run.messages,
            tool_history=run.tool_history,
            tool_errors=run.tool_errors,
            context_truncations=run.context_truncations,
            no_tool_nudges=run.no_tool_nudges,
        )

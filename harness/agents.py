"""Hosted research products, distinct from Scout's shared search-tool loop.

Use public HTTP endpoints without provider SDK dependencies. Persist accepted
run IDs before polling so retries and process restarts reuse paid runs. A POST
with an ambiguous outcome is never automatically repeated.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

AGENT_PROVIDERS = ("exa-agent", "parallel-task", "perplexity-agent")


class AgentRequestError(RuntimeError):
    """Do not repeat this task automatically (it may already have incurred cost)."""


def parse_rows(answer: Any) -> tuple[list[dict], str | None]:
    """Preserve every returned row; malformed output becomes an auditable zero."""
    if isinstance(answer, str):
        text = answer.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            answer = json.loads(text)
        except (ValueError, TypeError):
            return [], "Agent output was not valid JSON"
    rows = answer.get("rows") if isinstance(answer, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return [], "Agent output must be an object with a rows array of objects"
    return rows, None


def dollars(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and value >= 0:
            return float(value)
    return None


class HostedAgent:
    """Shared request limits and durable lifecycle for one hosted product.

    Checkpoints move from ``creating`` (POST outcome not yet known), to
    ``run_id`` (safe to poll), to ``response`` (safe to replay without network).
    Keep the marker on ambiguous errors; a fresh paid attempt requires an
    explicit new run. Each subclass only supplies its protocol and output parser.
    """

    provider: str
    base_url: str
    key_names: tuple[str, ...]
    base_delay = 1.0

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 3600,
        poll_interval_s: float = 2,
        max_retries: int = 3,
        max_concurrency: int = 3,
        qps: float = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout_s <= 0 or poll_interval_s < 0 or max_retries < 0:
            raise ValueError("invalid agent timeout, poll interval, or retries")
        if max_concurrency < 1 or qps <= 0:
            raise ValueError("agent concurrency and qps must be positive")
        key = api_key or next((os.environ[k] for k in self.key_names if os.environ.get(k)), None)
        if not key:
            raise ValueError(f"Missing {' or '.join(self.key_names)}")
        self.base_url = (base_url or self.base_url).rstrip("/")
        self.headers = (
            {"Authorization": f"Bearer {key}"}
            if self.provider == "perplexity-agent"
            else {"x-api-key": key}
        )
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._start_lock = asyncio.Lock()
        self._last_start = 0.0
        self._start_interval = 1 / qps
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(60))

    async def close(self) -> None:
        await self.client.aclose()

    @staticmethod
    def save(path: Path | None, state: dict) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, ensure_ascii=False))
            temporary.replace(path)

    async def create(self, endpoint: str, body: dict, state: dict, path: Path | None) -> dict:
        """Submit once; only an explicit rate-limit rejection permits retrying."""
        if state.get("creating"):
            raise AgentRequestError(
                "Previous agent creation has an unknown outcome; inspect the provider dashboard "
                "and checkpoint before retrying, or use --run-suffix for a new paid run"
            )
        for attempt in range(self.max_retries + 1):
            async with self._start_lock:
                await asyncio.sleep(
                    max(0.0, self._last_start + self._start_interval - time.monotonic())
                )
                self._last_start = time.monotonic()
            state["creating"] = True
            self.save(path, state)
            response = await self.client.post(
                self.base_url + endpoint,
                headers=self.headers,
                json=body,
                timeout=self.timeout_s,
            )
            if 400 <= response.status_code < 500 and response.status_code != 408:
                state.pop("creating", None)  # explicitly rejected, not an accepted paid run
                self.save(path, state)
                if response.status_code == 429 and attempt < self.max_retries:
                    await asyncio.sleep(self.base_delay * 2**attempt)
                    continue
            response.raise_for_status()
            return response.json()
        raise AgentRequestError("Agent creation retries exhausted")

    async def poll(self, endpoint: str, *, parallel: bool = False) -> dict:
        """Retry only GETs against the same accepted run, under the outer deadline."""
        while True:
            try:
                response = await self.client.get(
                    self.base_url + endpoint,
                    headers=self.headers,
                    params={"timeout": 30} if parallel else None,
                    timeout=60,
                )
            except httpx.TransportError:
                await asyncio.sleep(self.poll_interval_s)
                continue
            if response.status_code in (408, 429) or response.status_code >= 500:
                await asyncio.sleep(self.poll_interval_s)
                continue
            response.raise_for_status()
            return response.json()

    async def run(
        self,
        prompt: str,
        *,
        request: dict | None = None,
        state_path: Path | None = None,
    ) -> dict[str, Any]:
        """Return all rows plus raw output, cost and accumulated active wait time."""
        async with self._semaphore:
            return await self._run(prompt, request=request, state_path=state_path)

    async def _run(
        self,
        prompt: str,
        *,
        request: dict | None,
        state_path: Path | None,
    ) -> dict[str, Any]:
        request = request or {}
        state = json.loads(state_path.read_text()) if state_path and state_path.exists() else {}
        start = time.monotonic()
        try:
            if "response" not in state:
                try:
                    async with asyncio.timeout(self.timeout_s):
                        state["response"] = await self.fetch(prompt, request, state, state_path)
                    state.pop("creating", None)
                finally:
                    # Include prior polling attempts, but never time spent replaying a cache.
                    state["latency_ms"] = (
                        state.get("latency_ms", 0.0) + (time.monotonic() - start) * 1000
                    )
                    self.save(state_path, state)
            raw = state["response"]
            answer, cost, search_cost = self.unpack(raw)
        except (httpx.HTTPError, TimeoutError, ValueError, KeyError) as error:
            raise AgentRequestError(
                f"{self.provider}: {error}; accepted run ID: {state.get('run_id', 'unknown')}"
            ) from error
        text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        rows, parse_error = parse_rows(answer) if request.get("output_schema") else ([], None)
        usage = raw.get("usage") or {}
        search_count = usage.get("numSearches", usage.get("searches"))
        if self.provider == "perplexity-agent":
            search_count = sum(item.get("type") == "web_search" for item in raw.get("output", []))
        return {
            "answer": text,
            "rows": rows,
            "parse_error": parse_error,
            "raw_response": raw,
            "request_id": state.get("run_id") or raw.get("id"),
            "model_cost_usd": max(0.0, (cost or 0.0) - search_cost),
            "search_cost_usd": search_cost,
            "total_cost_usd": cost if cost is not None else search_cost,
            "cost_known": cost is not None,
            "cost_source": "configured_per_run" if self.provider == "parallel-task" else "provider",
            "num_searches": search_count or 0,
            "search_count_known": search_count is not None,
            "latency_ms": state.get("latency_ms", 0.0),
            "stop_reason": "hosted_agent",
        }

    async def fetch(self, prompt: str, request: dict, state: dict, path: Path | None) -> dict:
        """Create or resume a run, checkpoint its ID, and return its terminal response."""
        raise NotImplementedError

    def unpack(self, raw: dict) -> tuple[Any, float | None, float]:
        """Return (answer, total USD or unknown, search USD); reject failed runs."""
        raise NotImplementedError


class ExaAgent(HostedAgent):
    """Exa's asynchronous Agent run API, using JSON creation and GET polling."""

    provider = "exa-agent"
    base_url = "https://api.exa.ai"
    key_names = ("EXA_API_KEY",)
    base_delay = 5.0

    def __init__(self, *, effort: str = "auto", **kwargs: Any):
        if effort not in ("minimal", "low", "base", "medium", "high", "xhigh", "auto", "ultra"):
            raise ValueError(f"Unsupported Exa Agent effort: {effort}")
        self.effort = effort
        super().__init__(**kwargs)

    async def fetch(self, prompt: str, request: dict, state: dict, path: Path | None) -> dict:
        if state.get("run_id"):
            raw = await self.poll(f"/agent/runs/{quote(state['run_id'], safe='')}")
        else:
            body: dict[str, Any] = {"query": prompt}
            if self.effort != "auto":
                body["effort"] = self.effort
            if request.get("instructions"):
                body["systemPrompt"] = request["instructions"]
            if request.get("output_schema"):
                body["outputSchema"] = request["output_schema"]
            raw = await self.create("/agent/runs", body, state, path)
            state["run_id"] = raw["id"]
            state.pop("creating", None)
            self.save(path, state)
        while raw.get("status") not in ("completed", "failed", "cancelled"):
            await asyncio.sleep(self.poll_interval_s)
            raw = await self.poll(f"/agent/runs/{quote(state['run_id'], safe='')}")
        return raw

    def unpack(self, raw: dict) -> tuple[Any, float | None, float]:
        if raw.get("status") != "completed":
            raise AgentRequestError(
                f"Exa run {raw.get('id')} {raw.get('status')}: {raw.get('error')}"
            )
        output = raw.get("output") or {}
        answer = output.get("structured")
        if answer is None:
            answer = output.get("text", "")
        cost = raw.get("costDollars") or {}
        return answer, dollars(cost.get("total")), 0.0


class ParallelTask(HostedAgent):
    """Parallel's Task API, with long polling and explicitly configured run pricing."""

    provider = "parallel-task"
    base_url = "https://api.parallel.ai"
    key_names = ("PARALLEL_API_KEY", "PARALLELS_API_KEY")
    base_delay = 10.0

    def __init__(
        self, *, processor: str = "core", cost_per_run_usd: float | None = None, **kwargs: Any
    ):
        self.processor = processor
        self.cost_per_run_usd = dollars(cost_per_run_usd)
        super().__init__(**kwargs)

    async def fetch(self, prompt: str, request: dict, state: dict, path: Path | None) -> dict:
        if not state.get("run_id"):
            body: dict[str, Any] = {"input": prompt, "processor": self.processor}
            if request.get("output_spec"):
                body["task_spec"] = {"output_schema": request["output_spec"]}
            elif request.get("output_schema"):
                body["task_spec"] = {
                    "output_schema": {"type": "json", "json_schema": request["output_schema"]}
                }
            raw = await self.create("/v1/tasks/runs", body, state, path)
            state["run_id"] = raw["run_id"]
            state.pop("creating", None)
            self.save(path, state)
        return await self.poll(
            f"/v1/tasks/runs/{quote(state['run_id'], safe='')}/result", parallel=True
        )

    def unpack(self, raw: dict) -> tuple[Any, float | None, float]:
        if raw.get("run", {}).get("status") != "completed":
            raise AgentRequestError(f"Parallel task did not complete: {raw.get('run')}")
        return raw["output"]["content"], self.cost_per_run_usd, 0.0


class PerplexityAgent(HostedAgent):
    """Perplexity's synchronous Agent API through its Responses-compatible alias."""

    provider = "perplexity-agent"
    base_url = "https://api.perplexity.ai"
    key_names = ("PERPLEXITY_API_KEY",)

    def __init__(self, *, preset: str = "low", max_output_tokens: int = 16000, **kwargs: Any):
        self.preset = preset
        self.max_output_tokens = max_output_tokens
        super().__init__(**kwargs)

    async def fetch(self, prompt: str, request: dict, state: dict, path: Path | None) -> dict:
        body: dict[str, Any] = {
            "input": prompt,
            "preset": self.preset,
            "max_output_tokens": self.max_output_tokens,
        }
        if request.get("instructions"):
            body["instructions"] = request["instructions"]
        if request.get("output_schema"):
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "findallrows", "schema": request["output_schema"]},
            }
        return await self.create("/v1/responses", body, state, path)

    def unpack(self, raw: dict) -> tuple[Any, float | None, float]:
        if raw.get("status") != "completed":
            raise AgentRequestError(f"Perplexity response {raw.get('id')} {raw.get('status')}")
        answer = "".join(
            part.get("text", "")
            for item in raw.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
        cost = (raw.get("usage") or {}).get("cost")
        if isinstance(cost, dict):
            return (
                answer,
                dollars(cost.get("total_cost")),
                dollars(cost.get("tool_calls_cost")) or 0.0,
            )
        return answer, dollars(cost), 0.0


def build_agent(provider: str, options: dict[str, Any]) -> HostedAgent:
    return {
        "exa-agent": ExaAgent,
        "parallel-task": ParallelTask,
        "perplexity-agent": PerplexityAgent,
    }[provider](**options)

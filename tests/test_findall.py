"""Offline release contracts: provider protocols, resumption, grading and fleet metrics."""

import hashlib
import json
from collections import Counter

import httpx
import pytest

from benchmarks.company_findall import CompanyFindAll
from benchmarks.graders.findall import format_candidate
from data import loaders
from harness import cli
from harness.agents import AgentRequestError, ExaAgent, ParallelTask, PerplexityAgent, parse_rows
from harness.comparison import compare_findall
from harness.llm.judge import Judge
from harness.llm.types import Generation
from harness.runner import Runner, TaskOutcome, summarize, write_json
from harness.suite import Grade, Task
from harness.systems import Catalog, System


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def completed_exa(rows=None):
    return {
        "id": "r1",
        "status": "completed",
        "output": {"structured": {"rows": rows or []}},
        "costDollars": {"total": 0.023},
    }


def test_release_manifest_and_public_schema():
    root = loaders.dataset_path("company-findall").parent
    manifest = json.loads((root / "manifest.json").read_text())
    rows = loaders.load_rows("company-findall")
    assert len(rows) == len({row["query_id"] for row in rows}) == manifest["queries"] == 300
    assert Counter(len(row["criteria"]) for row in rows) == {1: 26, 2: 60, 3: 98, 4: 71, 5: 45}
    assert sum(len(row["criteria"]) for row in rows) == manifest["criteria"]
    assert hashlib.sha256((root / manifest["file"]).read_bytes()).hexdigest() == manifest["sha256"]
    for row in rows:
        assert set(row) == {"query_id", "query", "entity_type", "criteria"}
        assert row["entity_type"] == "company"
        assert row["query"].startswith("Find all companies matching: ")
    for name, digest in manifest["contract_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("value", ["not JSON", '{"rows":null}', '{"rows":[1]}', [], {}])
def test_malformed_answers_are_zero_rows_with_diagnostics(value):
    rows, error = parse_rows(value)
    assert rows == [] and error


def test_rows_preserve_all_candidates_and_evidence_beyond_requested_count():
    payload = {"rows": [{"name": str(i), "evidence": [{"claim": "proof"}]} for i in range(30)]}
    rows, error = parse_rows("```json\n" + json.dumps(payload) + "\n```")
    assert not error and rows == payload["rows"]


@pytest.mark.parametrize("effort", ["auto", "minimal", "base", "xhigh"])
async def test_exa_request_poll_cost_and_cached_resume(tmp_path, effort):
    calls = []

    def handler(req):
        calls.append(req)
        assert req.headers["x-api-key"] == "test"
        if req.method == "POST":
            body = json.loads(req.content)
            assert body["query"] == "public query"
            assert body["systemPrompt"] == "instructions"
            assert body["outputSchema"] == {"type": "object"}
            assert body.get("effort") == (None if effort == "auto" else effort)
            assert "Exa-Beta" not in req.headers
            return httpx.Response(200, json={"id": "r1", "status": "queued"})
        assert req.url.path == "/agent/runs/r1"
        return httpx.Response(200, json=completed_exa([{"name": "company", "evidence": []}]))

    agent = ExaAgent(api_key="test", effort=effort, client=client(handler), poll_interval_s=0)
    path = tmp_path / "state.json"
    request = {"instructions": "instructions", "output_schema": {"type": "object"}}
    result = await agent.run("public query", request=request, state_path=path)
    assert result["rows"][0]["name"] == "company"
    assert result["total_cost_usd"] == 0.023 and result["cost_known"]
    assert await agent.run("public query", request=request, state_path=path) == result
    assert len(calls) == 2
    await agent.close()


async def test_exa_resumes_accepted_run_and_retries_transient_poll(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"run_id": "r1", "latency_ms": 250.0}))
    attempts = 0

    def handler(req):
        nonlocal attempts
        assert req.method == "GET"
        attempts += 1
        if attempts == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=completed_exa())

    agent = ExaAgent(api_key="test", client=client(handler), poll_interval_s=0)
    result = await agent.run("q", state_path=path)
    assert attempts == 2
    assert result["latency_ms"] >= 250.0
    await agent.close()


@pytest.mark.parametrize("failure", ["disconnect", 408, 500])
async def test_ambiguous_post_is_not_repeated_even_after_restart(tmp_path, failure):
    calls = 0

    def handler(req):
        nonlocal calls
        calls += 1
        if failure != "disconnect":
            return httpx.Response(failure)
        raise httpx.ReadTimeout("disconnected after accepting request", request=req)

    agent = ExaAgent(api_key="test", client=client(handler))
    path = tmp_path / "state.json"
    with pytest.raises(AgentRequestError):
        await agent.run("q", state_path=path)
    with pytest.raises(AgentRequestError, match="unknown outcome"):
        await agent.run("q", state_path=path)
    assert calls == 1
    await agent.close()


async def test_explicit_rate_limit_can_retry_without_duplicating_accepted_run(tmp_path):
    calls = 0

    def handler(req):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429)
        return httpx.Response(200, json=completed_exa())

    agent = ExaAgent(api_key="test", client=client(handler))
    agent.base_delay = 0
    await agent.run("q", state_path=tmp_path / "state.json")
    assert calls == 2
    await agent.close()


@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_terminal_failure_is_cached_without_reposting(tmp_path, status):
    calls = 0

    def handler(req):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"id": "r1", "status": status})

    agent = ExaAgent(api_key="test", client=client(handler))
    for _ in range(2):
        with pytest.raises(AgentRequestError, match=status):
            await agent.run("q", state_path=tmp_path / "state.json")
    assert calls == 1
    await agent.close()


async def test_parallel_contract_and_long_poll_retry(tmp_path):
    calls = []
    spec = CompanyFindAll().agent_request()

    def handler(req):
        calls.append(req)
        if req.method == "POST":
            assert req.url.path == "/v1/tasks/runs"
            assert json.loads(req.content) == {
                "input": "q",
                "processor": "ultra",
                "task_spec": {"output_schema": spec["output_spec"]},
            }
            return httpx.Response(200, json={"run_id": "p1"})
        assert req.url.path == "/v1/tasks/runs/p1/result"
        assert req.url.params["timeout"] == "30"
        if len(calls) == 2:
            return httpx.Response(408, json={"error": {"message": "run still active"}})
        return httpx.Response(
            200,
            json={
                "run": {"status": "completed"},
                "output": {"content": '{"rows":[{"name":"A"}]}', "basis": []},
            },
        )

    agent = ParallelTask(
        api_key="test",
        processor="ultra",
        cost_per_run_usd=0.3,
        client=client(handler),
        poll_interval_s=0,
    )
    result = await agent.run("q", request=spec, state_path=tmp_path / "state.json")
    assert result["rows"] == [{"name": "A"}]
    assert result["total_cost_usd"] == 0.3 and result["cost_source"] == "configured_per_run"
    assert len(calls) == 3
    await agent.close()


@pytest.mark.parametrize(
    "cost,known,total",
    [
        ({"total_cost": 0.01, "tool_calls_cost": 0.002}, True, 0.01),
        (None, False, 0.0),
        ({"tool_calls_cost": 0.002}, False, 0.002),
    ],
)
async def test_perplexity_preset_structured_output_and_usage(cost, known, total):
    spec = CompanyFindAll().agent_request()

    def handler(req):
        body = json.loads(req.content)
        assert req.url.path == "/v1/responses"
        assert req.headers["Authorization"] == "Bearer test"
        assert body["preset"] == "low" and body["max_output_tokens"] == 16000
        assert "model" not in body and "tools" not in body
        assert body["instructions"] == spec["instructions"]
        assert body["response_format"]["json_schema"]["schema"] == spec["output_schema"]
        return httpx.Response(
            200,
            json={
                "id": "px",
                "status": "completed",
                "usage": {"cost": cost},
                "output": [
                    {"type": "web_search", "content": [{"type": "output_text", "text": "ignore"}]},
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"rows":[]}'}],
                    },
                ],
            },
        )

    agent = PerplexityAgent(api_key="test", client=client(handler))
    result = await agent.run("q", request=spec)
    assert result["cost_known"] is known and result["total_cost_usd"] == total
    assert result["model_cost_usd"] + result["search_cost_usd"] == pytest.approx(total)
    assert result["rows"] == [] and result["parse_error"] is None
    await agent.close()


class VerdictJudge:
    def __init__(self):
        self.calls = []

    async def complete_json(self, prompt, schema, *, system=None):
        self.calls.append((prompt, system))
        return schema(reasoning="supported", score=int('"passes": true' in prompt)), None


async def test_holistic_judge_one_call_per_returned_candidate_and_empty_no_calls():
    suite = CompanyFindAll()
    task = Task("q", "Find all companies matching: US; software", ["US", "software"])
    judge = VerdictJudge()
    rows = [{"passes": i % 2 == 0, "evidence": [{"claim": "proof"}]} for i in range(30)]
    grade = await suite.grade_result(task, {"rows": rows}, judge)
    assert len(judge.calls) == 30
    assert grade.scores["num_rows"] == 30
    assert grade.scores["num_passed"] == 15 and grade.scores["pass_rate"] == 0.5
    assert grade.details["candidates"][-1]["rank"] == 30
    assert suite.failure_scores({"rows": rows})["num_rows"] == 30
    prompt, system = judge.calls[0]
    assert "Companies matching all of:\n- US\n- software" in prompt
    assert '"claim": "proof"' in prompt and "Use 0 when uncertain" in system
    empty = await suite.grade_result(task, {"rows": []}, judge)
    assert empty.scores["zero_entities"] == 1 and len(judge.calls) == 30


async def test_system_keeps_gold_out_of_agent_request(monkeypatch):
    captured = {}

    class Agent:
        async def run(self, prompt, **kwargs):
            captured.update(prompt=prompt, **kwargs)
            return {"answer": "ok"}

    monkeypatch.setattr("harness.systems.build_agent", lambda *args: Agent())
    system = System(Catalog.load().resolve("exa-agent-auto"))
    task = Task("q", "public text", ["HIDDEN GOLD"], {"secret": "HIDDEN GOLD"})
    await system.execute(task, CompanyFindAll())
    assert captured["prompt"] == "public text"
    assert "HIDDEN GOLD" not in str(captured)


async def write_run(root, name, counts):
    suite = CompanyFindAll()
    tasks = suite.load()[: len(counts)]

    class SystemStub:
        async def execute(self, task, suite):
            return {"rows": [], "answer": "", "cost_known": True, "total_cost_usd": 0}

        async def close(self):
            pass

    runner = Runner(
        Catalog.load().resolve(name), suite, judge=Judge(), runs_root=root, system=SystemStub()
    )
    await runner.run(tasks)
    for task, (passed, rows) in zip(tasks, counts, strict=True):
        scores = {
            "num_passed": passed,
            "num_rows": rows,
            "pass_rate": passed / rows if rows else 0,
            "zero_entities": float(rows == 0),
        }
        write_json(runner.task_dir(task) / "grade.json", {"scores": scores})
    return runner.run_dir


async def test_fleet_normalization_is_per_query_and_entity_precision_is_weighted(tmp_path):
    a = await write_run(tmp_path, "exa-agent-auto", [(2, 2), (1, 4), (0, 0)])
    b = await write_run(tmp_path, "parallel-task-core", [(1, 4), (2, 2), (0, 0)])
    report = compare_findall([a, b])
    assert len(report["fleet"]) == 2
    for summary in report["summaries"]:
        assert summary["metrics"]["normalized_num_passed"] == 0.5
        assert summary["metrics"]["entity_precision"] == 0.5
        assert summary["metrics"]["zero_entities"] == pytest.approx(1 / 3)
        assert summary["confidence_intervals"]["metrics"]["normalized_num_passed"]["n"] == 3
    other_subset = await write_run(tmp_path, "exa-agent-low", [(1, 1)])
    with pytest.raises(ValueError, match="identical task IDs"):
        compare_findall([a, other_subset])
    with pytest.raises(ValueError, match="Duplicate"):
        compare_findall([a, a])


@pytest.mark.parametrize("missing", ["result.json", "grade.json"])
async def test_comparison_rejects_incomplete_artifacts(tmp_path, missing):
    a = await write_run(tmp_path, "exa-agent-auto", [(1, 1)])
    b = await write_run(tmp_path, "parallel-task-core", [(1, 1)])
    task_id = CompanyFindAll().load()[0].id
    (a / "tasks" / task_id / missing).unlink()
    with pytest.raises(ValueError, match="Incomplete task artifacts"):
        compare_findall([a, b])


async def test_rebuilding_summary_preserves_last_completed_query_selection(tmp_path):
    directory = await write_run(tmp_path, "exa-agent-auto", [(1, 1), (1, 1), (1, 1)])
    await write_run(tmp_path, "exa-agent-auto", [(1, 1)])
    args = cli.build_parser().parse_args(["summary", str(directory)])
    assert cli.cmd_summary(args) == 0
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["tasks"] == 1
    assert summary["task_ids"] == [CompanyFindAll().load()[0].id]


def test_failures_count_zero_and_do_not_drop_returned_ungraded_rows():
    suite = CompanyFindAll()
    tasks = suite.load()[:3]
    outcomes = [
        TaskOutcome(
            tasks[0],
            {"rows": [{}, {}]},
            Grade({"num_rows": 2, "num_passed": 1, "zero_entities": 0}),
            None,
        ),
        TaskOutcome(tasks[1], None, None, "search error"),
        TaskOutcome(tasks[2], {"rows": [{}, {}]}, None, "judge error"),
    ]
    config = {"system": {"name": "test"}, "suite_revision": suite.revision, "judge_model": "test"}
    summary = summarize(suite, outcomes, config)
    assert summary["metrics"]["num_passed"] == pytest.approx(1 / 3)
    assert summary["metrics"]["entity_precision"] == 0.25
    assert summary["metrics"]["zero_entities"] == pytest.approx(1 / 3)
    assert summary["graded"] == 1 and summary["failed"] == 2


async def test_runner_does_not_retry_ambiguous_agent_errors(tmp_path):
    calls = 0

    class FailingSystem:
        async def execute(self, task, suite):
            nonlocal calls
            calls += 1
            raise AgentRequestError("accepted run has unknown outcome")

        async def close(self):
            pass

    suite = CompanyFindAll()
    runner = Runner(
        Catalog.load().resolve("exa-agent-auto"),
        suite,
        judge=Judge(),
        runs_root=tmp_path,
        system=FailingSystem(),
    )
    summary = await runner.run(suite.load()[:1])
    assert calls == 1 and summary["failed"] == 1


def test_candidate_evidence_bounds_do_not_drop_identity():
    row = {
        "name": "Company",
        "canonical_url": "https://example.com",
        "summary": "a" * 5000,
        "evidence": [{"claim": f"proof{i}", "raw": "hidden"} for i in range(20)],
    }
    formatted = format_candidate(row)
    assert formatted.startswith("URL: https://example.com\nTitle: Company\n\n")
    assert "proof11" in formatted and "proof12" not in formatted and "hidden" not in formatted
    assert "a" * 4001 not in formatted
    assert len(formatted.split("\n\n", 1)[1]) <= 8192


@pytest.mark.parametrize("row_count", [1, 30])
async def test_real_agent_runner_grade_cost_and_resume(tmp_path, monkeypatch, row_count):
    provider_calls, judge_calls = [], []

    def handler(req):
        provider_calls.append(req)
        return httpx.Response(
            200,
            json=completed_exa(
                [
                    {"name": f"Company {i}", "canonical_url": f"https://company{i}.example"}
                    for i in range(row_count)
                ]
            ),
        )

    class JudgeClient:
        supports_response_schema = True

        async def generate(self, messages, **kwargs):
            judge_calls.append(kwargs)
            return Generation(text='{"reasoning":"yes","score":1}', cost_usd=0.001)

    monkeypatch.setattr(
        "harness.systems.build_agent",
        lambda *args: ExaAgent(api_key="test", client=client(handler)),
    )
    suite = CompanyFindAll()
    spec = Catalog.load().resolve("exa-agent-low")
    judge = Judge(client=JudgeClient(), **suite.judge_settings)
    first = Runner(spec, suite, judge=judge, runs_root=tmp_path)
    summary = await first.run(suite.load()[:1])
    assert summary["metrics"]["num_passed"] == row_count
    assert summary["metrics"]["num_rows"] == row_count
    assert summary["cost"]["system_usd"] == 0.023
    assert summary["cost"]["judge_usd"] == pytest.approx(0.001 * row_count)
    assert judge_calls[0]["reasoning_effort"] == "low"
    assert judge_calls[0]["temperature"] == 0.0
    assert judge_calls[0]["max_output_tokens"] == 2048
    # Simulate a lost result/grade after the remote response was checkpointed.
    task_dir = first.task_dir(suite.load()[0])
    (task_dir / "result.json").unlink()
    (task_dir / "grade.json").unlink()
    await Runner(spec, suite, judge=judge, runs_root=tmp_path).run(suite.load()[:1])
    assert len(provider_calls) == 1 and len(judge_calls) == 2 * row_count
    await Runner(spec, suite, judge=judge, runs_root=tmp_path).run(suite.load()[:1])
    assert len(provider_calls) == 1 and len(judge_calls) == 2 * row_count


def test_agent_catalog_credentials_and_dry_run(monkeypatch):
    for key in ("EXA_API_KEY", "PARALLEL_API_KEY", "PERPLEXITY_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(key, "test")
    catalog = Catalog.load()
    names = [name for name in catalog.systems if catalog.resolve(name).kind == "agent"]
    assert len(names) == 11
    for name in names:
        assert not cli.missing_credentials(catalog.resolve(name), "openai/gpt-6-luna")
    args = cli.build_parser().parse_args(
        ["run", "--suite", "company-findall", "--system", *names, "--limit", "1", "--dry-run"]
    )
    assert cli.cmd_run(args) == 0

# Company FindAll (synthetic)

Exa's company list-building benchmark: **300 synthetic GTM queries and 949 criteria**.
A pointwise judge scores **every returned company** against the entire query,
with no evaluation row cap. The bundled agent instructions request up to 25
companies with evidence; any additional returned companies are also graded.

`queries.jsonl` contains only `query_id`, `query`, `entity_type`, and `criteria`.
The [manifest](manifest.json) records the release hash, counts and output contract.

## Run

From the repository root, run `uv sync --locked`. Set `OPENAI_API_KEY` for the judge
and the keys for the products you select: `EXA_API_KEY`, `PARALLEL_API_KEY`, and/or
`PERPLEXITY_API_KEY`.

```bash
uv run bench run --suite company-findall \
  --system exa-agent-low parallel-task-core perplexity-agent-pro \
  --judge-model openai/gpt-6-luna --limit 5 --concurrency 3 \
  --output results/company-findall.json
```

Add `--dry-run` to validate without paid calls. Remove `--limit` for all 300
queries. The judge defaults to `openai/gpt-6-luna`, with low reasoning effort and
a 2,048-token output limit. Temperature is configured as zero; the existing
OpenAI client omits temperature for reasoning models. `--judge-model` can select
a different judge, but those results should be reported separately.

The complete historical fleet is available:

```bash
uv run bench run --suite company-findall \
  --system exa-agent-minimal exa-agent-low exa-agent-base exa-agent-medium \
    exa-agent-high exa-agent-xhigh exa-agent-auto \
    parallel-task-base parallel-task-core parallel-task-ultra perplexity-agent-pro \
  --concurrency 12 --output results/company-findall-fleet.json
```

A full historical search cost roughly $850–900 and took about six hours;
grading was additional. These are historical estimates, not spending limits.
Start with a small subset. The `exa-agent-base` preset preserves the historical
`effort: base` request, but **base is absent from the current documented Exa
effort list**. It may require legacy API support; the adapter never silently
maps it to another tier. See [Exa's run API](https://exa.ai/docs/reference/agent-api/create-a-run).

## Agent contract

Hosted products run their own research loops, exposed as `kind = "agent"` systems.
They receive the query and public output instructions, never the judge rubric.
Their models and research budgets are provider-controlled, so this evaluates
each complete product. Scout continues to evaluate search tools in a shared loop.
Hosted agents also support the answer-based public suites.

Every Company FindAll response requests `{"rows": [...]}`. Each row contains
`name`, `entity_type`, `canonical_url`, `summary`, `attributes`, and `evidence`
(source URL, title and claim). The prompts request distinct companies and prohibit
padding. The harness preserves and grades **all returned rows**, without
deduplication or further page fetching. The requested 25-company output budget
is not an evaluator cap. Invalid JSON or an invalid rows container yields an empty
list with a saved parse diagnostic. Provider failures
are recorded as failed tasks, separately from valid empty answers.

- Exa uses the [Agent run API](https://exa.ai/docs/reference/agent-api/create-a-run),
  the bundled instructions and JSON schema; auto omits the effort parameter.
- Parallel uses the [Task API](https://docs.parallel.ai/api-reference/tasks/create-task-run),
  with the bundled text output specification and base/core/ultra processors.
- Perplexity uses the [Agent API's Responses alias](https://docs.perplexity.ai/docs/agent-api/quickstart),
  the same instructions and JSON schema as Exa, `preset: low`, and 16,000 output
  tokens. The historical arm name `perplexity-agent-pro` is a label, not a model
  or preset named pro.

Adapters use the repository's existing HTTP client dependency. All have a
one-hour deadline, three retries for rejected rate-limited creations, and a
three-request-per-second creation limit. Concurrency caps are 12 for Exa,
3 for Parallel base/core and Perplexity, and 10 for Parallel ultra; the runner's
`--concurrency` can lower them. Systems execute sequentially within a fleet.
Accepted Exa/Parallel run IDs and completed raw responses are checkpointed in
`results/runs/.../agent_runs/`; restarts poll the same accepted runs. GET polling
retries transient failures. An ambiguous creation timeout is recorded and is not
automatically reposted, to avoid duplicate charges. Inspect its checkpoint and
provider dashboard before explicitly starting a new run with `--run-suffix`.
Latency includes active wait time across polling attempts, excluding downtime
between process restarts and time spent replaying cached responses.

## Scoring and comparisons

The [holistic grader](../../benchmarks/company_findall.py) uses the handoff's
single `candidate_satisfies_findall_query` criterion. It judges each candidate's
supplied JSON against all criteria, without web access, and uses zero when
uncertain. Every returned entity receives a judgment, regardless of its rank or
the requested output count. Per-candidate scores and reasoning are saved in task
artifacts.
The source grader's evidence bounds are preserved: strings are capped at 4,000
characters, nested lists at 12 items, and the formatted JSON body at 8,192
characters. The candidate URL and title are included separately.

| Metric | Definition |
|--------|------------|
| `num_rows` | Mean returned companies per query, including every returned row |
| `num_passed` | Mean passing companies per query, counting every returned row |
| `normalized_num_passed` | Per-query passed count divided by the best passed count in the selected fleet, then averaged over queries |
| `pass_rate`, `criteria_pass_rate` | Mean per-query fraction of returned companies passing; empty queries score zero |
| `entity_precision` | Total passing companies divided by total returned companies |
| `zero_entities` | Share of queries returning zero companies |

Search failures contribute zero rows and passes. Grading failures contribute zero
passes while retaining any returned row count. All-zero fleet queries have a
normalized score of zero. These explicit failure conventions apply to new public
runs; the aggregate-only historical export does not fully specify its denominators.

A single-system run reports `num_passed` as its primary metric. Running multiple
systems together also produces fleet-normalized summaries in `--output`. Compare
previously completed runs without API calls:

```bash
uv run bench compare results/runs/<exa-run> results/runs/<parallel-run> \
  results/runs/<perplexity-run> --output results/company-findall-comparison.json
```

Comparisons require identical task selections, dataset/grader revisions and judge
models, with a completed grade or recorded failure for every task. Rebuilding a
summary preserves the most recent completed query selection, even if its run
directory contains older results for additional queries. Comparisons save the
full resolved fleet. **Normalized scores depend on the
fleet**: do not compare scores from different fleets or interpret them as recall
against a complete list of companies. Standalone task grades remain reusable.

New runs report 95% bootstrap confidence intervals using 10,000 whole-query
resamples (seed 0). Entity precision resamples paired passing/returned counts;
normalized scores use the same query alignment across the fixed fleet. These
intervals measure task-sampling uncertainty, not repeated-run variation.

Costs separate provider execution from judge spend. Exa and Perplexity preserve
provider-reported costs; absent costs are marked unknown. Parallel uses explicit
per-run estimates in `systems.toml`: base $0.010, core $0.025, ultra $0.300, from
[published pricing](https://docs.parallel.ai/getting-started/pricing) checked on
2026-10-02. These differ from the historical costs below. Raw provider responses
are saved for auditing.

## Historical reference results

These are the supplied synthetic-query results regraded with the holistic
`gpt-6-luna` judge. They are **not measurements made by this public harness**.
The historical evaluation graded only the first 25 rows; the public evaluator
grades all rows, so results can differ when an agent returns more than requested.
The source reports 298 regrade queries from the 300-query release and 287 degraded
errors; per-task artifacts were not supplied, so exact per-arm denominators,
failure handling, and confidence intervals cannot be reconstructed. Costs are
approximate historical provider dollars per task, excluding judge spend.
Machine-readable values are in [reference_results.json](reference_results.json).

| System | Normalized passed | Passed / query | Pass rate | Approx. $ / task |
|--------|------------------:|---------------:|----------:|----------------:|
| exa-agent-auto | 0.849 | 16.640 | 0.853 | 0.940 |
| exa-agent-xhigh | 0.841 | 16.470 | 0.871 | 0.940 |
| exa-agent-high | 0.786 | 15.483 | 0.875 | 0.470 |
| parallel-task-ultra | 0.643 | 14.000 | 0.759 | 0.270 |
| exa-agent-medium | 0.448 | 9.130 | 0.671 | 0.094 |
| exa-agent-minimal | 0.396 | 7.690 | 0.728 | 0.011 |
| parallel-task-core | 0.394 | 8.910 | 0.691 | 0.025 |
| exa-agent-low | 0.389 | 7.553 | 0.775 | 0.023 |
| exa-agent-base | 0.386 | 7.643 | 0.768 | 0.048 |
| perplexity-agent-pro | 0.264 | 5.517 | 0.689 | 0.010 |
| parallel-task-base | 0.259 | 5.737 | 0.635 | 0.005 |

## Provenance and license

Exa generated synthetic twins of an internal company-prospecting benchmark using
`gpt-6-luna`: preserving entity type, criterion count and constraint categories
while generating new industries, locations, numeric thresholds and wording.
Eight rows are additional twins so the release contains exactly 300 queries.
There are 26, 60, 98, 71 and 45 queries with one through five criteria, respectively.
This preserves a task distribution; it does not provide exhaustive ground truth
or establish that the synthetic set is equally difficult as its source.

The public export omits original customer queries, customer/account identifiers,
internal pairing fields, internal URLs, generation metadata and the original
benchmark's comparison results. Dataset and harness are distributed under the
repository's [MIT license](../../LICENSE).

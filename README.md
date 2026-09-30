# Exa Search Benchmarks

Open benchmarks and an open evaluation harness for web search APIs.

- **[Evaluation harness](#evaluation-harness)** (`bench`) runs any search API on public
  benchmarks (BrowseComp, FRAMES, DeepSearchQA, WideSearch) inside the same
  research agent, so the only thing that differs between systems is the search API.
- **[Exa benchmarks](#benchmarks)** are task-specific datasets we built (code docs, people,
  companies, publications), run through the same harness.

## Evaluation harness

Comparing search APIs is only fair when everything around the search call is held fixed.
The harness defines the following execution modes in [`systems.toml`](systems.toml):

| Kind | What runs | Measures |
|------|-----------|----------|
| **Scout** over a search API | One research agent (Scout) with one `search` tool bound to the API under test. Model, prompts and budgets are shared by every Scout system. | How much the search API helps an agent |
| **Native search** | The same Scout loop, but the model provider's own hosted web search (OpenAI `web_search`, Anthropic `web_search`) replaces the search tool. | A model vendor's built-in search, in the same loop |
| **Single-step RAG** | One search with the question as the query, then one answer from only those results. | Retrieval quality of a single request |
| **Search** | One ranked retrieval request, scored directly. | People, company and publication retrieval |
| **Extract + RAG** | Extract evidence from the task URL, then synthesize once. | WebCode Highlights |

Built-in systems (`uv run bench list` prints the live catalog):

| Search API | Scout systems |
|------------|---------------|
| Exa highlights | `scout-exa-instant-highlights`, `scout-exa-fast-highlights`, `scout-exa-auto-highlights` |
| Perplexity Search | `scout-perplexity-web`, `scout-perplexity-fast` |
| Parallel Search | `scout-parallel-turbo`, `scout-parallel-fast`, `scout-parallel-basic`, `scout-parallel-advanced` |
| Brave LLM Context | `scout-brave-llm-context` |
| OpenAI hosted web search | `openai-native-search`, `openai-native-search-luna` |
| Anthropic hosted web search | `anthropic-native-search` |

Each of the ten API presets also has `rag-` and `search-` variants. Brave uses only
LLM Context; its `/web/search` endpoint is not supported.

Scout and RAG systems default to `openai/gpt-5.6-luna`; pass `--model` to run any of them
with another OpenAI or Anthropic model (native-search systems stay on their own provider).

### Suites

| Suite | Tasks | Primary metric | Source |
|-------|------:|----------------|--------|
| `browsecomp` | 1,266 | accuracy (`score`) | [BrowseComp](https://openai.com/index/browsecomp/) |
| `frames` | 824 | accuracy (`score`) | [google/frames-benchmark](https://huggingface.co/datasets/google/frames-benchmark) |
| `dsqa` | 900 | `f1` | [google/deepsearchqa](https://huggingface.co/datasets/google/deepsearchqa) |
| `widesearch` | 200 | `f1_by_row` | [ByteDance-Seed/WideSearch](https://huggingface.co/datasets/ByteDance-Seed/WideSearch) |

The four public suites above accept Scout and single-step RAG systems. Their data is
downloaded at pinned revisions and verified. The repository datasets below use the
same runner, grading interface, resume artifacts and cost summaries:

| Suite | Tasks | System kind | Primary metric |
|-------|------:|-------------|----------------|
| `people` | 1,400 | `search` | `recall_at_10` |
| `company-retrieval` | 605 | `search` | `recall_at_10` |
| `company-rag` | 234 | `scout`, `rag` | `accuracy` |
| `publication` | 1,472 | `search` | `recall_at_10` |
| `publication-tot` | 394 | `search` | `recall_at_10` |
| `webcode-rag` | 307 | `rag` | `grounded` |
| `webcode-highlights` | 250 | `extract-rag` | `grounded` |

The judge defaults to `openai/gpt-5.6-luna` (`--judge-model` overrides it).
Publication grading is deterministic and needs no model key. Empty retrievals
receive zero recall and precision. WebCode E2E remains a **dataset-only export**
of 33 tasks, outside the runnable suite catalog; it has no coding-agent executor
or bundled setup files. The Contents track has been removed because its licensed
reference data is unavailable.

### Quick start

```bash
git clone https://github.com/exa-labs/benchmarks.git
cd benchmarks
uv sync --all-packages --all-groups --locked

export OPENAI_API_KEY=...        # Scout model and the judge
export EXA_API_KEY=...           # plus a key for each search API you run:
# BRAVE_SEARCH_API_KEY, PARALLEL_API_KEY, PERPLEXITY_API_KEY, ANTHROPIC_API_KEY

uv run bench list
uv run bench run --system scout-exa-auto-highlights --suite browsecomp --limit 5
uv run bench run --system scout-brave-llm-context --suite browsecomp --limit 5
uv run bench run --system scout-exa-auto-highlights --model anthropic/claude-sonnet-5 --suite dsqa --limit 20
```

These commands make paid API calls. `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` point the model
clients at any compatible gateway. `uv run bench download` fetches and verifies every suite
up front.

Run every runnable suite with compatible Exa systems (one task per suite first):

```bash
uv run bench run --suite all \
  --system rag-exa-auto-highlights search-exa-auto-highlights extract-rag-exa-extract \
  --limit 1 --dry-run
# Remove --dry-run to execute; remove --limit for the full datasets.
```

Both selectors accept multiple names or `all`; the runner executes compatible pairs.
All selected suites must have a compatible system. Data, selection and required
credentials are checked for the entire plan before the first paid call. `--limit`
applies per suite; `--output` saves a JSON list of run summaries. The legacy commands
below translate into this same CLI, with no separate execution loops.

### How Scout works

Scout gives the model two tools: `search` and `submit_final_result`. Every search API gets the
same `search({query})` definition, except Parallel, whose API takes a natural-language
`objective` plus three keyword queries in one request; its tool follows Parallel's published
tool definition. Results come back as a numbered list of title, URL and highlighted text.

- The run ends when the model calls `submit_final_result`. A turn without a tool call gets a
  nudge; three nudges end the research phase.
- Budgets: 25 turns and one hour of wall clock by default; search-count and USD caps are
  optional. An exhausted budget triggers one tools-disabled synthesis turn over the evidence
  gathered so far, and the result records its `stop_reason`.
- If the conversation outgrows the context window, the oldest completed exchange is dropped.

### Run artifacts, resuming and cost

Each run writes `runs/{system}-{suite}[-{suffix}]-{hash}/`:

```text
config.json               resolved system, suite revision, judge model
tasks/<task-id>/result.json   answer, citations, costs and the full trajectory
tasks/<task-id>/grade.json    scores, grader reasoning, judge cost
tasks/<task-id>/cost.json     cumulative judge spend, including failed grading attempts
summary.json              metrics, failed-as-zero primary metric, cost per task, stop reasons
```

The hash covers the system settings, suite revision and judge model. Re-running the same
command resumes: graded tasks are skipped, answered-but-ungraded tasks are only re-graded, and
failed tasks are retried. `--run-suffix rep2` starts an independent repeat.

Cost per task is split into model tokens (priced from [`harness/harness/llm/pricing.py`](harness/harness/llm/pricing.py))
and search calls (the provider's reported cost when it returns one, otherwise its list price);
judge cost is reported separately. A model with no listed price is reported as unknown, not
zero. Answered tasks retain their costs even when grading fails; unsuccessful model
or search calls with unknown spend mark accounting incomplete. Grade retries accumulate
judge spend across resumed runs.

### Adding a search API

Implement `Searcher.run` in [`shared/shared/searchers/`](shared/shared/searchers/) (return
results plus the request's cost), register the provider in `build_searcher`
([`harness/harness/systems.py`](harness/harness/systems.py)), and add a `[searchers.*]` entry
and a `scout-*` system to `systems.toml`. Scout systems inherit `[defaults.scout]`, so the new
system is comparable with the rest without further changes.

### Third-party datasets

Benchmarks are downloaded from their upstream homes at pinned revisions and remain under their
own terms:

- BrowseComp is published by OpenAI through
  [simple-evals](https://github.com/openai/simple-evals) (MIT). It is encrypted
  upstream to keep it out of training data; do not republish decrypted questions or answers.
- FRAMES and DeepSearchQA are Apache-2.0.
- WideSearch data is CC0-1.0; the evaluator it adapts is MIT.

## Benchmarks

| Benchmark | Queries | Tracks | Description |
|-----------|---------|--------|-------------|
| [WebCode](webcode-benchmark/) | 557 + 33 | Highlights, RAG; E2E dataset only | Code documentation retrieval and grounded QA |
| [People Search](simple-people-benchmark/) | 1,400 | Retrieval | Find people profiles by role, location, seniority |
| [Company Search](simple-company-benchmark/) | ~800 | Retrieval + RAG | Find companies by name, industry, geography, funding |
| [Publication Retrieval](publication-benchmark/) | 1,866 | Publication, ToT | Find the exact publication by grounded question or tip-of-the-tongue recollection |

> The competitor rows below were measured with earlier versions of the provider adapters
> (Perplexity through its Sonar answer API, Parallel through the v1beta Search API, Brave LLM
> Context read from a `results` field rather than `grounding`). The adapters now call
> Perplexity's Search API, Parallel Search v1 and Brave's current response format; rerun a
> table with the commands under
> [Running the Exa benchmarks](#running-the-exa-benchmarks) before comparing against it.

## WebCode Results

**Highlights** — in-document retrieval given a URL + query (250 queries)

| Searcher | Groundedness | Correctness | Avg Tokens |
|----------|:---:|:---:|:---:|
| Exa | **94.8** | **93.2** | 696 |
| Parallel | 85.6 | 86.4 | 858 |
| Claude | 81.5 | 85.9 | **319** |

**RAG** — full-web retrieval + synthesis (307 queries)

| Searcher | Groundedness | Avg Tokens | Citation Prec. |
|----------|:---:|:---:|:---:|
| Exa | **79.4** | 688 | 0.259 |
| Brave | 76.3 | 1229 | **0.328** |
| Parallel | 75.3 | 622 | 0.168 |
| Perplexity | 64.6 | 754 | 0.220 |
| Tavily | 61.1 | 464 | 0.159 |

See [webcode-benchmark/](webcode-benchmark/) for details and [blog post](https://exa.ai/blog/web-code).

## People Search Results

| Searcher | R@1 | R@10 | Precision | Queries |
|----------|-----|------|-----------|---------|
| exa | **72.0%** | **94.5%** | **63.3%** | 1399 |
| brave | 44.4% | 77.9% | 30.2% | 1373 |
| parallel | 20.8% | 74.7% | 26.9% | 1387 |

## Company Search Results

Two tracks designed to separate retrieval from fact extraction.

**Retrieval Track** — Ranked lists of companies matching criteria (named lookup, attribute filtering, funding queries, composite constraints, semantic descriptions).

| Searcher | R@1 | R@5 | R@10 | Precision |
|----------|-----|-----|------|-----------|
| exa | **61.8%** | **90.6%** | **94.2%** | **65.9%** |
| brave | 35.9% | 61.8% | 72.9% | 39.2% |
| parallel | 36.6% | 66.3% | 78.6% | 40.4% |

**RAG Track** — Extract specific facts (founding year, employee count, funding rounds, founders). Static facts use exact-match; dynamic facts get ±20% tolerance.

| Searcher | Accuracy |
|----------|----------|
| exa | **79%** |
| brave | 65% |
| parallel | 66% |

## Publication Retrieval Results

**Tip-of-the-Tongue (ToT) Track**

| Searcher | Recall | MRR | Mean latency ± SEM |
|----------|-------:|----:|-------------------:|
| Exa | 86.4% | 0.726 | 0.578 ± 0.012 s |
| Perplexity | 66.8% | 0.568 | 1.277 ± 0.016 s |
| Parallel Advanced | 50.0% | 0.312 | 3.118 ± 0.082 s |

**Publication Track**

| Searcher | Recall | MRR | Mean latency ± SEM |
|----------|-------:|----:|-------------------:|
| Exa | 68.0% | 0.583 | 0.681 ± 0.034 s |
| Perplexity | 54.0% | 0.475 | 1.169 ± 0.012 s |
| Parallel Advanced | 52.0% | 0.349 | 2.924 ± 0.069 s |

## Running the Exa benchmarks

From the repository root after the workspace install above:

```bash
uv run pbench --searchers exa --limit 50
uv run cbench --limit 50                    # both company tracks
uv run cbench --track retrieval --split static
uv run cbench --track rag
uv run pubbench --limit 50                 # both publication tracks; no judge key needed
uv run pubbench --track tot --searchers exa brave parallel --output results.json
uv run python -m evals.highlights --searchers exa tavily parallel --limit 20
uv run python -m evals.rag --searchers exa brave perplexity --limit 20
uv run python -m evals.e2e --info           # inspect dataset only
```

Legacy provider aliases select task-specific presets in `systems.toml` (for example,
`pbench --searchers exa` selects `search-exa-people`). They accept `--dry-run`,
`--runs-dir`, `--run-suffix`, `--judge-model` and `--model`. Results now use the common
per-task artifact layout; `--output` writes aggregate summaries. Historical tables
above predate the unified model defaults and empty-result handling and must be rerun
for current comparisons.

## Implementing Your Own Searcher

All benchmarks use the same `Searcher` interface:

```python
from shared.searchers import Searcher, SearchResult

class MySearcher(Searcher):
    name = "my-search"
    
    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        response = await my_api.search(query, limit=num_results)
        return [
            SearchResult(url=r.url, title=r.title, text=r.snippet)
            for r in response.results
        ]
    
    async def extract(self, url: str, query: str | None = None) -> list[SearchResult]:
        content = await my_api.extract(url)
        return [SearchResult(url=url, text=content)]
```

The `search` method is used by retrieval and RAG evals. The `extract` method is used by the highlights eval for URL-based extraction. The evaluation harness calls `run`, which returns the same results plus the request's cost and latency; the default `run` wraps `search` and reports the cost as unknown, so override it for a provider with a known price (see [`exa.py`](shared/shared/searchers/exa.py)).

## Requirements

- Python 3.11+
- OpenAI API key (for LLM grading), or an Anthropic key for Anthropic-model systems
- Search API credentials

## License

MIT

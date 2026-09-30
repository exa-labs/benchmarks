# Exa Search Benchmarks

Open benchmarks and an open evaluation harness for web search APIs.

- **[Evaluation harness](#evaluation-harness)** (`bench`) runs any search API on public
  benchmarks (BrowseComp, SimpleQA, FRAMES, DeepSearchQA, WideSearch, HLE) inside the same
  research agent, so the only thing that differs between systems is the search API.
- **[Exa benchmarks](#benchmarks)** are task-specific datasets we built (code docs, people,
  companies, publications), each with its own runner.

## Evaluation harness

Comparing search APIs is only fair when everything around the search call is held fixed.
The harness defines three kinds of system in [`systems.toml`](systems.toml):

| Kind | What runs | Measures |
|------|-----------|----------|
| **Scout** over a search API | One research agent (Scout) with one `search` tool bound to the API under test. Model, prompts and budgets are shared by every Scout system. | How much the search API helps an agent |
| **Native search** | The same Scout loop, but the model provider's own hosted web search (OpenAI `web_search`, Anthropic `web_search`) replaces the search tool. | A model vendor's built-in search, in the same loop |
| **Single-step RAG** | One search with the question as the query, then one answer from only those results. | Retrieval quality of a single request |

Built-in systems (`uv run bench list` prints the live catalog):

| Search API | Scout | Single-step RAG |
|------------|-------|-----------------|
| Exa (`auto`, `fast`, `instant`; highlights) | `scout-exa-auto`, `scout-exa-fast`, `scout-exa-instant` | `rag-exa-auto`, `rag-exa-fast` |
| Brave (LLM Context) | `scout-brave` | `rag-brave` |
| Parallel Search (`advanced`, `fast`) | `scout-parallel-advanced`, `scout-parallel-fast` | `rag-parallel-advanced` |
| Perplexity Search API | `scout-perplexity` | `rag-perplexity` |
| OpenAI hosted web search | `openai-native-search`, `openai-native-search-luna` | — |
| Anthropic hosted web search | `anthropic-native-search` | — |

Scout and RAG systems default to `openai/gpt-5.6-luna`; pass `--model` to run any of them
with another OpenAI or Anthropic model (native-search systems stay on their own provider).

### Suites

| Suite | Tasks | Primary metric | Source |
|-------|------:|----------------|--------|
| `browsecomp` | 1,266 | accuracy (`score`) | [BrowseComp](https://openai.com/index/browsecomp/) |
| `simpleqa` | 4,326 | `correct` (plus official F1) | [SimpleQA](https://openai.com/index/introducing-simpleqa/) |
| `frames` | 824 | accuracy (`score`) | [google/frames-benchmark](https://huggingface.co/datasets/google/frames-benchmark) |
| `dsqa` | 900 | `f1` | [google/deepsearchqa](https://huggingface.co/datasets/google/deepsearchqa) |
| `widesearch` | 200 | `f1_by_row` | [ByteDance-Seed/WideSearch](https://huggingface.co/datasets/ByteDance-Seed/WideSearch) |
| `hle` | text-only subset | accuracy (`score`) | [cais/hle](https://huggingface.co/datasets/cais/hle) (gated) |

Data is downloaded at pinned revisions on first use and verified; nothing is redistributed
here. Answers are graded by an LLM judge (`openai/gpt-5.6-luna` by default, `--judge-model`
to change it) with each benchmark's grading prompt.

### Quick start

```bash
git clone https://github.com/exa-labs/benchmarks.git
cd benchmarks
uv sync --all-packages

export OPENAI_API_KEY=...        # Scout model and the judge
export EXA_API_KEY=...           # plus a key for each search API you run:
# BRAVE_SEARCH_API_KEY, PARALLEL_API_KEY, PERPLEXITY_API_KEY, ANTHROPIC_API_KEY

uv run bench list
uv run bench run --system scout-exa-auto --suite browsecomp --limit 5
uv run bench run --system scout-brave --suite browsecomp --limit 5
uv run bench run --system scout-exa-auto --model anthropic/claude-sonnet-5 --suite dsqa --limit 20
```

These commands make paid API calls. `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` point the model
clients at any compatible gateway. HLE needs `HF_TOKEN` after accepting its terms on Hugging
Face. `uv run bench download` fetches and verifies every suite up front.

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
summary.json              metrics, failed-as-zero primary metric, cost per task, stop reasons
```

The hash covers the system settings, suite revision and judge model. Re-running the same
command resumes: graded tasks are skipped, answered-but-ungraded tasks are only re-graded, and
failed tasks are retried. `--run-suffix rep2` starts an independent repeat.

Cost per task is split into model tokens (priced from [`harness/harness/llm/pricing.py`](harness/harness/llm/pricing.py))
and search calls (the provider's reported cost when it returns one, otherwise its list price);
judge cost is reported separately. A model with no listed price is reported as unknown, not
zero.

### Adding a search API

Implement `Searcher.run` in [`shared/shared/searchers/`](shared/shared/searchers/) (return
results plus the request's cost), register the provider in `build_searcher`
([`harness/harness/systems.py`](harness/harness/systems.py)), and add a `[searchers.*]` entry
and a `scout-*` system to `systems.toml`. Scout systems inherit `[defaults.scout]`, so the new
system is comparable with the rest without further changes.

### Third-party datasets

Benchmarks are downloaded from their upstream homes at pinned revisions and remain under their
own terms:

- BrowseComp and SimpleQA are published by OpenAI through
  [simple-evals](https://github.com/openai/simple-evals) (MIT). BrowseComp is encrypted
  upstream to keep it out of training data; do not republish decrypted questions or answers.
- FRAMES and DeepSearchQA are Apache-2.0.
- WideSearch data is CC0-1.0; the evaluator it adapts is MIT.
- Humanity's Last Exam is MIT and gated: accept its terms on Hugging Face and set `HF_TOKEN`.
  Its maintainers ask that the questions not be used for training.

## Benchmarks

| Benchmark | Queries | Tracks | Description |
|-----------|---------|--------|-------------|
| [WebCode](webcode-benchmark/) | ~840 | Contents, Highlights, RAG, E2E | Code docs extraction, query-aware highlights, long-context QA |
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

**Contents** — extraction fidelity against golden markdown (250 URLs)

| Searcher | Completeness | Accuracy | Structure | Signal | Code Recall | Table Recall | ROUGE-L |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Exa | **82.8** | **89.3** | **81.8** | **94.5** | **96.7** | 91.9 | **83.2** |
| Parallel | 74.2 | 89.2 | 80.8 | 77.6 | 94.1 | **92.2** | 73.7 |
| Claude | 59.8 | 81.1 | 75.1 | 55.1 | 82.4 | 82.0 | 66.8 |

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

```bash
git clone https://github.com/exa-labs/benchmarks.git
cd benchmarks
```

### WebCode Benchmark

```bash
cd webcode-benchmark
uv sync

export EXA_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

python -m evals.contents --searchers exa tavily parallel --limit 20
python -m evals.highlights --searchers exa tavily parallel --limit 20
python -m evals.rag --searchers exa brave perplexity --limit 20
python -m evals.e2e --info
```

### People Benchmark

```bash
cd simple-people-benchmark
uv sync

export EXA_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

pbench --limit 50
```

### Company Benchmark

```bash
cd simple-company-benchmark
uv sync

export EXA_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

cbench --limit 50
cbench --track retrieval
cbench --track rag
```

### Publication Retrieval Benchmark

```bash
cd publication-benchmark
uv sync

export EXA_API_KEY="your-key"

pubbench --limit 50
pubbench --track paper
pubbench --track tot
pubbench --searchers exa brave parallel --output results.json
```

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

The `search` method is used by retrieval and RAG evals. The `extract` method is used by the contents and highlights evals for URL-based extraction. The evaluation harness calls `run`, which returns the same results plus the request's cost and latency; the default `run` wraps `search` and reports the cost as unknown, so override it for a provider with a known price (see [`exa.py`](shared/shared/searchers/exa.py)).

## Requirements

- Python 3.11+
- OpenAI API key (for LLM grading), or an Anthropic key for Anthropic-model systems
- Search API credentials

## License

MIT

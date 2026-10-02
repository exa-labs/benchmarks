# Exa Search Benchmarks

Open datasets and a shared evaluation harness for search APIs, with 12 runnable
suites across the Exa datasets and four public benchmarks below.

**Scout evaluates search tools on agentic tasks.** It swaps Exa, Perplexity,
Parallel or Brave into a standard research agent while keeping the model,
prompts, research loop and budgets fixed across the search API presets. The agent
searches repeatedly, follows up on results and synthesizes an answer from the
evidence it retrieves. Each adapter preserves its provider's search request format.

The runner also supports single-step RAG, direct retrieval and URL extraction + RAG.
OpenAI and Anthropic native-search presets use hosted web search within Scout,
evaluating each provider's model and search together. End-to-end agentic search
products like Exa Agent are out of scope. All modes share grading, resumable runs,
cost reporting and 95% bootstrap confidence intervals.

## Benchmarks

| Dataset | Queries | Tracks | Description |
|---------|--------:|--------|-------------|
| [WebCode](data/webcode/) · [blog](https://exa.ai/blog/webcode) | 840 | Highlights, RAG | Code documentation retrieval and grounded QA |
| [People Search](data/people.jsonl) · [blog](https://exa.ai/blog/people-search-benchmark) | 1,400 | Retrieval | Find profiles by role, location and seniority |
| [Company Search](data/company.jsonl) · [blog](https://exa.ai/blog/company-search-benchmarks) | 839 | Retrieval, RAG | Find companies and extract facts |
| [Publication Retrieval](data/publication.jsonl) · [blog](https://exa.ai/blog/publications-search) | 1,866 | Publication, ToT | Find papers from questions or tip-of-the-tongue recollections |
| [SWEChat Searches](data/swechatsearches/) | 586 | Retrieval | Exa-derived search benchmark using queries from SWE-chat, graded by result-content rubrics |

**SWEChat Searches** uses coding-agent search queries from
[SALT-NLP/SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat), with
rubrics derived from the coding agent traces. It measures
how many criteria the top 1, 5, and 10 results cover using returned snippets;
it does not run the original coding tasks. See [setup and scoring](data/swechatsearches/README.md).

WebCode has 557 runnable QA tasks. Its 250 [Contents](data/webcode/contents.jsonl)
records and 33 [E2E](data/webcode/e2e.jsonl) tasks are dataset-only exports:
Contents lacks the licensed golden markdown; E2E lacks an executor and setup files.

The results below are historical measurements with earlier adapters and model
settings. Rerun them for current comparisons; these aggregate tables have no CIs.

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

## Quick Start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/exa-labs/benchmarks.git
cd benchmarks
uv sync --locked

export EXA_API_KEY=...
export OPENAI_API_KEY=...  # answering model and judge

uv run bench list
uv run bench run --suite browsecomp --system scout-exa-auto-highlights --limit 5
```

Run the Exa datasets from the same directory:

```bash
uv run pbench --searchers exa --limit 50
uv run cbench --limit 50       # both company tracks; --track retrieval or rag
uv run pubbench --limit 50    # both publication tracks; --track paper or tot
uv run bench run --suite webcode-rag --system rag-exa-webcode --limit 20
uv run bench run --suite webcode-highlights --system extract-rag-exa-extract --limit 20
uv run bench run --suite swechatsearches --system search-exa-auto-highlights --judge-model openai/gpt-6-luna --limit 20
```

These commands make paid calls. `--dry-run` validates data, compatible systems and
credentials first. Publication grading is deterministic and needs no judge key.

## Public Benchmarks

The same runner also supports four upstream datasets, downloaded and cached at
pinned revisions by [`data/loaders.py`](data/loaders.py):

| Suite | Tasks | Source |
|-------|------:|--------|
| `browsecomp` | 1,266 | [OpenAI BrowseComp](https://openai.com/index/browsecomp/) |
| `frames` | 824 | [Google FRAMES](https://huggingface.co/datasets/google/frames-benchmark) |
| `dsqa` | 900 | [Google DeepSearchQA](https://huggingface.co/datasets/google/deepsearchqa) |
| `widesearch` | 200 | [ByteDance WideSearch](https://huggingface.co/datasets/ByteDance-Seed/WideSearch) |

Use Scout or single-step RAG for these suites. `uv run bench download` fetches all
data up front. To preflight all 12 runnable suites with compatible Exa systems:

```bash
uv run bench run --suite all \
  --system rag-exa-auto-highlights search-exa-auto-highlights extract-rag-exa-extract \
  --limit 1 --dry-run
```

Remove `--dry-run` to execute; remove `--limit` for the full datasets.

## Systems and Models

**RAG** searches once, then answers.
**Search** grades ranked results directly. **Extract + RAG** answers from a supplied URL.

The ten primary API presets below each have `scout-`, `rag-` and `search-` variants
(for example, `scout-perplexity-fast`). All provider settings live in
[`systems.toml`](systems.toml); `uv run bench list` shows the full catalog.

| API | Presets |
|-----|---------|
| Exa highlights | `exa-instant-highlights`, `exa-fast-highlights`, `exa-auto-highlights` |
| Perplexity Search | `perplexity-web`, `perplexity-fast` |
| Parallel Search | `parallel-turbo`, `parallel-fast`, `parallel-basic`, `parallel-advanced` |
| Brave LLM Context | `brave-llm-context` |

URL extraction supports Exa, Parallel and Claude. Set the corresponding `PERPLEXITY_API_KEY`,
`PARALLEL_API_KEY`, `BRAVE_SEARCH_API_KEY` or `ANTHROPIC_API_KEY` when using them.

| Role | Default model |
|------|---------------|
| Scout | `openai/gpt-6-astra` |
| Judge | `openai/gpt-6-luna` |
| RAG / Extract + RAG answering | `openai/gpt-5.6-luna` |
| `openai-native-search` | `openai/gpt-6-astra` |
| `anthropic-native-search` and Claude extraction | `claude-opus-5-5` |

`SCOUT_DEFAULT` and `JUDGE_DEFAULT` live in [`harness/llm/__init__.py`](harness/llm/__init__.py).
`--model` overrides the answering model; `--judge-model` overrides the judge.
Use provider-prefixed model names, such as `anthropic/claude-opus-5-5`.
Hosted search stays on its own provider. Claude's extraction model is configured
separately from the answering model in `systems.toml`.

## Results

Runs save resumable task artifacts under gitignored `results/runs/`. Export
shareable summaries, including costs and 95% bootstrap confidence intervals:

```bash
uv run bench run --suite publication --system search-exa-publication --output results/publication.json
uv run bench summary results/runs/<run-directory> --output results/publication.json
```

Intervals use 10,000 task resamples (seed 0); the primary metric also reports
failures as zero. They measure task-sampling uncertainty, not variation between
repeated model runs. See [`harness/statistics.py`](harness/statistics.py).

## Repository Layout

```text
data/          Exa datasets and public/local loaders
benchmarks/    suite definitions, prompts and graders
harness/       Scout, RAG, runner, model clients and API adapters
tests/         offline tests
results/       exported summaries
```

Add providers through [`harness/searchers/`](harness/searchers/) and
[`systems.toml`](systems.toml). Run the tests with `uv run pytest`.

## License

MIT. Upstream datasets retain their own terms: BrowseComp is MIT, FRAMES and
DeepSearchQA are Apache-2.0, and WideSearch data is CC0-1.0. BrowseComp is decrypted
only in memory; do not republish its decrypted questions or answers.

The SWEChat Searches data in [`data/swechatsearches/`](data/swechatsearches/) is not MIT: it contains
information from [SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat), which
is made available under the
[ODC Attribution License](https://opendatacommons.org/licenses/by/1-0/), and is
redistributed under the same license. See the [dataset attribution](data/swechatsearches/README.md).

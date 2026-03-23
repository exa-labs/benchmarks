# Exa Search Benchmarks

Open benchmarks for evaluating search APIs.

## Benchmarks

| Benchmark | Queries | Tracks | Description |
|-----------|---------|--------|-------------|
| [WebCode](webcode-benchmark/) | ~840 | Contents, Highlights, RAG, E2E | Code docs extraction, query-aware highlights, long-context QA |
| [People Search](simple-people-benchmark/) | 1,400 | Retrieval | Find people profiles by role, location, seniority |
| [Company Search](simple-company-benchmark/) | ~800 | Retrieval + RAG | Find companies by name, industry, geography, funding |

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

## Quick Start

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

The `search` method is used by retrieval and RAG evals. The `extract` method is used by the contents and highlights evals for URL-based extraction.

## Requirements

- Python 3.11+
- OpenAI API key (for LLM grading)
- Search API credentials

## License

MIT

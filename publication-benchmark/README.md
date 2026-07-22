# Publication Retrieval Benchmark

An open benchmark for evaluating academic **paper retrieval**. Given a query, can a search API surface the exact paper it refers to? Every query is grounded in a single gold publication, and results are scored on **paper identity** — not content format — so heterogeneous searchers (academic APIs, general web search, agents) are compared apples-to-apples.

## Overview

### Two Tracks

**Paper Track** — specific-question retrieval. Natural research questions grounded in a paper's findings, methods, or numbers, with title/author/year leakage guards.

> *"What optical pigment absorbance ratio was used as an indicator of phytoplankton nutritional status in these lakes, and which lake showed the higher annual mean value?"*

**Tip-of-the-Tongue (ToT) Track** — known-item retrieval from a vague, degraded human recollection. No gold title words, so there is no keyword to lean on.

> *"Trying to find that paper on teaching legal ethics that used the real drug-dealing conveyancing case … it laid out three levels of how ethics and technical skill get integrated — automatic, problematic, unintegrated with a Guantanamo torture example. Also leaned on Vygotsky's zone of proximal development…"*

### Metrics

| Metric | Description |
|--------|-------------|
| **R@1** | % of queries where the first result is the gold paper |
| **R@5** | % of queries with the gold paper in the top 5 |
| **R@10** | % of queries with the gold paper in the top 10 |
| **MRR** | Mean reciprocal rank of the gold paper |

Because there is exactly one gold paper per query, recall@k collapses to hit@k.

### Dataset

**1,866 queries** across two tracks:

| Track | Queries | Description |
|-------|---------|-------------|
| Paper | 1,472 | Specific-question retrieval grounded in a paper's findings, methods, or numbers |
| ToT | 394 | Tip-of-the-tongue known-item retrieval from vague, degraded recollections |

Each row of `data/publication/publication_search.jsonl`:

```json
{
  "query_id": "paper_0001",
  "text": "In a German study of 145 cosmetic samples, what percentage contained NDMA...",
  "track": "paper",
  "bucket": "quantitative",
  "gold_paper": {"doi": "10.1590/...", "title": "..."},
  "metadata": {"year": 2009, "gold_answer": "...", "supporting_quote": "...", "original_text": "..."},
  "tags": ["paper-retrieval", "quantitative"]
}
```

`gold_paper` is the single paper the query was generated from. `bucket` is the
question type (`quantitative`, `method`, `finding`, `mechanism`) for the paper
track, or the recollection style (`statement`, `question`, `fragment`) for ToT.

`text` is phrased as a neutral search query for programmatic/agent consumption:
first-person and conversational framing is stripped while the substantive
detail — and the deliberate vagueness of ToT recollections — is preserved.
`metadata.original_text` keeps the source phrasing for reference.

## Installation

```bash
cd publication-benchmark
uv sync
```

## Quick Start

```bash
export EXA_API_KEY="your-exa-key"

# Full benchmark (default searcher: exa)
pubbench

# Limit queries / results
pubbench --limit 50 --num-results 10

# One track only
pubbench --track paper
pubbench --track tot

# Compare searchers, save results
pubbench --searchers exa brave parallel perplexity --output results.json
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--limit N` | Limit number of queries |
| `--num-results N` | Results per query (default: 10) |
| `--track {paper,tot}` | Run only a specific track |
| `--output FILE` | Save results to JSON file |
| `--searchers NAME...` | Searchers to use (default: exa) |

## Evaluation Methodology

Each result is reduced to candidate paper identifiers and matched to the gold
paper in **priority order**:

1. **DOI** — explicit field, or extracted from the result URL/text
2. **Fuzzy title** — word-level Jaccard similarity ≥ 0.7

The rank of the first hit is recorded, from which R@1/R@5/R@10 and MRR are
computed. Grading is fully deterministic (no LLM), so runs are reproducible and
free.

## Implementing Custom Searchers

```python
from src.searchers import Searcher, SearchResult

class MySearcher(Searcher):
    name = "my-searcher"

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        return [SearchResult(url="https://...", title="Paper Title", text="...")]
```

To help identity matching, searchers may also populate
`SearchResult.metadata["doi"]` when known.

## License

MIT

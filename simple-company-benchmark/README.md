# Company Search Benchmark

An open benchmark for evaluating company search. Test how well your search API finds companies by name, industry, geography, funding, and more.

## Overview

### Two Evaluation Tracks

**Retrieval Track** - Return ranked lists of companies matching criteria

| Metric | Description |
|--------|-------------|
| **R@1** | % of queries where the first result is correct |
| **R@10** | % of queries with a correct result in top 10 |
| **Precision** | % of returned results that are relevant |

**RAG Track** - Extract specific facts about companies

| Metric | Description |
|--------|-------------|
| **Accuracy** | % of queries with correct extracted answer |

### Dataset

**~800 queries** across two tracks:

| Track | Split | Queries | Description |
|-------|-------|---------|-------------|
| Retrieval | Static | 345 | Named lookup, industry/geo, founding year, disambiguation |
| Retrieval | Dynamic | 260 | Employee count, funding stage/amount, composite, semantic |
| RAG | Static | 171 | Founding year, location, industry, YC batch, founders |
| RAG | Dynamic | 63 | Employee count, funding amounts, acquisitions |

### Query Types

**Retrieval Track:**
- Named lookup: Find specific company by name
- Industry + geography: Filter by industry and location
- Founded year: Filter by founding year
- Disambiguation: Distinguish similar company names
- Employee count: Filter by workforce size
- Funding stage/amount: Filter by funding criteria
- Composite: Multiple constraints combined
- Semantic: Natural language descriptions

**RAG Track:**
- Founding year extraction
- Location/HQ extraction
- Industry classification
- YC batch identification
- Founder names
- Employee count
- Funding amounts
- Acquisition history

## Installation

```bash
cd simple-company-benchmark
uv sync
```

## Quick Start

```bash
# Set API keys
export EXA_API_KEY="your-exa-key"
export OPENAI_API_KEY="your-openai-key"

# Run full benchmark
cbench

# Run with limit
cbench --limit 50

# Run specific track
cbench --track retrieval
cbench --track rag

# Run specific split
cbench --split static
cbench --split dynamic

# Save results
cbench --output results.json
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--limit N` | Limit number of queries |
| `--num-results N` | Results per query (default: 10) |
| `--track {retrieval,rag}` | Run only specific track |
| `--split {static,dynamic}` | Run only specific split |
| `--enrich-exa-contents` | Fetch full page contents via Exa API |
| `--output FILE` | Save results to JSON file |
| `--searchers NAME...` | Searchers to use (default: exa) |

## Evaluation Methodology

**Retrieval Track:**
- Queries with `gold_company_homepage`: Check if URL appears in top-k results
- Queries with `constraints`: LLM validates returned companies against structured filters

**RAG Track:**
- Compare extracted answers against ground truth
- Dynamic facts (employees, funding): 20% tolerance window
- Text answers: LLM-based semantic equivalence

## Implementing Custom Searchers

Create a new searcher by extending the base class:

```python
from src.searchers import Searcher, SearchResult

class MySearcher(Searcher):
    name = "my-searcher"

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        # Your implementation
        return [
            SearchResult(
                url="https://example.com",
                title="Company Name",
                text="Company description...",
            )
        ]
```

## License

MIT

# Company Search Benchmark

An open benchmark for evaluating company search. Test how well your search API finds companies by name, industry, geography, funding, and more.

## Overview

This benchmark has two evaluation tracks:

**Retrieval Track** tests ranked list quality across multiple query types: named lookup, industry/geography filtering, founding year, employee count, funding stage, and composite queries.

**RAG Track** tests fact extraction accuracy for company attributes: founding year, location, industry, YC batch, founders, funding amounts, and more.

### Metrics

| Track | Metric | Description |
|-------|--------|-------------|
| Retrieval | R@1 | % of queries where the first result is correct |
| Retrieval | R@10 | % of queries with a correct result in top 10 |
| Retrieval | Precision | % of returned results that are relevant |
| RAG | Accuracy | % of queries with correct extracted answer |

### Dataset

**~800 queries** across two tracks:

| Track | Split | Queries | Description |
|-------|-------|---------|-------------|
| Retrieval | Static | 345 | Named lookup, industry/geo, founding year, disambiguation |
| Retrieval | Dynamic | 260 | Employee count, funding stage/amount, composite, semantic |
| RAG | Static | 171 | Founding year, location, industry, YC batch, founders |
| RAG | Dynamic | 63 | Employee count, funding amounts, acquisitions |

The benchmark deliberately avoids well-known unicorns (Stripe, Notion, Figma) to ensure queries are "LLM-hard" and not answerable from pre-training alone. Companies include Series A/B startups, regional players (EU, APAC, LATAM), and niche B2B verticals.

## Installation

```bash
git clone https://github.com/exa-labs/benchmarks.git
cd benchmarks/simple-company-benchmark

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

```python
import asyncio
from src.benchmark import Benchmark, BenchmarkConfig
from src.searchers.exa import ExaSearcher

async def main():
    searcher = ExaSearcher(category="company")
    
    benchmark = Benchmark([searcher])
    results = await benchmark.run(BenchmarkConfig(limit=100))

asyncio.run(main())
```

Or use the CLI:

```bash
export EXA_API_KEY="your-api-key"
export OPENAI_API_KEY="your-api-key"  # For LLM grading

cbench --limit 50
```

## Query Types

### Retrieval Track

**Named lookup** finds a specific company:
```json
{
  "query": "FERNRIDE autonomous logistics Munich",
  "gold_company_homepage": "https://www.fernride.com"
}
```

**Attribute filtering** by industry and geography:
```json
{
  "query": "fintech companies in Switzerland",
  "constraints": {
    "hq_country": {"eq": "Switzerland"},
    "categories": {"contains": "fintech"}
  }
}
```

**Dynamic attributes** like employee count:
```json
{
  "query": "cybersecurity companies with over 200 employees",
  "constraints": {
    "employee_count": {"gte": 200},
    "industry": {"contains": "Security"}
  }
}
```

### RAG Track

**Static facts** (founding year, location, industry):
```json
{
  "query": "When was phospho in YC?",
  "expected_answer": "W24"
}
```

**Dynamic facts** (employees, funding) use 20% tolerance:
```json
{
  "query": "How many employees does Wakeo have?",
  "expected_answer": "86"
}
```

## Grading

**Retrieval track:**
- Queries with `gold_company_homepage` check if URL appears in top-k
- Queries with `constraints` validate returned companies against structured filters (rule-based + LLM grading)

**RAG track:**
- Compare extracted answers against ground truth
- Dynamic facts (employees, funding): 20% tolerance
- Text answers: LLM-as-judge for semantic equivalence

## CLI Options

```bash
cbench --help

Options:
  --limit N              Limit number of queries
  --num-results N        Results per query (default: 10)
  --output FILE          Save results to JSON file
  --enrich-exa-contents  Fetch page contents via Exa API
  --track TYPE           Run only 'retrieval' or 'rag' track
  --split TYPE           Run only 'static' or 'dynamic' split
  --searchers NAME...    Searchers to use (default: exa)
```

## Implementing Your Searcher

```python
from src.searchers import Searcher, SearchResult

class MySearcher(Searcher):
    name = "my-search"
    
    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        response = await my_api.search(query, limit=num_results)
        
        return [
            SearchResult(
                url=r.url,
                title=r.title,
                text=r.snippet,
                metadata={"score": r.score},
            )
            for r in response.results
        ]
```

## Requirements

- Python 3.11+
- OpenAI API key (for LLM grading)
- Search API credentials

## License

MIT

---

*Built by the Exa team. Questions or feedback? Reach out at hello@exa.ai.*

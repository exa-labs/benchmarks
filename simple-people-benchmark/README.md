# People Search Benchmark

An open benchmark for evaluating people search. Test how well your search API finds LinkedIn profiles by role, location, and seniority.

## Overview

| Metric | Description |
|--------|-------------|
| **R@1** | % of queries where the first result is correct |
| **R@10** | % of queries with a correct result in top 10 |
| **Precision** | % of returned results that are relevant |

### Dataset

**1,400 synthetic role-based queries** across job functions:

| Category | Queries | Examples |
|----------|---------|----------|
| `engineering` | 365 | Software, DevOps, Security |
| `marketing` | 180 | Marketing, Brand, Growth |
| `sales` | 160 | Sales, BD, Account Management |
| `people_hr` | 100 | HR, Recruiting, People Ops |
| `design` | 100 | Product Design, UX, Creative |
| `product` | 90 | Product Management |
| `finance` | 85 | Finance, Accounting, FP&A |
| `legal` | 70 | Legal, Compliance, IP |
| `data_analytics` | 70 | Data Science, Analytics |
| `trust_safety` | 80 | Trust & Safety, Policy |

## Results

| Searcher | R@1 | R@10 | Precision | Queries |
|----------|-----|------|-----------|---------|
| exa | 72.0% | 94.5% | 63.3% | 1399 |
| brave | 44.4% | 77.9% | 30.2% | 1373 |
| parallel | 20.8% | 74.7% | 26.9% | 1387 |

## Installation

```bash
git clone https://github.com/exa-labs/people-benchmark.git
cd people-benchmark

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
    searcher = ExaSearcher(api_key="your-api-key", category="people")
    
    benchmark = Benchmark([searcher])
    results = await benchmark.run(BenchmarkConfig(limit=100))

asyncio.run(main())
```

Or use the CLI:

```bash
export EXA_API_KEY="your-api-key"
export OPENAI_API_KEY="your-api-key"  # For LLM grading

pbench --limit 50
```

## Built-in Searchers

```python
from src.searchers import ExaSearcher, BraveSearcher, ParallelSearcher

# Exa with people category
exa = ExaSearcher(category="people", include_text=True)

# Brave filtered to LinkedIn profiles
brave = BraveSearcher(site_filter="linkedin.com/in")

# Parallel (internal) filtered to LinkedIn
parallel = ParallelSearcher(source_policy={"include_domains": ["linkedin.com"]})
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
                text=r.snippet,  # Profile content for grading
                metadata={"score": r.score},
            )
            for r in response.results
        ]
```

## Grading

Binary LLM evaluation for each result:

- **Score 1**: Profile matches ALL criteria (job title, location, seniority)
- **Score 0**: Profile fails ANY criterion

Strict matching — "close enough" = 0, partial matches = 0.

### Role Equivalence

Accepted variations within same function:
- "Security Engineer" ≈ "Application Security Engineer" ✓
- "Head of X" ≈ "Director of X" ≈ "VP of X" ✓
- "ML Engineer" ≈ "Machine Learning Engineer" ✓

Not accepted (different functions):
- "Data Analyst" ≠ "Data Engineer"
- "Project Manager" ≠ "Product Manager"
- "UX Designer" ≠ "Head of UX" (IC vs leadership)

## Data Format

```json
{
  "query_id": "people_role_0001",
  "text": "senior payroll specialist in boston",
  "bucket": "finance",
  "metadata": {
    "role_title": "Senior Payroll Specialist",
    "role_function": "finance",
    "role_seniority": "ic",
    "geo_name": "Boston",
    "geo_type": "city"
  }
}
```

## CLI Options

```bash
pbench --help

Options:
  --limit N              Limit number of queries
  --num-results N        Results per query (default: 10)
  --output FILE          Save results to JSON file
  --enrich-exa-contents  Fetch page contents via Exa API
  --searchers NAME...    Searchers to use (default: exa brave parallel)
```

## Requirements

- Python 3.11+
- OpenAI API key (for LLM grading)
- Search API credentials

## License

MIT

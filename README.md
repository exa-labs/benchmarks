# Exa Search Benchmarks

Open benchmarks for evaluating search APIs. Test how well your search finds people, companies, and more.

## Benchmarks

| Benchmark | Queries | Tracks | Description |
|-----------|---------|--------|-------------|
| [People Search](simple-people-benchmark/) | 1,400 | Retrieval | Find LinkedIn profiles by role, location, seniority |
| [Company Search](simple-company-benchmark/) | ~800 | Retrieval + RAG | Find companies by name, industry, geography, funding |

## People Search Results

| Searcher | R@1 | R@10 | Precision | Queries |
|----------|-----|------|-----------|---------|
| exa | 72.0% | 94.5% | 63.3% | 1399 |
| brave | 44.4% | 77.9% | 30.2% | 1373 |
| parallel | 20.8% | 74.7% | 26.9% | 1387 |

## Company Search Results

**Retrieval Track**

| Searcher | R@1 | R@5 | R@10 | Precision |
|----------|-----|-----|------|-----------|
| exa | 61.8% | 90.6% | 94.2% | 65.9% |
| brave | 35.9% | 61.8% | 72.9% | 39.2% |
| parallel | 36.6% | 66.3% | 78.6% | 40.4% |

**RAG Track**

| Searcher | Accuracy |
|----------|----------|
| exa | 79% |
| brave | 65% |
| parallel | 66% |

### Evaluation Tracks

Two tracks designed to separate retrieval from fact extraction:

**Retrieval Track** — Return ranked lists of companies matching criteria

| Type | Example |
|------|---------|
| Named lookup | "Acme Robotics company" (with disambiguation) |
| Attribute filtering | Industry, geography, founding year, employee count |
| Funding queries | Stage, amount raised, recent rounds |
| Composite | Multiple constraints: "Israeli security companies founded after 2015" |
| Semantic | Natural language descriptions of company characteristics |

**RAG Track** — Extract specific facts from retrieved content

| Query Type | Example | Expected |
|------------|---------|----------|
| Founding year | "When was [Company] founded?" | "2019" |
| Employee count | "How many people work at [Company]?" | "86" |
| Last funding | "When did [Company] raise their last round?" | "November 2024" |
| YC batch | "What YC batch was [Company] in?" | "S24" |
| Founders | "Who founded [Company]?" | "Alice Chen, Bob Park" |

Static facts get exact-match scoring. Dynamic facts (employees, funding) get ±20% tolerance.

## Quick Start

```bash
git clone https://github.com/exa-labs/benchmarks.git
cd benchmarks
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

# Run full benchmark
cbench --limit 50

# Run specific track
cbench --track retrieval
cbench --track rag

# Run specific split (static vs dynamic facts)
cbench --split static
cbench --split dynamic
```

## Implementing Your Own Searcher

Both benchmarks use the same `Searcher` interface:

```python
from shared.searchers import Searcher, SearchResult

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

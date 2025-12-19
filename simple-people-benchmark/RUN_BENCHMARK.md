# Running the People Search Benchmark

This guide will help you recreate the benchmark results shown in the README.

## Prerequisites

1. **API Keys Required:**
   - `EXA_API_KEY` - Get one at https://exa.ai
   - `BRAVE_SEARCH_API_KEY` - Get one at https://brave.com/search/api
   - `PARALLEL_API_KEY` - Get one at https://parallel.ai
   - `OPENAI_API_KEY` - Get one at https://platform.openai.com (required for LLM grading)

2. **Installation:**
   ```bash
   cd simple-people-benchmark
   uv sync
   ```

## Running the Benchmark

### Option 1: Using the Interactive Script

```bash
cd simple-people-benchmark
source .venv/bin/activate  # or: uv run python run_benchmark.py
python run_benchmark.py
```

This script will:
- Check which API keys are available
- Let you configure the benchmark interactively
- Run the benchmark with available searchers

### Option 2: Using the CLI (pbench)

Set your API keys:
```bash
export EXA_API_KEY="your-exa-key"
export BRAVE_SEARCH_API_KEY="your-brave-key"
export PARALLEL_API_KEY="your-parallel-key"
export OPENAI_API_KEY="your-openai-key"
```

Run the benchmark:
```bash
# Run with all searchers (default)
uv run pbench

# Run with limited queries (for testing)
uv run pbench --limit 50

# Run specific searchers only
uv run pbench --searchers exa brave

# Save results to file
uv run pbench --output results.json

# Run with Exa content enrichment
uv run pbench --enrich-exa-contents
```

### Option 3: Using Python Directly

```python
import asyncio
from src.benchmark import Benchmark, BenchmarkConfig
from src.searchers.exa import ExaSearcher
from src.searchers.brave import BraveSearcher
from src.searchers.parallel import ParallelSearcher

async def main():
    # Initialize searchers
    searchers = [
        ExaSearcher(category="people"),
        BraveSearcher(site_filter="linkedin.com/in"),
        ParallelSearcher(source_policy={"include_domains": ["linkedin.com"]}),
    ]
    
    # Run benchmark
    benchmark = Benchmark(searchers)
    config = BenchmarkConfig(
        limit=None,  # Use all queries (~1400)
        num_results=10,
        output_file="results.json",
    )
    results = await benchmark.run(config)
    
    # Clean up
    for searcher in searchers:
        await searcher.close()

asyncio.run(main())
```

## Expected Results

Based on the README, you should see results similar to:

| Searcher | R@1 | R@10 | Precision | Queries |
|----------|-----|------|-----------|---------|
| exa | 72.0% | 94.5% | 63.3% | 1399 |
| brave | 44.4% | 77.9% | 30.2% | 1373 |
| parallel | 20.8% | 74.7% | 26.9% | 1387 |

**Note:** Results may vary slightly due to:
- API changes over time
- Different query processing
- Network conditions
- LLM grading variations

## Troubleshooting

### "No queries found!"
- Make sure `data/people/simple_people_search.jsonl` exists
- The file should have ~1400 lines

### "No searchers available!"
- Check that you've set the required API keys
- Verify the keys are valid

### "People grading failed"
- Ensure `OPENAI_API_KEY` is set correctly
- Check your OpenAI API quota

### Searcher-specific errors
- **Exa**: Verify your API key at https://exa.ai
- **Brave**: Check your subscription status
- **Parallel**: Ensure you're using the correct API endpoint

## Running a Quick Test

To test with a small subset first:

```bash
uv run pbench --searchers brave --limit 10
```

This will run only 10 queries with the Brave searcher, which is much faster for testing.


# Benchmark Setup Complete ✅

The People Search Benchmark is now ready to run. Here's what has been set up:

## What's Ready

1. ✅ **Package Installed**: All dependencies are installed via `uv sync`
2. ✅ **Data Available**: 1,400 queries in `data/people/simple_people_search.jsonl`
3. ✅ **Scripts Created**: Helper scripts for running the benchmark

## Next Steps

### 1. Set API Keys

You'll need API keys for the search services and OpenAI (for grading):

```bash
export EXA_API_KEY="your-exa-key"              # https://exa.ai
export BRAVE_SEARCH_API_KEY="your-brave-key"   # https://brave.com/search/api
export PARALLEL_API_KEY="your-parallel-key"     # https://parallel.ai
export OPENAI_API_KEY="your-openai-key"         # https://platform.openai.com
```

**Note**: You can run the benchmark with just one searcher if you only have some API keys.

### 2. Run the Benchmark

#### Quick Test (Recommended First)
```bash
cd simple-people-benchmark
uv run pbench --limit 10 --searchers exa
```

This runs 10 queries with just the Exa searcher to verify everything works.

#### Full Benchmark
```bash
# Run all searchers with all queries
uv run pbench

# Or save results to a file
uv run pbench --output results.json
```

#### Interactive Script
```bash
python run_benchmark.py
```

This will guide you through the configuration interactively.

### 3. Verify Setup

Run the quick test script to check everything:
```bash
./quick_test.sh
```

## Expected Results

When you run the full benchmark, you should see results similar to:

| Searcher | R@1 | R@10 | Precision | Queries |
|----------|-----|------|-----------|---------|
| exa | 72.0% | 94.5% | 63.3% | 1399 |
| brave | 44.4% | 77.9% | 30.2% | 1373 |
| parallel | 20.8% | 74.7% | 26.9% | 1387 |

**Note**: Results may vary slightly due to API changes, network conditions, or LLM grading variations.

## Files Created

- `run_benchmark.py` - Interactive Python script to run the benchmark
- `RUN_BENCHMARK.md` - Detailed guide on running the benchmark
- `quick_test.sh` - Quick setup verification script
- `README.md` - Copied from parent directory (required for package build)

## Troubleshooting

See `RUN_BENCHMARK.md` for detailed troubleshooting steps.

## Time Estimate

- **Quick test (10 queries)**: ~1-2 minutes
- **Full benchmark (~1400 queries)**: ~2-4 hours (depending on API rate limits)

The benchmark runs queries in parallel with concurrency limits, so it's optimized for speed while respecting API limits.


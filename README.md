# Exa Search Benchmarks

Open benchmarks for evaluating search APIs. Test how well your search finds people, companies, and more.

## Benchmarks

| Benchmark | Queries | Description |
|-----------|---------|-------------|
| [People Search](simple-people-benchmark/) | 1,400 | Find LinkedIn profiles by role, location, seniority |
| [Company Search](simple-company-benchmark/) | ~800 | Find companies by name, industry, geography, funding |

## Quick Start

```bash
# People benchmark
cd simple-people-benchmark
uv sync
export EXA_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
pbench --limit 50

# Company benchmark
cd simple-company-benchmark
uv sync
export EXA_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
cbench --limit 50
```

## License

MIT

# Simple People Search Benchmark

Open benchmark for evaluating people search.

## Usage

```bash
uv sync --all-packages --all-groups --locked  # from the repository root
export EXA_API_KEY=...
export OPENAI_API_KEY=...
uv run pbench --searchers exa --limit 10
```


## Shared harness

This command translates into the common `bench` runner; it owns no separate
execution, grading, aggregation or persistence loop. For example:

```bash
uv run bench run --suite people --system search-exa-people --limit 10 --dry-run
```

`--dry-run` checks data and credentials before paid calls. `--output` saves a JSON
list of aggregate summaries; detailed results and grades are stored under
`runs/<run>/tasks/<id>/`. Resume, model overrides and cost accounting follow the
[common harness](../README.md). Empty retrievals count as zero in the query denominator.

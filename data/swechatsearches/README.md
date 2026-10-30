# SWEChat Searches

Exa's search benchmark derived from [SALT-NLP/SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat):
**586 queries and 887 criteria**, with rubrics derived from the coding agent traces.
Each row in `searches.jsonl` contains `id`, `query`, and `criteria` (`id`, `description`).

## Run

From the repository root, run `uv sync --locked` and set `EXA_API_KEY` and
`OPENAI_API_KEY` in your shell.

```bash
uv run bench run --suite swechatsearches --system search-exa-auto-highlights \
  --judge-model openai/gpt-6-luna --num-results 10 \
  --output results/swechatsearches.json
```

Add `--limit 20` for a smaller run. Other searchers and their credentials are
listed in [Systems and Models](../../README.md#systems-and-models).

## Scoring

The judge checks each criterion against each of the top 10 results using its
URL, title, available date, and returned content capped at 16,384 characters.
`covered_at_1`, `covered_at_5`, and `covered_at_10` measure the fraction of criteria
satisfied by at least one top-k result, averaged across queries. The headline
`covered_at_10_failed_as_zero` counts failed queries as zero. This measures
retrieved evidence coverage, without answer generation or extra page fetching.

## License

Contains information from [SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat),
made available under the [ODC Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/).
This derivative is distributed under the same license; the harness is MIT-licensed.

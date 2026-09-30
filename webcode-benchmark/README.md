# WebCode Benchmark

Search evals for coding agents. [Blog post](https://exa.ai/blog/web-code).

## Runnable suites

| Suite | Tasks | What it measures | System kind |
|-------|------:|------------------|-------------|
| `webcode-highlights` | 250 | Given a URL and query, extract relevant evidence and synthesize | `extract-rag` |
| `webcode-rag` | 307 | Full-web retrieval and synthesis over code documentation | `rag` |

Both suites use the [common harness](../README.md), one RAG implementation, and the
same correctness and groundedness graders. Exa, Tavily and Parallel support both
tracks; Claude supports extraction; Brave LLM Context and Perplexity support RAG.

## Quick start

From the repository root:

```bash
uv sync --all-packages --all-groups --locked
export EXA_API_KEY=...
export OPENAI_API_KEY=...
uv run bench run --suite webcode-highlights --system extract-rag-exa-extract --limit 20
uv run bench run --suite webcode-rag --system rag-exa-webcode --limit 20
# Compatibility entry points, backed by the same runner:
uv run python -m evals.highlights --searchers exa tavily parallel --limit 20
uv run python -m evals.rag --searchers exa brave perplexity --limit 20
uv run python -m evals.e2e --info
```

Add `--dry-run` for data and credential preflight without paid calls. Run artifacts,
resume behavior, model overrides and cost reporting follow the common CLI.
`--output results.json` saves a list of aggregate summaries; per-query results and
grades live in `runs/<run>/tasks/<id>/`.

## Datasets

JSONL files remain in `data/`:

| Dataset | Rows | Schema |
|---------|-----:|--------|
| **highlights** | 250 | `{id, query, expected_answer, citation_url, citation_excerpt}` |
| **rag** | 307 | `{id, query, expected_answer, source_url, citation_excerpt}` |
| **e2e** | 33 | `{id, slug, repo, repo_url, release_tag, task_description, test_patch, metadata}` |

E2E is a **dataset-only export**. Its inspection CLI reports task metadata; the repo
has no coding-agent executor and does not bundle the referenced setup files. It is
therefore outside `bench list`'s runnable suite catalog. The Contents track has been
removed because its licensed golden markdown is unavailable. Some URLs were omitted
from Highlights for licensing reasons.

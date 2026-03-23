# WebCode Benchmark

Search evals for coding agents. [Blog post](https://exa.ai/blog/web-code).

## Evals

| Eval | What it measures | Runner |
|------|-----------------|--------|
| **Contents** | Extraction fidelity against golden markdown (250 URLs) | `python -m evals.contents` |
| **Highlights** | In-document retrieval — given a URL + query, surface the relevant section (250 queries) | `python -m evals.highlights` |
| **RAG** | Full-web retrieval + synthesis on long-context code docs (307 queries) | `python -m evals.rag` |
| **E2E** | Sandboxed coding tasks requiring web search (33 tasks, dataset only) | `python -m evals.e2e --info` |

## Quick Start

```bash
cd webcode-benchmark
uv sync

export EXA_API_KEY="..."
export OPENAI_API_KEY="..."

python -m evals.contents --searchers exa tavily parallel --limit 20
python -m evals.highlights --searchers exa tavily parallel --limit 20
python -m evals.rag --searchers exa brave perplexity --limit 20
python -m evals.e2e --info
```

## Searchers

| Provider | Contents | Highlights | RAG |
|----------|----------|------------|-----|
| Exa | ✓ | ✓ | ✓ |
| Tavily | ✓ | ✓ | ✓ |
| Parallel | ✓ | ✓ | ✓ |
| Claude | ✓ | ✓ | — |
| Brave | — | — | ✓ |
| Perplexity | — | — | ✓ |

## Datasets

All datasets are JSONL files in `data/`:

| Dataset | Rows | Schema |
|---------|------|--------|
| **contents** | 250 | `{id, url, title, tags}` |
| **highlights** | 250 | `{id, query, expected_answer, citation_url, citation_excerpt}` |
| **rag** | 307 | `{id, query, expected_answer, source_url, citation_excerpt}` |
| **e2e** | 33 | `{id, slug, repo, repo_url, release_tag, task_description, test_patch, metadata}` |

> **Note**: Some URLs have been excluded from the contents and highlights datasets due to licensing restrictions.

### Golden markdown (contents eval)

The contents dataset contains URLs only; the golden markdown is not included for licensing reasons. To run the contents eval, you need to generate `data/contents/golden_markdown.jsonl` yourself. Each row should have the shape `{id, expected_markdown}`.

We built the golden references using the following pipeline:

1. **Render** each URL in a cloud browser (e.g. [Browserbase](https://browserbase.com/)) with full JS execution, lazy loading, and dynamic rendering
2. **Capture** full-page screenshots and extract the HTML DOM
3. **Feed** screenshots + DOM into a multimodal language model to produce markdown faithful to the rendered page

See the [blog post](https://exa.ai/blog/web-code) for more details on this approach.

## Output

Pass `--output results.json` to save per-query scores. Results are structured as:

```json
{
  "exa": [{"id": "contents_001", "completeness": 0.9, "accuracy": 0.95, ...}],
  "tavily": [...]
}
```

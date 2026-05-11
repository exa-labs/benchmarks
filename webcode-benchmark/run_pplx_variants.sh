#!/usr/bin/env bash
# Run 4 Perplexity variant arms on the webcode RAG eval.
#
# Arms (all use /v1/agent + web_search tool):
#   pplx-nocf-noyt  — content-filtration-disabled key, youtube excluded
#   pplx-nocf       — content-filtration-disabled key, no domain filter
#   pplx-beta-noyt  — beta key, youtube excluded
#   pplx-beta       — beta key, no domain filter
#
# Requires: uv sync (run once from this directory first).
# Output: results/pplx_variants.json
set -euo pipefail
cd "$(dirname "$0")"

CONC="${CONC:-10}"
OUTDIR="results"
mkdir -p "$OUTDIR"

echo "==> START webcode RAG — Perplexity variants  $(date +%H:%M:%S)"

uv run python -m evals.rag \
  --searchers pplx-nocf-noyt pplx-nocf pplx-beta-noyt pplx-beta \
  --concurrency "$CONC" \
  --output "$OUTDIR/pplx_variants.json"

echo "==> DONE  $(date +%H:%M:%S)"
echo "Results: $OUTDIR/pplx_variants.json"

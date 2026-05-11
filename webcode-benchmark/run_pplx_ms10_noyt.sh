#!/usr/bin/env bash
# Run pplx-ms10-noyt: max_steps=10, max_tokens=10000, max_tokens_per_page=4000, youtube excluded.
#
# Output: results/pplx_ms10_noyt.json
set -euo pipefail
cd "$(dirname "$0")"

CONC="${CONC:-10}"
OUTDIR="results"
mkdir -p "$OUTDIR"

echo "==> START webcode RAG — pplx-ms10-noyt  $(date +%H:%M:%S)"

uv run python -m evals.rag \
  --searchers pplx-ms10-noyt \
  --concurrency "$CONC" \
  --output "$OUTDIR/pplx_ms10_noyt.json"

echo "==> DONE  $(date +%H:%M:%S)"
echo "Results: $OUTDIR/pplx_ms10_noyt.json"

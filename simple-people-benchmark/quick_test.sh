#!/bin/bash
# Quick test script to verify the benchmark setup

set -e

echo "People Search Benchmark - Quick Test"
echo "===================================="
echo ""

cd "$(dirname "$0")"

# Check if data exists
if [ ! -f "data/people/simple_people_search.jsonl" ]; then
    echo "❌ Error: data/people/simple_people_search.jsonl not found!"
    exit 1
fi

QUERY_COUNT=$(wc -l < data/people/simple_people_search.jsonl)
echo "✓ Found $QUERY_COUNT queries in dataset"
echo ""

# Check API keys
echo "Checking API keys..."
MISSING_KEYS=()

if [ -z "$EXA_API_KEY" ]; then
    MISSING_KEYS+=("EXA_API_KEY")
fi

if [ -z "$BRAVE_SEARCH_API_KEY" ] && [ -z "$BRAVE_API_KEY" ]; then
    MISSING_KEYS+=("BRAVE_SEARCH_API_KEY")
fi

if [ -z "$PARALLEL_API_KEY" ] && [ -z "$PARALLELS_API_KEY" ]; then
    MISSING_KEYS+=("PARALLEL_API_KEY")
fi

if [ -z "$OPENAI_API_KEY" ]; then
    MISSING_KEYS+=("OPENAI_API_KEY")
fi

if [ ${#MISSING_KEYS[@]} -eq 0 ]; then
    echo "✓ All API keys are set"
else
    echo "⚠ Missing API keys: ${MISSING_KEYS[*]}"
    echo "   Set them with: export KEY_NAME='your-key'"
fi
echo ""

# Test import
echo "Testing package installation..."
if uv run python -c "from src.benchmark import Benchmark; print('✓ Package installed correctly')" 2>/dev/null; then
    echo ""
else
    echo "❌ Error: Package not installed. Run: uv sync"
    exit 1
fi

echo "Setup looks good! You can now run:"
echo "  uv run pbench --limit 10  # Test with 10 queries"
echo "  uv run pbench              # Run full benchmark"
echo "  python run_benchmark.py   # Interactive script"


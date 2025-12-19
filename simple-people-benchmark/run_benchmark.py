#!/usr/bin/env python3
"""
Script to run the People Search Benchmark and recreate the results.

This script checks for required API keys and runs the benchmark with all three searchers.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.benchmark import Benchmark, BenchmarkConfig
from src.searchers.exa import ExaSearcher
from src.searchers.brave import BraveSearcher
from src.searchers.parallel import ParallelSearcher


def check_api_keys():
    """Check which API keys are available."""
    keys = {
        "EXA_API_KEY": os.getenv("EXA_API_KEY"),
        "BRAVE_SEARCH_API_KEY": os.getenv("BRAVE_SEARCH_API_KEY") or os.getenv("BRAVE_API_KEY"),
        "PARALLEL_API_KEY": os.getenv("PARALLEL_API_KEY") or os.getenv("PARALLELS_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    }
    
    print("API Key Status:")
    print("=" * 50)
    for key_name, value in keys.items():
        status = "✓ Set" if value else "✗ Missing"
        print(f"  {key_name:25} {status}")
    print()
    
    return keys


def build_searchers(api_keys):
    """Build searchers based on available API keys."""
    searchers = []
    
    if api_keys["EXA_API_KEY"]:
        try:
            searchers.append(ExaSearcher(category="people"))
            print("✓ ExaSearcher initialized")
        except Exception as e:
            print(f"✗ ExaSearcher failed: {e}")
    else:
        print("✗ ExaSearcher skipped (no EXA_API_KEY)")
    
    if api_keys["BRAVE_SEARCH_API_KEY"]:
        try:
            searchers.append(BraveSearcher(site_filter="linkedin.com/in"))
            print("✓ BraveSearcher initialized")
        except Exception as e:
            print(f"✗ BraveSearcher failed: {e}")
    else:
        print("✗ BraveSearcher skipped (no BRAVE_SEARCH_API_KEY)")
    
    if api_keys["PARALLEL_API_KEY"]:
        try:
            searchers.append(ParallelSearcher(source_policy={"include_domains": ["linkedin.com"]}))
            print("✓ ParallelSearcher initialized")
        except Exception as e:
            print(f"✗ ParallelSearcher failed: {e}")
    else:
        print("✗ ParallelSearcher skipped (no PARALLEL_API_KEY)")
    
    print()
    return searchers


async def main():
    print("People Search Benchmark - Results Recreation")
    print("=" * 60)
    print()
    
    # Check API keys
    api_keys = check_api_keys()
    
    if not api_keys["OPENAI_API_KEY"]:
        print("⚠ WARNING: OPENAI_API_KEY is required for grading!")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Build searchers
    print("Initializing searchers...")
    searchers = build_searchers(api_keys)
    
    if not searchers:
        print("❌ No searchers available! Please set at least one API key.")
        print("\nRequired API keys:")
        print("  - EXA_API_KEY: https://exa.ai")
        print("  - BRAVE_SEARCH_API_KEY: https://brave.com/search/api")
        print("  - PARALLEL_API_KEY: https://parallel.ai")
        print("  - OPENAI_API_KEY: https://platform.openai.com")
        return
    
    # Ask for configuration
    print("Benchmark Configuration:")
    print("=" * 60)
    
    limit_input = input("Limit queries (press Enter for all ~1400, or enter a number): ").strip()
    limit = int(limit_input) if limit_input else None
    
    num_results_input = input("Results per query (default 10, press Enter): ").strip()
    num_results = int(num_results_input) if num_results_input else 10
    
    output_file = input("Output file (optional, press Enter to skip): ").strip() or None
    
    enrich = input("Enrich Exa contents? (y/n, default n): ").strip().lower() == 'y'
    
    print()
    print("Starting benchmark...")
    print("=" * 60)
    print()
    
    # Run benchmark
    config = BenchmarkConfig(
        limit=limit,
        num_results=num_results,
        output_file=output_file,
        enrich_exa_contents=enrich,
    )
    
    benchmark = Benchmark(searchers)
    results = await benchmark.run(config)
    
    print()
    print("=" * 60)
    print("Benchmark completed!")
    
    if output_file:
        print(f"Results saved to: {output_file}")
    
    # Close searchers
    for searcher in searchers:
        if hasattr(searcher, 'close'):
            await searcher.close()


if __name__ == "__main__":
    asyncio.run(main())


"""
benchmark_demo.py
==================
Runs the full 44-question routing benchmark (no LLM calls — routing only).
Prints a summary and optionally saves an Excel workbook.

Usage:
    # Routing analysis only (no API key needed):
    python examples/benchmark_demo.py

    # With live SQL generation on first 5 questions (needs API key + DKL Excel):
    export ANTHROPIC_API_KEY=sk-...
    python examples/benchmark_demo.py --dkl Olist_DataLens_Output.xlsx --live_n 5
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yoro.yoro_hybrid_inference import (
    compute_complexity_score,
    route_question,
    InferencePath,
)
from yoro.yoro_pipeline import TEST_QUESTIONS


def routing_only_benchmark(questions):
    """Run routing analysis with token counting — no LLM calls."""
    # Approximate token counts per path
    TOKEN_MAP = {
        InferencePath.YORO_PURE:     50,
        InferencePath.YORO_HYBRID:   700,
        InferencePath.GRAPHRAG_ONLY: 3900,
    }
    BASELINE = 3900  # Graph-RAG baseline

    records = []
    for q in questions:
        decision = route_question(q, yoro_available=True)
        complexity, signals = compute_complexity_score(q)
        tokens = TOKEN_MAP[decision.path]
        records.append({
            "question": q[:80],
            "path": decision.path.value,
            "complexity": round(complexity, 2),
            "confidence": round(decision.confidence, 2),
            "tokens": tokens,
            "reduction_pct": round(100 * (1 - tokens / BASELINE), 1),
            "signals": " | ".join(signals[:3]),
        })
    return records


def print_summary(records):
    from collections import Counter
    paths = [r["path"] for r in records]
    dist = Counter(paths)
    avg_tokens = sum(r["tokens"] for r in records) / len(records)
    avg_reduction = sum(r["reduction_pct"] for r in records) / len(records)

    bar = "=" * 68
    print(bar)
    print("  YORO HYBRID BENCHMARK — Routing & Token Summary")
    print(bar)
    print()
    print(f"  Questions analysed: {len(records)}")
    print()
    print(f"  {'Path':<22} {'Count':>6} {'Share':>7} {'Avg Tokens':>12}")
    print(f"  {'-'*22} {'-'*6} {'-'*7} {'-'*12}")
    token_map = {"YORO_PURE": 50, "YORO_HYBRID": 700, "GRAPHRAG_ONLY": 3900}
    for path, count in sorted(dist.items()):
        share = 100 * count / len(records)
        tok = token_map.get(path, 0)
        print(f"  {path:<22} {count:>6} {share:>6.1f}%  {tok:>12,}")

    print()
    print(f"  Baseline (Graph-RAG full)  : ~3,900 tokens")
    print(f"  YORO Hybrid (this routing) : ~{avg_tokens:,.0f} tokens")
    print(f"  Average reduction          : {avg_reduction:.1f}%")
    print()
    print(bar)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dkl", default=None,
                        help="DKL Excel path (optional, needed for live SQL gen)")
    parser.add_argument("--live_n", type=int, default=0,
                        help="Number of questions for live SQL generation (0 = routing only)")
    args = parser.parse_args()

    print()
    records = routing_only_benchmark(TEST_QUESTIONS)
    print_summary(records)

    if args.live_n > 0 and args.dkl:
        print()
        print("  Running live SQL generation...")
        # Import pipeline components for live run
        from yoro.yoro_schema_profiler import OlistSchemaProfiler
        from yoro.yoro_hybrid_inference import YOROExpertClient, YOROHybridPipeline

        profiler = OlistSchemaProfiler(args.dkl)
        expert = YOROExpertClient(backend="anthropic", profiler=profiler)
        pipeline = YOROHybridPipeline(
            expert_client=expert,
            profiler=profiler,
            yoro_available=True,
        )

        for q in TEST_QUESTIONS[:args.live_n]:
            result = pipeline.generate(q, verbose=True)

        stats = pipeline.stats_summary()
        print()
        print(f"  Live stats: {stats}")
    elif args.live_n > 0 and not args.dkl:
        print()
        print("  (Pass --dkl Olist_DataLens_Output.xlsx to enable live SQL generation)")


if __name__ == "__main__":
    main()

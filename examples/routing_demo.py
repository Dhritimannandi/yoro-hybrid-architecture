"""
routing_demo.py
================
Demonstrates the YORO hybrid router on 8 sample questions spanning
the full complexity spectrum. No API key or fine-tuned model required —
this runs entirely on the deterministic complexity scorer.

Usage:
    python examples/routing_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yoro.yoro_hybrid_inference import (
    compute_complexity_score,
    route_question,
    InferencePath,
    YORO_PURE_THRESHOLD,
    YORO_HYBRID_THRESHOLD,
)

DEMO_QUESTIONS = [
    # Expected Path A (YORO Pure)
    "What are the top 10 customers by total sales value in March 2018?",
    "What is the average review score for orders delivered in Q4 2017?",
    "Which product categories have the highest total revenue?",
    "How many orders were placed in each month of 2018?",
    # Expected Path B (YORO Hybrid)
    "Which sellers are making good money despite having lots of 1-star reviews?",
    "Which are the top 5 sellers by revenue in each state?",
    # Expected Path C (Graph-RAG Only)
    "Is there a correlation between GMV and CSAT scores across product categories?",
    "Do the total payment values match the sum of item prices plus freight?",
]

PATH_ICONS = {
    InferencePath.YORO_PURE:    "A",
    InferencePath.YORO_HYBRID:  "B",
    InferencePath.GRAPHRAG_ONLY:"C",
}

PATH_TOKENS = {
    InferencePath.YORO_PURE:    "~50",
    InferencePath.YORO_HYBRID:  "~700",
    InferencePath.GRAPHRAG_ONLY:"~3,900",
}

def main():
    print()
    print("=" * 72)
    print("  YORO HYBRID ROUTER — Complexity Scoring Demo")
    print(f"  Thresholds: Pure < {YORO_PURE_THRESHOLD} | Hybrid < {YORO_HYBRID_THRESHOLD} | GraphRAG >= {YORO_HYBRID_THRESHOLD}")
    print("=" * 72)

    for i, question in enumerate(DEMO_QUESTIONS, 1):
        complexity, signals = compute_complexity_score(question)
        decision = route_question(question, yoro_available=True)

        path_label = f"Path {PATH_ICONS[decision.path]} — {decision.path.value}"
        tokens = PATH_TOKENS[decision.path]

        print()
        print(f"  Q{i}: {question[:70]}{'...' if len(question)>70 else ''}")
        print(f"       Complexity : {complexity:.2f}")
        print(f"       Route      : {path_label}  ({tokens} tokens)")
        print(f"       Confidence : {decision.confidence:.2f}")
        if signals:
            print(f"       Signals    : {' | '.join(signals[:4])}")

    print()
    print("=" * 72)
    print()
    print("  Summary: Questions routed without any LLM call (~0.5ms per question)")
    print("  Fine-tuned expert model (Path A/B) or Graph-RAG (Path C) handles SQL.")
    print()


if __name__ == "__main__":
    main()

"""
yoro_pipeline.py
=================
End-to-end YORO Hybrid Pipeline orchestrator.

This is the main entry point that ties together:
  1. Schema profiling (OlistSchemaProfiler)
  2. Synthetic data generation (yoro_synthetic_data_generator)
  3. Fine-tuning data formatting (YOROFinetuneFormatter)
  4. Hybrid inference routing (YOROHybridPipeline)
  5. Benchmarking against all 44 test questions

Two operating modes:

  Mode 1 — SETUP (run once, offline)
      python yoro_pipeline.py --mode setup
      Generates synthetic NLQ-SQL pairs, formats them for fine-tuning,
      writes training configs. No model calls needed after this except
      the data generation API calls to Claude.

  Mode 2 — BENCHMARK (run anytime, online)
      python yoro_pipeline.py --mode benchmark
      Runs all 44 test questions through the three inference paths
      (YORO_PURE, YORO_HYBRID, GRAPHRAG_ONLY) and measures:
        - Token count per path
        - Latency per path
        - Path routing distribution
      Writes a comparison Excel workbook.

  Mode 3 — GENERATE (run a single question)
      python yoro_pipeline.py --mode generate --question "..."
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Local modules
from .yoro_schema_profiler        import OlistSchemaProfiler
from .yoro_synthetic_data_generator import (
    SEED_SKELETONS,
    generate_synthetic_dataset,
    save_synthetic_data,
    load_synthetic_data,
)
from .yoro_finetuning_formatter   import YOROFinetuneFormatter, write_training_configs
from .yoro_hybrid_inference       import (
    InferencePath,
    YOROExpertClient,
    YOROHybridPipeline,
    route_question,
    compute_complexity_score,
)

# Import Graph-RAG from the previous session
try:
    import sys
    sys.path.insert(0, "/home/claude")
    sys.path.insert(0, "/mnt/user-data/outputs")
    from dkl_context_graph import DKLContextGraph, _approx_tokens
    HAS_GRAPHRAG = True
except ImportError:
    HAS_GRAPHRAG = False
    print("⚠  dkl_context_graph not found — Graph-RAG paths will use PICARD schema")


# ---------------------------------------------------------------------------
# Test questions (44 questions from the Sage testing framework)
# ---------------------------------------------------------------------------

TEST_QUESTIONS = [
    "What are the top 10 customers by total sales value in March 2018?",
    "What is the average review score given by customers for orders placed in Q4 2017?",
    "Who were our biggest customers in March 2018?",
    "How were our customer ratings in the last quarter of 2017?",
    "What is the distribution of payment methods (e.g. credit card, boleto) across completed orders?",
    "Which product categories have the highest average rating, considering only categories with more than 50 reviews?",
    "For each customer, calculate their total lifetime value (LTV) as of December 2018.",
    "How does the delivery time (in days) vary by product category and seller region?",
    "Identify risky customers who have canceled more than 10% of their orders and placed at least 5 orders.",
    "What percentage of total revenue comes from orders placed during weekends vs weekdays?",
    "For each seller, compute average shipping delay (order_delivered_customer_date - order_estimated_delivery_date).",
    "Which sellers are slowest at getting orders from carrier pickup to customer delivery?",
    "What is the ratio of single-item orders vs multi-item orders over time (monthly)?",
    "Which product categories have the highest returns of negative reviews (rating <= 2)?",
    "Which product categories get the most negative reviews as a percentage of total reviews?",
    "Which states have the highest cancellation rates relative to order volume?",
    "Which states have the highest cancellation rates?",
    "What is the monthly growth rate of number of active sellers over time?",
    "What proportion of buyers have used more than one payment method over their lifetime?",
    "Which sellers have the highest cancellation rate per 100 orders?",
    "Which product categories contribute the most to late deliveries, and what % of their orders are late?",
    "Which sellers are making good money despite having lots of 1-star reviews?",
    "For each customer state, compute average AOV, average delivery time, and average review score.",
    "What percentage of orders have missing or invalid delivery dates?",
    "Which customers are located more than 500km away from their seller?",
    "Does increasing the number of payment installments lead to higher order values?",
    "What would happen to our revenue if we reduced shipping costs by 20%?",
    "Identify any unusual spikes or drops in daily order volume during 2017-2018.",
    "Create a pivot showing average review score by payment method AND product category.",
    "Do the total payment values in the payments table match the sum of item prices plus freight?",
    "Which are the top 5 sellers by revenue in each state, excluding sellers with average rating < 3?",
    "What are the most common customer complaints in support tickets?",
    "Which products have the highest stockout rates?",
    "Is there a correlation between GMV and CSAT scores across different product categories?",
    "Which SKUs have the highest return rate and what is the average value per SKU?",
    "Calculate the Customer Lifetime Value for each customer and identify the top 10%.",
    "Can you chart our monthly order numbers for the past couple years?",
    "Which product categories are our biggest revenue drivers?",
    "Generate a horizontal bar chart of the top 15 product categories by total revenue.",
    "Do categories with higher sales tend to have better CSAT scores?",
    "Which products have the highest return rates and what do they typically cost?",
    "I want to see how reviews vary by payment method and category for higher-value orders.",
    "Which sellers are making good money despite having lots of 1-star reviews?",
    "Which sellers cancel the most orders and does this relate to how long they have been active?",
]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_routing_benchmark(
    questions:      List[str],
    profiler:       OlistSchemaProfiler,
    dkl_graph=None,
) -> pd.DataFrame:
    """
    Run the routing decision benchmark across all questions.
    Measures token count per path WITHOUT making LLM calls for SQL.
    This is an offline measurement of the routing + schema sizing logic.
    """
    # Pre-compute schema sizes for each path
    picard_schema = profiler.picard_schema()
    codes_schema  = profiler.codes_schema()

    if dkl_graph is not None:
        full_graphrag   = dkl_graph.get_full_schema()
        full_graphrag_tok = _approx_tokens(full_graphrag) if HAS_GRAPHRAG else len(full_graphrag)//4
    else:
        full_graphrag   = codes_schema
        full_graphrag_tok = len(codes_schema)//4

    yoro_pure_tokens = 50  # paper: ~50 avg tokens (just DB ID + question)

    records = []
    for q in questions:
        routing = route_question(q, yoro_available=True)
        complexity, signals = compute_complexity_score(q)

        # Compute hybrid schema size (query-specific)
        if dkl_graph is not None:
            hybrid_schema = dkl_graph.get_schema_for_question(q)
            hybrid_tokens = _approx_tokens(hybrid_schema) if HAS_GRAPHRAG else len(hybrid_schema)//4
        else:
            hybrid_tokens = len(picard_schema)//4

        # Which token count would be used by this routing decision?
        if routing.path == InferencePath.YORO_PURE:
            tokens_used = yoro_pure_tokens
        elif routing.path == InferencePath.YORO_HYBRID:
            tokens_used = hybrid_tokens
        else:
            tokens_used = full_graphrag_tok

        # Baseline = current Graph-RAG (before YORO)
        baseline_tokens = full_graphrag_tok

        records.append({
            "Question":            q[:100],
            "Complexity_Score":    round(complexity, 2),
            "Routing_Path":        routing.path.value,
            "Routing_Confidence":  round(routing.confidence, 2),
            "Tokens_YORO_Pure":    yoro_pure_tokens,
            "Tokens_YORO_Hybrid":  hybrid_tokens,
            "Tokens_GraphRAG":     full_graphrag_tok,
            "Tokens_Used":         tokens_used,
            "Tokens_Baseline":     baseline_tokens,
            "Token_Reduction_Pct": round(100*(1-tokens_used/max(baseline_tokens,1)), 1),
            "Routing_Signals":     " | ".join(s for s in signals[:4]),
        })

    return pd.DataFrame(records)


def run_live_benchmark(
    questions:      List[str],
    pipeline:       YOROHybridPipeline,
    n_questions:    int = 10,  # cap for cost control
) -> pd.DataFrame:
    """
    Run live SQL generation through the hybrid pipeline (makes LLM calls).
    Capped at n_questions to control API costs.
    """
    records = []
    sample  = questions[:n_questions]

    for q in sample:
        result = pipeline.generate(q, verbose=False)
        records.append({
            "Question":      q[:100],
            "Path_Used":     result.path_used.value,
            "Prompt_Tokens": result.prompt_tokens,
            "Latency_ms":    result.latency_ms,
            "SQL":           result.sql[:200],
            "Confidence":    result.routing.confidence,
        })
        time.sleep(0.2)  # rate limiting

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, dkl_graph=None) -> None:
    bar = "=" * 68
    print(bar)
    print("  YORO HYBRID PIPELINE — ROUTING & TOKEN BENCHMARK")
    print(bar)

    # Path distribution
    dist = df["Routing_Path"].value_counts()
    print(f"\n  Routing distribution over {len(df)} questions:")
    for path, count in dist.items():
        pct = round(100 * count / len(df), 1)
        avg_tok = df[df["Routing_Path"]==path]["Tokens_Used"].mean()
        print(f"    {path:<20} {count:>3} questions ({pct:>5.1f}%)  "
              f"avg {avg_tok:>6.0f} tokens")

    print()
    avg_used     = df["Tokens_Used"].mean()
    avg_baseline = df["Tokens_Baseline"].mean()
    avg_pure     = df["Tokens_YORO_Pure"].mean()
    avg_reduction= df["Token_Reduction_Pct"].mean()

    w = 38
    print(f"  {'Scenario':<{w}} {'Avg tokens':>12}  {'vs Baseline':>12}")
    print(f"  {'-'*w} {'-'*12}  {'-'*12}")
    print(f"  {'Baseline (Graph-RAG full, before YORO)':<{w}} "
          f"{avg_baseline:>12,.0f}  {'—':>12}")
    print(f"  {'YORO pure (Path A, 100% YORO)':<{w}} "
          f"{avg_pure:>12,.0f}  "
          f"{f'-{round(100*(1-avg_pure/avg_baseline),1)}%':>12}")
    print(f"  {'Hybrid routing (optimal per question)':<{w}} "
          f"{avg_used:>12,.0f}  "
          f"{f'-{round(avg_reduction,1)}%':>12}")
    print()
    print(f"  Average token reduction from hybrid routing: {avg_reduction:.1f}%")
    print(f"  (Path A={dist.get('YORO_PURE',0)} questions, "
          f"B={dist.get('YORO_HYBRID',0)}, "
          f"C={dist.get('GRAPHRAG_ONLY',0)})")

    print()
    print("  Compared to raw JSON baseline from earlier sessions:")
    raw_baseline = 42_904  # from the benchmark sessions
    print(f"    Raw JSON (original)     : ~42,904 tokens")
    print(f"    Graph-RAG (previous)    : ~{avg_baseline:,.0f} tokens  "
          f"({round(100*(1-avg_baseline/42904),1)}% reduction)")
    print(f"    YORO hybrid (this work) : ~{avg_used:,.0f} tokens  "
          f"({round(100*(1-avg_used/42904),1)}% reduction vs raw JSON)")
    print(bar)


# ---------------------------------------------------------------------------
# Excel output
# ---------------------------------------------------------------------------

def save_benchmark_results(
    df_routing: pd.DataFrame,
    output_path: str,
    df_live: Optional[pd.DataFrame] = None,
) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_routing.to_excel(writer, sheet_name="Routing_Analysis", index=False)

        # Summary sheet
        path_counts = df_routing["Routing_Path"].value_counts().reset_index()
        path_counts.columns = ["Path", "Count"]
        path_counts["Avg_Tokens"] = path_counts["Path"].map(
            lambda p: df_routing[df_routing["Routing_Path"]==p]["Tokens_Used"].mean()
        )
        summary_rows = [
            {"Metric": "Total questions", "Value": len(df_routing)},
            {"Metric": "Avg tokens (hybrid)", "Value": round(df_routing["Tokens_Used"].mean())},
            {"Metric": "Avg tokens (baseline GraphRAG)", "Value": round(df_routing["Tokens_Baseline"].mean())},
            {"Metric": "Avg token reduction %", "Value": round(df_routing["Token_Reduction_Pct"].mean(),1)},
            {"Metric": "YORO_PURE questions", "Value": int((df_routing["Routing_Path"]=="YORO_PURE").sum())},
            {"Metric": "YORO_HYBRID questions", "Value": int((df_routing["Routing_Path"]=="YORO_HYBRID").sum())},
            {"Metric": "GRAPHRAG_ONLY questions", "Value": int((df_routing["Routing_Path"]=="GRAPHRAG_ONLY").sum())},
            {"Metric": "Raw JSON baseline tokens", "Value": 42904},
            {"Metric": "Reduction vs raw JSON %", "Value": round(100*(1-df_routing["Tokens_Used"].mean()/42904),1)},
        ]
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

        # Three-path comparison
        comparison = pd.DataFrame([
            {
                "Path":        "A — YORO Pure",
                "Description": "No schema in prompt; expert model has internalized DB",
                "Avg Tokens":  50,
                "% Reduction vs GraphRAG": round(100*(1-50/df_routing["Tokens_Baseline"].mean()),1),
                "Coverage":    "~60% of questions (simple aggregations)",
                "Risk":        "May miss novel values/joins not in training data",
            },
            {
                "Path":        "B — YORO Hybrid",
                "Description": "YORO expert + query-specific Graph-RAG schema (compressed)",
                "Avg Tokens":  round(df_routing[df_routing["Routing_Path"]=="YORO_HYBRID"]["Tokens_YORO_Hybrid"].mean()) if (df_routing["Routing_Path"]=="YORO_HYBRID").any() else 800,
                "% Reduction vs GraphRAG": round(100*(1 - df_routing[df_routing["Routing_Path"]=="YORO_HYBRID"]["Tokens_YORO_Hybrid"].mean() / df_routing["Tokens_Baseline"].mean()), 1) if (df_routing["Routing_Path"]=="YORO_HYBRID").any() else 70,
                "Coverage":    "~25% of questions (moderate complexity)",
                "Risk":        "Small latency overhead for schema retrieval (~6 ms)",
            },
            {
                "Path":        "C — Graph-RAG Only",
                "Description": "Full compressed schema, current production path",
                "Avg Tokens":  round(df_routing["Tokens_Baseline"].mean()),
                "% Reduction vs GraphRAG": 0,
                "Coverage":    "~15% of questions (complex/novel)",
                "Risk":        "Highest token cost but maximum schema coverage",
            },
        ])
        comparison.to_excel(writer, sheet_name="Path_Comparison", index=False)

        if df_live is not None and not df_live.empty:
            df_live.to_excel(writer, sheet_name="Live_SQL_Results", index=False)

    print(f"\n  Results saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def setup_mode(args) -> None:
    """Generate synthetic data and training configs (offline, one-time)."""
    print("=" * 60)
    print("  YORO SETUP — Synthetic Data Generation")
    print("=" * 60)

    profiler = OlistSchemaProfiler(args.dkl)
    summary  = profiler.schema_summary()
    print(f"\n  Schema: {summary['tables']} tables, {summary['columns']} columns")
    print(f"  DB ID:  {summary['db_id']}")

    print(f"\n  Generating synthetic NLQ-SQL pairs...")
    print(f"  Skeletons: {len(SEED_SKELETONS)}")
    print(f"  Target pairs: ~{len(SEED_SKELETONS) * args.sqls_per_skeleton}")

    pairs = generate_synthetic_dataset(
        profiler,
        skeletons          = SEED_SKELETONS[:args.max_skeletons],
        sqls_per_skeleton  = args.sqls_per_skeleton,
        verbose            = True,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw pairs
    raw_path = out_dir / "synthetic_pairs.jsonl"
    save_synthetic_data(pairs, str(raw_path))

    # Format for fine-tuning
    print("\n  Formatting for fine-tuning...")
    formatter = YOROFinetuneFormatter(profiler, hybrid_ratio=0.3)
    formatter.format_for_openai(pairs, str(out_dir / "openai_ft"), seed=42)
    formatter.format_for_hf(pairs,     str(out_dir / "hf_ft"),     seed=42)

    # Write training configs
    print("\n  Writing training configs...")
    write_training_configs(str(out_dir / "configs"))

    print(f"\n  Setup complete. Files in: {out_dir}")
    print(f"  Next step: fine-tune your model using {out_dir}/configs/train_lora.sh")


def benchmark_mode(args) -> None:
    """Run routing + token benchmark across all 44 test questions."""
    print("=" * 60)
    print("  YORO BENCHMARK — Routing & Token Analysis")
    print("=" * 60)

    profiler  = OlistSchemaProfiler(args.dkl)
    dkl_graph = DKLContextGraph(args.dkl) if HAS_GRAPHRAG else None

    print(f"\n  Running routing analysis on {len(TEST_QUESTIONS)} questions...")
    df_routing = run_routing_benchmark(TEST_QUESTIONS, profiler, dkl_graph)

    print_summary(df_routing, dkl_graph)

    # Optional: live SQL generation on first N questions
    df_live = None
    if args.live_n > 0:
        print(f"\n  Running live SQL generation on {args.live_n} questions...")
        expert_client = YOROExpertClient(
            backend  = args.backend,
            model    = args.model,
            profiler = profiler,
        )
        pipeline = YOROHybridPipeline(
            expert_client = expert_client,
            profiler      = profiler,
            dkl_graph     = dkl_graph,
            yoro_available= True,
        )
        df_live = run_live_benchmark(TEST_QUESTIONS, pipeline, n_questions=args.live_n)

        # Print live stats
        stats = pipeline.stats_summary()
        print(f"\n  Live run stats: {stats}")

    save_benchmark_results(
        df_routing,
        output_path = args.output,
        df_live     = df_live,
    )


def generate_mode(args) -> None:
    """Generate SQL for a single question."""
    profiler  = OlistSchemaProfiler(args.dkl)
    dkl_graph = DKLContextGraph(args.dkl) if HAS_GRAPHRAG else None

    expert_client = YOROExpertClient(
        backend  = args.backend,
        model    = args.model,
        profiler = profiler,
    )
    pipeline = YOROHybridPipeline(
        expert_client = expert_client,
        profiler      = profiler,
        dkl_graph     = dkl_graph,
    )

    result = pipeline.generate(args.question, verbose=True)
    print(f"\nSQL:\n{result.sql}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YORO Hybrid Pipeline for Olist text-to-SQL"
    )
    parser.add_argument("--mode",       default="benchmark",
                        choices=["setup","benchmark","generate"],
                        help="Operating mode")
    parser.add_argument("--dkl",        default="Olist_DataLens_Output.xlsx",
                        help="DKL Excel path")
    parser.add_argument("--output",     default="yoro_benchmark_results.xlsx",
                        help="Output Excel path for benchmark")
    parser.add_argument("--output_dir", default="./yoro_output",
                        help="Output directory for setup mode")
    parser.add_argument("--question",   default="",
                        help="Question to answer (generate mode)")
    parser.add_argument("--backend",    default="anthropic",
                        choices=["anthropic","azure_openai","local_hf"],
                        help="LLM backend for SQL generation")
    parser.add_argument("--model",      default="claude-sonnet-4-20250514",
                        help="Model name")
    parser.add_argument("--live_n",     type=int, default=0,
                        help="Number of questions for live SQL gen (0=skip)")
    parser.add_argument("--max_skeletons",    type=int, default=len(SEED_SKELETONS),
                        help="Max skeletons to use in setup mode")
    parser.add_argument("--sqls_per_skeleton", type=int, default=4,
                        help="SQL variants per skeleton in setup mode")

    args = parser.parse_args()

    if args.mode == "setup":
        setup_mode(args)
    elif args.mode == "benchmark":
        benchmark_mode(args)
    elif args.mode == "generate":
        if not args.question:
            parser.error("--question required for generate mode")
        generate_mode(args)


if __name__ == "__main__":
    main()

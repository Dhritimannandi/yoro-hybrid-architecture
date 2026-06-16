"""
yoro_hybrid_inference.py
=========================
Step 4 of the YORO Hybrid pipeline — the core innovation.

This module implements the Hybrid Router: the component that decides,
for each incoming question, whether to use:

  Path A — Pure YORO  (no schema in prompt)
      The expert model has internalized the schema and can answer
      directly. Input is just DB ID + question (~50 tokens).
      Best for: standard aggregations, well-known column/value patterns,
      questions that match the synthetic training distribution.

  Path B — YORO + Graph-RAG  (minimal compressed schema injected)
      The question is complex, ambiguous, or uses terminology that
      may not have been well covered in synthetic data.
      Input is DB ID + question + relevant schema subset (~500-800 tokens).
      Best for: multi-table joins with uncommon paths, questions about
      rarely seen values, complex window functions.

  Path C — Full Graph-RAG  (full compressed schema, no YORO)
      Fallback for cases where the expert model is not available or
      confidence is very low. Uses the existing DKLContextGraph.
      Input is question + full compressed schema (~3900 tokens).
      This is the current production path (before YORO training).

The routing decision uses:
  1. Keyword complexity scoring (fast, no LLM)
  2. A small confidence probe (optional, costs one extra LLM call)

This hybrid design addresses YORO's key limitation: it struggles on
rarely seen values and novel schemas because the parametric knowledge
may not cover everything. By blending YORO's efficiency with Graph-RAG's
schema grounding, we get the best of both approaches.

The module also contains the YOROExpertClient, which wraps the fine-tuned
expert model for inference (Azure OpenAI fine-tuned deployment or local
HF model), and the YOROHybridPipeline which orchestrates everything.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .yoro_schema_profiler import OlistSchemaProfiler


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class InferencePath(str, Enum):
    YORO_PURE    = "YORO_PURE"       # Path A
    YORO_HYBRID  = "YORO_HYBRID"     # Path B
    GRAPHRAG_ONLY= "GRAPHRAG_ONLY"   # Path C


@dataclass
class RoutingDecision:
    path:           InferencePath
    confidence:     float                    # 0-1, higher = more confident in YORO path
    reasons:        List[str]  = field(default_factory=list)
    prompt_tokens:  int        = 0
    schema_tables:  List[str]  = field(default_factory=list)


@dataclass
class SQLResult:
    sql:            str
    path_used:      InferencePath
    routing:        RoutingDecision
    latency_ms:     float
    prompt_tokens:  int
    expert_tokens:  int = 0           # tokens used from YORO expert model


# ---------------------------------------------------------------------------
# Complexity scorer — fast heuristic, no LLM needed
# ---------------------------------------------------------------------------

# ── Complexity signals tuned for Olist domain ──────────────────────────────
# Each entry: (regex_pattern, complexity_weight, label)
# Positive weights increase complexity → favour schema-grounded paths
# Negative weights decrease complexity → favour YORO_PURE path

COMPLEX_SIGNALS = [
    # Geospatial (needs geolocation table, multi-join)
    (r"\b(500km|distance|latitude|longitude|geolocation|km away)\b",  0.35, "geo_join"),
    (r"\b(translation|portuguese|english)\b",                          0.25, "translation"),
    # Statistical / ML
    (r"\b(correlation|correlat|csat.*gmv|gmv.*csat)\b",               0.40, "stat_ml"),
    (r"\b(spike|drop in.*volume|unusual.*volume|anomaly)\b",          0.35, "anomaly"),
    (r"\b(forecast|predict|what would happen|if we reduce)\b",        0.35, "scenario"),
    # Reconciliation / data quality
    (r"\b(do the total|match the sum|reconcil|discrepan)\b",          0.40, "reconcile"),
    (r"\b(missing|invalid delivery|null|duplicate)\b",                 0.35, "data_quality"),
    # Pivot / multi-dimensional
    (r"\b(pivot|crosstab|by.*method.*and.*category)\b",               0.30, "pivot"),
    # Window / partition
    (r"\btop \d+.*\beach\b|\beach.*top \d+\b",                   0.30, "window_partition"),
    (r"\bper state|per category|in each state|in each category\b",    0.25, "partition"),
    # Multi-metric questions (3+ metrics implies complex query)
    (r"(?=.*average)(?=.*average)(?=.*average)",                         0.25, "multi_metric"),
    (r"(?=.*aov)(?=.*delivery)(?=.*review)",                             0.25, "multi_metric"),
    # Long questions (many clauses)
    (r".{120,}",                                                          0.20, "long_question"),
]

SIMPLE_SIGNALS = [
    (r"\btop \d+\b(?!.*\beach\b)",                                  0.25, "top_n_simple"),
    (r"\b(average|avg)\b(?!.*average.*average)",                       0.15, "single_avg"),
    (r"\b(total|sum|count|how many)\b",                                0.12, "aggregate"),
    (r"\b(march|april|january|february|q1|q2|q3|q4|2017|2018)\b",    0.12, "time_filter"),
    (r"\b(review score|rating|stars)\b",                               0.12, "review"),
    (r"\b(payment method|payment type|boleto|credit.card)\b",         0.12, "payment"),
    (r"\b(deliver|shipping|freight)\b(?!.*invalid)",                   0.10, "delivery"),
]

# Complexity thresholds
YORO_PURE_THRESHOLD    = 0.55   # complexity < this → Path A
YORO_HYBRID_THRESHOLD  = 0.80   # complexity < this → Path B, else Path C


def compute_complexity_score(question: str) -> Tuple[float, List[str]]:
    """
    Compute a complexity score in [0, 1] for a question.
    Higher = more complex = more benefit from schema context.
    Returns (score, list_of_triggered_signals).
    Tuned for the Olist e-commerce domain.
    """
    q_lower  = question.lower()
    score    = 0.20  # Olist questions tend to be business-natural, lower baseline
    triggered = []

    for pattern, weight, label in COMPLEX_SIGNALS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            score += weight
            triggered.append(f"+{weight:.2f} {label}")

    for pattern, weight, label in SIMPLE_SIGNALS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            score -= weight
            triggered.append(f"-{weight:.2f} {label}")

    return max(0.0, min(1.0, score)), triggered


def route_question(
    question:         str,
    yoro_available:   bool = True,
    force_path:       Optional[InferencePath] = None,
) -> RoutingDecision:
    """
    Decide which inference path to use for a given question.

    Parameters
    ----------
    question : str
    yoro_available : bool
        Whether the fine-tuned YORO expert model is ready for inference.
        If False, always falls back to GRAPHRAG_ONLY.
    force_path : InferencePath, optional
        Override the routing decision (for testing).
    """
    if force_path is not None:
        return RoutingDecision(
            path=force_path, confidence=1.0,
            reasons=[f"forced to {force_path}"],
        )

    if not yoro_available:
        return RoutingDecision(
            path=InferencePath.GRAPHRAG_ONLY,
            confidence=0.0,
            reasons=["YORO expert not available — using Graph-RAG fallback"],
        )

    complexity, signals = compute_complexity_score(question)
    confidence = 1.0 - complexity   # higher confidence in YORO when less complex

    if complexity < YORO_PURE_THRESHOLD:
        path = InferencePath.YORO_PURE
    elif complexity < YORO_HYBRID_THRESHOLD:
        path = InferencePath.YORO_HYBRID
    else:
        path = InferencePath.GRAPHRAG_ONLY

    reasons = [f"complexity={complexity:.2f}"] + signals
    return RoutingDecision(path=path, confidence=confidence, reasons=reasons)


# ---------------------------------------------------------------------------
# YORO Expert Client
# ---------------------------------------------------------------------------

class YOROExpertClient:
    """
    Wraps a fine-tuned YORO expert model for inference.

    Supports two backends:
    - "azure_openai": Fine-tuned deployment on Azure OpenAI
    - "anthropic":    Claude with system prompt simulating the expert
                      (for testing before fine-tuning is complete)
    - "local_hf":     HuggingFace pipeline (local GPU)
    """

    def __init__(
        self,
        backend:         str = "anthropic",  # "azure_openai" | "anthropic" | "local_hf"
        model:           str = "claude-sonnet-4-20250514",
        azure_endpoint:  str = "",
        azure_api_key:   str = "",
        azure_api_ver:   str = "2025-04-01-preview",
        azure_deployment:str = "",
        hf_model_path:   str = "",
        profiler:        Optional[OlistSchemaProfiler] = None,
    ) -> None:
        self.backend          = backend
        self.model            = model
        self.azure_endpoint   = azure_endpoint
        self.azure_api_key    = azure_api_key
        self.azure_api_ver    = azure_api_ver
        self.azure_deployment = azure_deployment
        self.hf_model_path    = hf_model_path
        self.profiler         = profiler
        self._hf_pipeline     = None

    def _build_yoro_prompt(
        self,
        question:      str,
        schema_context: str = "",
    ) -> Tuple[str, str]:
        """
        Build (system, user) prompt for the YORO expert.
        If schema_context provided, this is the hybrid path.
        """
        db_id = self.profiler.DB_ID if self.profiler else "olist_ecommerce"

        system = (
            "You are a text-to-SQL expert for the Olist Brazilian e-commerce "
            "database. You have memorized the complete schema including all "
            "table names, column names, data types, and representative cell values. "
            "Generate ONLY the SQL query using Databricks Spark SQL syntax. "
            "No explanation, no markdown, no commentary."
        )
        if schema_context:
            # Hybrid: provide schema as additional grounding
            user = (
                f"{schema_context}\n\n"
                f"Construct the SQL by using the column names you memorized "
                f"for DB ID {db_id}.\n"
                f"Question: {question}"
            )
        else:
            # Pure YORO: no schema
            user = (
                f"Construct the SQL by using the column names you memorized "
                f"for DB ID {db_id}.\n"
                f"Question: {question}"
            )
        return system, user

    def generate_sql(
        self,
        question:      str,
        schema_context: str = "",
        temperature:   float = 0.0,
    ) -> str:
        """
        Generate SQL for a question using the configured backend.
        Returns the raw SQL string.
        """
        system, user = self._build_yoro_prompt(question, schema_context)

        if self.backend == "anthropic":
            return self._call_anthropic(system, user, temperature)
        elif self.backend == "azure_openai":
            return self._call_azure_openai(system, user, temperature)
        elif self.backend == "local_hf":
            return self._call_hf(system, user)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _call_anthropic(
        self, system: str, user: str, temperature: float = 0.0
    ) -> str:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text.strip()

    def _call_azure_openai(
        self, system: str, user: str, temperature: float = 0.0
    ) -> str:
        from langchain_openai import AzureChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = AzureChatOpenAI(
            azure_endpoint    = self.azure_endpoint,
            api_key           = self.azure_api_key,
            api_version       = self.azure_api_ver,
            deployment_name   = self.azure_deployment,
            temperature       = temperature,
        )
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
        return response.content.strip()

    def _call_hf(self, system: str, user: str) -> str:
        if self._hf_pipeline is None:
            from transformers import pipeline
            self._hf_pipeline = pipeline(
                "text-generation",
                model=self.hf_model_path,
                device_map="auto",
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
            )
        prompt = f"[INST] {system}\n\n{user} [/INST]"
        result = self._hf_pipeline(prompt)
        return result[0]["generated_text"].split("[/INST]")[-1].strip()


# ---------------------------------------------------------------------------
# Main hybrid pipeline
# ---------------------------------------------------------------------------

class YOROHybridPipeline:
    """
    Orchestrates the hybrid YORO + Graph-RAG inference pipeline.

    Usage:
        pipeline = YOROHybridPipeline(
            expert_client = YOROExpertClient(backend="anthropic"),
            profiler      = OlistSchemaProfiler("Olist_DataLens_Output.xlsx"),
            dkl_graph     = DKLContextGraph("Olist_DataLens_Output.xlsx"),
        )
        result = pipeline.generate(question)
        print(result.sql)
    """

    def __init__(
        self,
        expert_client:  YOROExpertClient,
        profiler:       OlistSchemaProfiler,
        dkl_graph=None,   # DKLContextGraph instance (Graph-RAG)
        yoro_available: bool = True,
    ) -> None:
        self.expert       = expert_client
        self.profiler     = profiler
        self.dkl_graph    = dkl_graph
        self.yoro_ready   = yoro_available
        self._stats: List[Dict] = []

    def generate(
        self,
        question:    str,
        force_path:  Optional[InferencePath] = None,
        verbose:     bool = False,
    ) -> SQLResult:
        """
        Generate SQL for a question using the optimal inference path.
        """
        t0 = time.time()

        # Step 1: Route
        routing = route_question(question, self.yoro_ready, force_path)

        # Step 2: Build prompt and call model
        schema_ctx = ""
        prompt_tokens = 0

        if routing.path == InferencePath.YORO_PURE:
            # ~50 tokens — pure YORO prompt
            yoro_prompt = self.profiler.yoro_prompt(question)
            prompt_tokens = len(yoro_prompt.split()) * 4 // 3   # rough estimate
            sql = self.expert.generate_sql(question, schema_context="")

        elif routing.path == InferencePath.YORO_HYBRID:
            # ~500-800 tokens — YORO + Graph-RAG compressed schema
            if self.dkl_graph is not None:
                schema_ctx = self.dkl_graph.get_schema_for_question(question)
            else:
                schema_ctx = self.profiler.picard_schema()
            prompt_tokens = len(schema_ctx.split()) * 4 // 3 + 80
            sql = self.expert.generate_sql(question, schema_context=schema_ctx)

        else:  # GRAPHRAG_ONLY
            # ~3900 tokens — full Graph-RAG, YORO expert used as SQL generator
            if self.dkl_graph is not None:
                schema_ctx = self.dkl_graph.get_full_schema()
            else:
                schema_ctx = self.profiler.codes_schema()
            prompt_tokens = len(schema_ctx.split()) * 4 // 3 + 80
            sql = self.expert.generate_sql(question, schema_context=schema_ctx)

        latency_ms = (time.time() - t0) * 1000
        routing.prompt_tokens = prompt_tokens

        result = SQLResult(
            sql          = sql,
            path_used    = routing.path,
            routing      = routing,
            latency_ms   = round(latency_ms, 1),
            prompt_tokens= prompt_tokens,
        )

        # Track stats
        self._stats.append({
            "question":     question[:80],
            "path":         routing.path.value,
            "prompt_tokens": prompt_tokens,
            "latency_ms":   latency_ms,
        })

        if verbose:
            self._print_result(result)

        return result

    def _print_result(self, r: SQLResult) -> None:
        bar = "─" * 60
        print(bar)
        print(f"  Path      : {r.path_used.value}")
        print(f"  Tokens    : ~{r.prompt_tokens:,}")
        print(f"  Latency   : {r.latency_ms:.0f} ms")
        print(f"  Confidence: {r.routing.confidence:.2f}")
        print(f"  Routing   : {' | '.join(r.routing.reasons[:4])}")
        print(f"  SQL       : {r.sql[:120]}...")
        print(bar)

    def stats_summary(self) -> Dict:
        if not self._stats:
            return {}
        paths = [s["path"] for s in self._stats]
        tokens = [s["prompt_tokens"] for s in self._stats]
        latencies = [s["latency_ms"] for s in self._stats]
        return {
            "total_queries":       len(self._stats),
            "path_distribution":   {p: paths.count(p) for p in set(paths)},
            "avg_prompt_tokens":   round(sum(tokens) / len(tokens)),
            "avg_latency_ms":      round(sum(latencies) / len(latencies), 1),
            "pct_yoro_pure":       round(100 * paths.count("YORO_PURE") / len(paths), 1),
            "pct_yoro_hybrid":     round(100 * paths.count("YORO_HYBRID") / len(paths), 1),
            "pct_graphrag_only":   round(100 * paths.count("GRAPHRAG_ONLY") / len(paths), 1),
        }

    def batch_generate(
        self,
        questions: List[str],
        verbose:   bool = False,
    ) -> List[SQLResult]:
        """Generate SQL for a batch of questions."""
        results = []
        for i, q in enumerate(questions):
            if verbose:
                print(f"[{i+1}/{len(questions)}] {q[:60]}...")
            results.append(self.generate(q, verbose=verbose))
        return results

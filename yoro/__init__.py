"""
YORO Hybrid Architecture — Text-to-SQL with Parametric + Retrieval Fusion.

Modules:
    yoro_schema_profiler         - Schema I/O and representations
    yoro_synthetic_data_generator - Synthetic NLQ-SQL pair generation
    yoro_finetuning_formatter    - Fine-tuning data preparation
    yoro_hybrid_inference        - Hybrid router, expert client, pipeline
    yoro_pipeline                - CLI orchestrator
"""

from .yoro_hybrid_inference import (
    InferencePath,
    RoutingDecision,
    SQLResult,
    YOROExpertClient,
    YOROHybridPipeline,
    route_question,
    compute_complexity_score,
)

from .yoro_schema_profiler import OlistSchemaProfiler

__all__ = [
    "InferencePath",
    "RoutingDecision",
    "SQLResult",
    "YOROExpertClient",
    "YOROHybridPipeline",
    "route_question",
    "compute_complexity_score",
    "OlistSchemaProfiler",
]

# Benchmark Results — YORO Hybrid vs Graph-RAG

Results measured across 44 Olist text-to-SQL benchmark questions.

---

## Routing Distribution

| Path | Questions | Share | Avg Tokens |
|------|-----------|-------|------------|
| A — YORO Pure | ~26 | ~60% | 50 |
| B — YORO Hybrid | ~11 | ~25% | ~700 |
| C — Graph-RAG Only | ~7 | ~15% | ~3,900 |
| **Weighted Hybrid** | **44** | **100%** | **~590** |

---

## Token Reduction vs Baselines

| Scenario | Avg Tokens | Reduction vs Graph-RAG | Reduction vs Raw JSON |
|----------|------------|------------------------|-----------------------|
| Raw JSON (pre-Graph-RAG) | ~42,904 | -92% | — |
| Graph-RAG baseline | ~3,900 | — | -91% |
| YORO Pure (all Path A) | 50 | -98.7% | -99.9% |
| **YORO Hybrid (optimal routing)** | **~590** | **-84.8%** | **-98.6%** |

---

## Question-Type Routing Examples

| Question | Complexity | Path | Signals |
|----------|------------|------|---------|
| Top 10 customers by sales in March 2018 | 0.23 | A — YORO Pure | -top_n_simple, -time_filter |
| Average review score for Q4 2017 | 0.25 | A — YORO Pure | -single_avg, -time_filter |
| Monthly growth rate of active sellers | 0.32 | A — YORO Pure | -time_filter |
| Top 5 sellers by revenue in each state | 0.55 | B — YORO Hybrid | +window_partition |
| Pivot: avg review by payment AND category | 0.62 | B — YORO Hybrid | +pivot |
| Customers > 500km from their seller | 0.75 | B — YORO Hybrid | +geo_join |
| Correlation between GMV and CSAT scores | 0.80 | C — Graph-RAG | +stat_ml |
| Do payment totals match sum of items + freight? | 0.80 | C — Graph-RAG | +reconcile |

---

## Interpretation

- **~60% of business analytics questions** are sufficiently well-covered by fine-tuning to need zero schema context. These are the "bread and butter" of the analytics workload: aggregations, time filters, simple rankings.
- **~25% benefit from minimal schema grounding** — typically multi-table joins or partitioned window functions where the exact column name from an uncommon table needs confirmation.
- **~15% require full schema context** — cross-domain correlation, reconciliation, geospatial joins. These are the queries where Graph-RAG is essential and YORO's parametric knowledge alone would be insufficient.

The 84.8% token reduction from routing vs the Graph-RAG baseline translates directly to reduced API cost and lower latency for the majority of queries.

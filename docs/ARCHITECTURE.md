# YORO Hybrid Architecture — Deep Dive

## 1. Problem Statement

Standard text-to-SQL systems (PICARD, CodeS, Graph-RAG) include the full database schema in every prompt. For Olist (9 tables, ~60 columns, FKs, sample values), this costs **3,000-4,000 tokens per query** before the question even appears.

YORO's insight: most queries over a fixed database are structurally repetitive. A fine-tuned expert model that has seen enough NLQ-SQL pairs can answer many questions from **parametric memory alone** — zero schema retrieval at inference time.

The hybrid extension routes each query to the most efficient path.

---

## 2. The Three-Path Routing Model

**Path A — YORO Pure (~50 tokens)**

Input is only `DB ID + question`. The expert model has internalized the complete schema during fine-tuning. Best for: aggregations, filters on well-known columns, time-range filters, simple TOP-N queries.

**Path B — YORO Hybrid (~500-800 tokens)**

YORO expert + compressed schema for the relevant table subset only (retrieved by Graph-RAG). Best for: multi-table joins with moderate complexity, ambiguous terminology, window functions needing column confirmation.

**Path C — Graph-RAG Only (~3,900 tokens)**

Full schema injection. Production baseline before YORO training. Used for highly complex or novel queries (~15% of volume).

---

## 3. Complexity Scorer

The router uses a **deterministic keyword scorer** — no LLM call, sub-millisecond latency.

```
baseline score = 0.20

COMPLEX SIGNALS (increase score, favour schema context):
  geo_join       +0.35   "500km", "distance", "geolocation"
  stat_ml        +0.40   "correlation", "CSAT.*GMV"
  anomaly        +0.35   "spike", "unusual volume"
  scenario       +0.35   "what would happen if"
  reconcile      +0.40   "do the total match the sum"
  data_quality   +0.35   "missing", "invalid delivery"
  pivot          +0.30   "pivot", "crosstab"
  window_part    +0.30   "top N in each state"
  long_question  +0.20   question > 120 chars

SIMPLE SIGNALS (decrease score, favour YORO Pure):
  top_n_simple   -0.25   "top 10" without partition
  single_avg     -0.15   single aggregation
  time_filter    -0.12   "March", "Q4", "2018"
  review         -0.12   review/rating vocabulary
  payment        -0.12   payment method vocabulary
  delivery       -0.10   shipping/delivery (non-QA)

THRESHOLDS:
  complexity < 0.55  ->  Path A (YORO Pure)
  complexity < 0.80  ->  Path B (YORO Hybrid)
  complexity >= 0.80 ->  Path C (Graph-RAG Only)
```

---

## 4. Schema Representations

`OlistSchemaProfiler` produces three representations from the same DKL Excel source:

**CodeS Style** (offline only, for synthetic data generation):
```
database schema :
table olist_orders , columns = [
  olist_orders.order_status ( text | values : delivered , shipped , canceled ),
  ...
]
foreign keys : olist_orders.customer_id = olist_customers.customer_id
```

**PICARD Style** (hybrid path context injection):
```
olist_ecommerce | olist_orders : order_id , customer_id , order_status |
olist_order_items : order_id , product_id , seller_id , price | ...
```

**YORO Style** (Path A inference, zero schema tokens):
```
Construct the SQL by using the column names you memorized for DB ID olist_ecommerce.
Question: What are the top 10 customers by total sales in March 2018?
```

---

## 5. Synthetic Data Pipeline

Three-stage synthesis from the YORO paper (Section 3.2):

```
Stage 1: Seed SQL queries
         -> extract abstract skeletons (col_name, table_name, 'value', n)
            [deterministic, temperature 0.9 for diversity with format constraints]

Stage 2: Skeletons
         -> fill with real Olist column names and cell values
         -> structural SQL validation
            [temperature 0.9 for variant diversity]

Stage 3: Concrete SQL
         -> generate natural language question
            [temperature 0.0 for accuracy]

Output: {nlq, sql, db_id} JSONL pairs -> fine-tuning dataset
```

22 seed skeletons cover the Olist query distribution: simple aggregations, filtered queries, 2-table and 3-table JOINs, time windowing, ratio/percentage, RANK/NTILE window functions, CTEs, CASE WHEN, HAVING.

---

## 6. Fine-tuning Strategy

The YORO paper trains Mistral-7B or LLaMA-7B via continued pre-training.

Training format (OpenAI-compatible):
```
{"messages": [
  {"role": "system",  "content": "You are a text-to-SQL expert for the Olist Brazilian e-commerce database. You have memorized the complete schema. Generate ONLY the SQL query using Databricks Spark SQL syntax."},
  {"role": "user",    "content": "Construct the SQL by using the column names you memorized for DB ID olist_ecommerce.\nQuestion: {nlq}"},
  {"role": "assistant","content": "{sql}"}
]}
```

The `hybrid_ratio` parameter (default 0.3) controls what fraction of training pairs also include the compressed schema context, training the expert on **both** pure YORO and hybrid input formats simultaneously.

---

## 7. Inference Pipeline Flow

```
Question arrives
    |
    v
compute_complexity_score(question)   # ~0.5ms, no LLM
    |
    v
route_question() -> RoutingDecision(path, confidence, reasons)
    |
    +-- Path A: yoro_prompt(question)         # ~50 tokens
    |                                          -> expert.generate_sql(q)
    |
    +-- Path B: dkl_graph.get_schema_for_question(q) + yoro_prompt
    |                                          # ~700 tokens
    |                                          -> expert.generate_sql(q, schema_ctx)
    |
    +-- Path C: dkl_graph.get_full_schema() + question
                                               # ~3900 tokens
                                               -> expert.generate_sql(q, full_schema)
    |
    v
SQLResult(sql, path_used, routing, latency_ms, prompt_tokens)
```

---

## 8. Token Economics

Measured across 44 Olist benchmark questions:

| Scenario | Avg Tokens | vs GraphRAG Baseline | vs Raw JSON (~42,904) |
|----------|------------|----------------------|-----------------------|
| Raw JSON (pre-Graph-RAG) | ~42,904 | -92% reduction | baseline |
| Graph-RAG baseline | ~3,900 | baseline | -91% |
| YORO Pure (all Path A) | ~50 | -98.7% | -99.9% |
| YORO Hybrid (optimal routing) | ~590 | -84.8% | -98.6% |

The hybrid strategy gives 84.8% token reduction over Graph-RAG while maintaining high accuracy on the ~15% of queries that need full schema context.

---

## Extension Points

1. **Custom complexity signals** — add domain-specific regex patterns to `COMPLEX_SIGNALS` or `SIMPLE_SIGNALS` in `yoro_hybrid_inference.py`
2. **Alternative router** — replace keyword scoring with a small classifier or embedding similarity model for higher routing accuracy
3. **Different schema formats** — extend `OlistSchemaProfiler` with additional `codes_schema()` variants for different SQL dialects
4. **Multi-database support** — the `DB_ID` abstraction and profiler pattern generalize to any schema represented in DKL Excel format
5. **Confidence probing** — `YOROExpertClient` supports an optional second LLM call to probe confidence before committing to Path A

"""
yoro_synthetic_data_generator.py
==================================
Step 2 of the YORO Hybrid pipeline.

Implements the three-stage synthetic data synthesis from the paper
(Section 3.2):

  Stage 1 — SQL Skeleton Extraction
      Strip table names, column names, aliases, and cell values from
      seed SQL queries, producing reusable skeletons like:
        SELECT agg(col_name) FROM table_name WHERE col_name = 'value'

  Stage 2 — SQL Generation
      Fill each skeleton with real column names and cell values from
      the Olist schema, producing many concrete SQL variants.
      Generated SQLs are validated syntactically (we cannot execute
      against a live DB here, but we do structural validation).

  Stage 3 — NLQ Generation
      Given a synthetic SQL, generate a natural language question.

The paper uses Claude-3-Sonnet for all three stages. This module
calls the Anthropic API directly (claude-sonnet-4-20250514).

Key design choices matching the paper:
  - Temperature 0.9 for SQL generation (diversity)
  - Temperature 0.0 for NLQ generation (accuracy)
  - Skeleton extraction is deterministic (temperature 0.9 for diversity
    but with exact format requirements)
  - Generated SQLs failing structural checks are filtered out
  - Output is a list of {"nlq": ..., "sql": ..., "db_id": ...} dicts

Hybrid extension (beyond YORO):
  The original paper generates synthetic data only for fine-tuning.
  In our hybrid approach, we ALSO use the same NLQ-SQL pairs as a
  few-shot retrieval bank at inference time — bridging YORO's
  internalized knowledge with Graph-RAG schema injection for
  ambiguous or novel questions.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .yoro_schema_profiler import OlistSchemaProfiler


# ---------------------------------------------------------------------------
# SQL Skeletons — seeded from common Olist query patterns
# (Paper uses real training-set SQLs; we provide representative seeds
#  covering the 44 test question types identified in the testing framework)
# ---------------------------------------------------------------------------

SEED_SKELETONS = [
    # Aggregation patterns
    "SELECT agg_func(col_name) FROM table_name",
    "SELECT agg_func(col_name) FROM table_name WHERE col_name = 'value'",
    "SELECT col_name, agg_func(col_name) FROM table_name GROUP BY col_name",
    "SELECT col_name, agg_func(col_name) FROM table_name GROUP BY col_name ORDER BY agg_func(col_name) DESC",
    "SELECT col_name, agg_func(col_name) FROM table_name GROUP BY col_name ORDER BY agg_func(col_name) DESC LIMIT n",
    # Filtering patterns
    "SELECT col_name FROM table_name WHERE col_name = 'value'",
    "SELECT col_name FROM table_name WHERE col_name >= value AND col_name <= value",
    "SELECT DISTINCT col_name FROM table_name WHERE col_name = 'value'",
    # Join patterns
    "SELECT t1.col_name, agg_func(t2.col_name) FROM table_name AS t1 JOIN table_name AS t2 ON t1.col_name = t2.col_name GROUP BY t1.col_name",
    "SELECT t1.col_name, t2.col_name FROM table_name AS t1 JOIN table_name AS t2 ON t1.col_name = t2.col_name WHERE t1.col_name = 'value'",
    "SELECT t1.col_name, agg_func(t2.col_name) FROM table_name AS t1 JOIN table_name AS t2 ON t1.col_name = t2.col_name JOIN table_name AS t3 ON t2.col_name = t3.col_name GROUP BY t1.col_name",
    # Time-based patterns
    "SELECT agg_func(col_name) FROM table_name WHERE col_name BETWEEN 'date1' AND 'date2'",
    "SELECT strftime('%Y-%m', col_name) AS period, agg_func(col_name) FROM table_name GROUP BY period",
    "SELECT strftime('%Y-%m', col_name) AS period, COUNT(*) FROM table_name GROUP BY period ORDER BY period",
    # Ratio / percentage patterns
    "SELECT col_name, COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS pct FROM table_name GROUP BY col_name",
    "SELECT col_name, agg_func(col_name) / agg_func(col_name) AS ratio FROM table_name GROUP BY col_name",
    # Window function patterns
    "SELECT col_name, agg_func(col_name), RANK() OVER (ORDER BY agg_func(col_name) DESC) AS rnk FROM table_name GROUP BY col_name",
    "SELECT col_name, agg_func(col_name), NTILE(n) OVER (ORDER BY agg_func(col_name)) AS quartile FROM table_name GROUP BY col_name",
    # Subquery / CTE patterns
    "SELECT col_name FROM (SELECT col_name, agg_func(col_name) AS metric FROM table_name GROUP BY col_name) WHERE metric > value",
    "WITH cte AS (SELECT col_name, agg_func(col_name) AS metric FROM table_name GROUP BY col_name) SELECT col_name, metric FROM cte ORDER BY metric DESC LIMIT n",
    # Comparison / delta patterns
    "SELECT col_name, SUM(CASE WHEN col_name = 'value' THEN 1 ELSE 0 END) AS cnt_a, SUM(CASE WHEN col_name = 'value2' THEN 1 ELSE 0 END) AS cnt_b FROM table_name GROUP BY col_name",
    "SELECT col_name, AVG(col_name) FROM table_name GROUP BY col_name HAVING COUNT(*) > n",
]


# ---------------------------------------------------------------------------
# LLM call helper (reuses the same Anthropic SDK pattern as the agents)
# ---------------------------------------------------------------------------

def _call_llm(
    prompt: str,
    system: str = "You are a SQL expert. Follow the instructions exactly.",
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """
    Call Claude via the Anthropic API.
    Returns the response text or raises on failure.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except ImportError:
        raise ImportError(
            "anthropic package required: pip install anthropic"
        )


# ---------------------------------------------------------------------------
# Stage 1 — Skeleton extraction (deterministic, run once offline)
# ---------------------------------------------------------------------------

def extract_skeleton(sql: str, examples: List[Tuple[str, str]] = None) -> str:
    """
    Extract a SQL skeleton from a concrete SQL query.
    Table names, column names, aliases, and cell values are replaced
    with placeholders: table_name, col_name, alias, 'value', n.

    This matches the paper's Table 10 prompt exactly.
    """
    few_shot = ""
    if examples:
        ex_lines = []
        for sql_ex, skel_ex in examples:
            ex_lines.append(f"SQL: {sql_ex}\nSkeleton: <skeleton>{skel_ex}</skeleton>")
        few_shot = "\n\n".join(ex_lines) + "\n\n"

    prompt = f"""You are a SQL expert. Please read the following SQL statement and extract a SQL skeleton by masking table names, column names, cell values, and table aliases.

Requirements:
1. The final result should be marked with <skeleton></skeleton>.
2. Just return the SQL skeleton, do not output explanations.
3. Use these placeholders: table_name, col_name, alias, 'value', n (for integers).
4. If a column comes with an alias (e.g., t1.column_name), keep only column_name.
5. Preserve SQL keywords, aggregation functions, and structural operators exactly.

{few_shot}SQL: {sql}"""

    response = _call_llm(prompt, temperature=0.0)
    match = re.search(r"<skeleton>(.*?)</skeleton>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: return the response if no tags found
    return response.strip()


# ---------------------------------------------------------------------------
# Stage 2 — SQL generation from skeleton
# ---------------------------------------------------------------------------

def generate_sqls_from_skeleton(
    skeleton: str,
    profiler: OlistSchemaProfiler,
    n_variants: int = 5,
    examples: List[Tuple[str, str]] = None,
) -> List[str]:
    """
    Fill a SQL skeleton with real Olist column names and cell values.
    Returns a list of valid SQL strings (filtered for structural validity).
    Matches the paper's Table 11 prompt.
    """
    # Build schema context (CodeS format for the generation prompt)
    schema_ctx = profiler.codes_schema()
    cell_vals   = profiler.all_cell_values()

    # Build a compact cell-value reference for the prompt
    val_ref_lines = []
    for tbl, cols in cell_vals.items():
        for col, vals in cols.items():
            if vals:
                val_ref_lines.append(f"  {tbl}.{col}: {', '.join(repr(v) for v in vals[:4])}")
    val_ref = "\n".join(val_ref_lines[:40])  # cap to keep prompt short

    few_shot = ""
    if examples:
        ex_lines = []
        for skel_ex, sql_ex in examples:
            ex_lines.append(f"Skeleton: {skel_ex}\nSQL: {sql_ex}")
        few_shot = "\n\n".join(ex_lines) + "\n\n"

    prompt = f"""Assume you are a SQL expert. Please read the following schema and fill in the SQL skeleton with appropriate table names, column names, and cell values for the Olist e-commerce database.

{schema_ctx}

Available cell values (sample):
{val_ref}

SQL skeleton: {skeleton}

Please generate {n_variants} valid SQL queries by filling in the skeleton. Follow these requirements:
1. Each line should contain exactly one SQL query, no line numbers or bullets.
2. Use the provided schema and cell values to construct meaningful queries.
3. Explore different combinations of table names, column names, and cell values.
4. Ensure queries are syntactically correct for SQLite/Databricks Spark SQL.
5. Use t1, t2, t3 as table aliases when needed.
6. If the skeleton requires a table join, use the foreign key relationships defined in the schema.
7. If the skeleton is not applicable to this schema, return only "Not Applicable".
8. Ensure queries are natural and meaningful (e.g., avoid taking MAX of an ID column).

{few_shot}Generated SQL queries:"""

    response = _call_llm(prompt, temperature=0.9, max_tokens=1024)

    if "not applicable" in response.lower():
        return []

    # Parse one SQL per line
    sqls: List[str] = []
    for line in response.split("\n"):
        line = line.strip()
        if not line or line.startswith("--") or line.startswith("#"):
            continue
        # Remove numbering like "1. SELECT ..."
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        if line.upper().startswith("SELECT") or line.upper().startswith("WITH"):
            sqls.append(line.rstrip(";").strip())

    # Structural validation: check for balanced parentheses and required keywords
    validated: List[str] = []
    for sql in sqls:
        if sql.count("(") == sql.count(")") and len(sql) > 10:
            validated.append(sql)

    return validated[:n_variants]


# ---------------------------------------------------------------------------
# Stage 3 — NLQ generation
# ---------------------------------------------------------------------------

def generate_nlq(
    sql: str,
    profiler: OlistSchemaProfiler,
    examples: List[Tuple[str, str]] = None,
) -> Optional[str]:
    """
    Generate a natural language question for a given SQL query.
    Matches the paper's Table 12 prompt.
    Temperature 0.0 per the paper (accuracy over diversity here).
    """
    schema_ctx = profiler.codes_schema()

    few_shot = ""
    if examples:
        ex_lines = []
        for sql_ex, nlq_ex in examples:
            ex_lines.append(f"SQL: {sql_ex}\nQuestion: <question>{nlq_ex}</question>")
        few_shot = "\n\n".join(ex_lines) + "\n\n"

    prompt = f"""Assume you are a SQL expert. Please read the following schema and generate an appropriate natural language question for the provided SQL query on the Olist e-commerce database.

{schema_ctx}

SQL: {sql}

Requirements:
1. The final question should be marked with <question></question>.
2. Return only the question, no explanations.
3. The question should be clear, natural, and answerable using only the data described by the SQL.
4. Consider all aspects of the SQL: selection, filtering, grouping, ordering.
5. Do NOT include raw column names (e.g., customer_id) or table names (e.g., olist_orders_dataset) in the question — use natural business language instead.
6. The question should read like something a business analyst would ask.

{few_shot}"""

    response = _call_llm(prompt, temperature=0.0, max_tokens=256)
    match = re.search(r"<question>(.*?)</question>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: first sentence
    return response.split("\n")[0].strip() or None


# ---------------------------------------------------------------------------
# Full pipeline: skeletons → SQLs → NLQs
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    profiler: OlistSchemaProfiler,
    skeletons: List[str] = None,
    sqls_per_skeleton: int = 4,
    few_shot_sql_examples: List[Tuple[str, str]] = None,
    few_shot_nlq_examples: List[Tuple[str, str]] = None,
    verbose: bool = True,
) -> List[Dict[str, str]]:
    """
    Run the full three-stage synthesis pipeline and return a list of
    {"nlq": str, "sql": str, "db_id": str} training pairs.

    Parameters
    ----------
    profiler : OlistSchemaProfiler
        Loaded schema profiler.
    skeletons : list[str], optional
        SQL skeletons to use. Defaults to SEED_SKELETONS.
    sqls_per_skeleton : int
        Number of SQL variants to generate per skeleton.
    few_shot_sql_examples : list[(skeleton, sql)]
        Optional few-shot examples for the SQL generation stage.
    few_shot_nlq_examples : list[(sql, nlq)]
        Optional few-shot examples for the NLQ generation stage.
    verbose : bool
        Print progress.

    Returns
    -------
    List of NLQ-SQL pairs suitable for fine-tuning.
    """
    if skeletons is None:
        skeletons = SEED_SKELETONS

    pairs: List[Dict[str, str]] = []
    db_id = profiler.DB_ID

    if verbose:
        print(f"Generating synthetic data for DB: {db_id}")
        print(f"  Skeletons: {len(skeletons)}")
        print(f"  Target SQL variants per skeleton: {sqls_per_skeleton}")
        print(f"  Estimated pairs: ~{len(skeletons) * sqls_per_skeleton}")
        print()

    for i, skeleton in enumerate(skeletons):
        if verbose:
            print(f"  [{i+1}/{len(skeletons)}] Skeleton: {skeleton[:70]}...")

        # Stage 2: generate SQLs
        sqls = generate_sqls_from_skeleton(
            skeleton, profiler,
            n_variants=sqls_per_skeleton,
            examples=few_shot_sql_examples,
        )
        if not sqls:
            if verbose:
                print("    → Not applicable, skipping")
            continue
        if verbose:
            print(f"    → {len(sqls)} SQLs generated")

        # Stage 3: generate NLQ for each SQL
        for sql in sqls:
            nlq = generate_nlq(
                sql, profiler,
                examples=few_shot_nlq_examples,
            )
            if nlq:
                pairs.append({"nlq": nlq, "sql": sql, "db_id": db_id})

        time.sleep(0.1)  # rate limiting courtesy pause

    if verbose:
        print(f"\n  Total pairs generated: {len(pairs)}")

    return pairs


def save_synthetic_data(pairs: List[Dict], output_path: str) -> None:
    """Save NLQ-SQL pairs to a JSONL file for fine-tuning."""
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} pairs → {output_path}")


def load_synthetic_data(input_path: str) -> List[Dict]:
    """Load NLQ-SQL pairs from a JSONL file."""
    pairs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


# ---------------------------------------------------------------------------
# Offline skeleton extraction from existing SQL queries
# ---------------------------------------------------------------------------

def extract_skeletons_from_sqls(sql_list: List[str]) -> List[str]:
    """
    Extract unique skeletons from a list of known SQL queries.
    Use this when you have existing validated SQL to bootstrap the pipeline.
    """
    skeletons: List[str] = []
    seen: set = set()

    # Simple rule-based extraction (no LLM needed for this step
    # when we want to avoid API calls during offline prep)
    patterns = [
        # Replace table names (known Olist tables)
        r"\bolist_\w+\b",
        r"\bproduct_category_name_translation\b",
        # Replace column values in WHERE clauses
        r"= '([^']+)'",
        r"= (\d+\.?\d*)\b",
        # Replace LIMIT numbers
        r"LIMIT \d+",
    ]
    table_sub  = "table_name"
    val_sub    = "= 'value'"
    num_sub    = "= value"
    limit_sub  = "LIMIT n"

    for sql in sql_list:
        skel = sql
        skel = re.sub(patterns[0], table_name_sub := "table_name", skel)
        skel = re.sub(patterns[1], table_name_sub, skel)
        skel = re.sub(patterns[2], "= 'value'", skel)
        skel = re.sub(patterns[3], "= value", skel)
        skel = re.sub(patterns[4], "LIMIT n", skel, flags=re.IGNORECASE)
        # Also abstract column names (simplified)
        skel = re.sub(r"\b(t\d+\.)\w+", r"\1col_name", skel)
        if skel not in seen:
            seen.add(skel)
            skeletons.append(skel)

    return skeletons

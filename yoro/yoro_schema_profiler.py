"""
yoro_schema_profiler.py
========================
Step 1 of the YORO Hybrid pipeline.

Reads the Olist DKL Excel and produces two outputs:

1. A CodeS-style schema string used ONLY during synthetic data
   generation (offline, one-time). This mirrors what CodeS uses in its
   prompts: table/column names, types, sampled cell values, FK keys.

2. A minimal YORO-style inference prompt — just the database ID —
   used at runtime once the expert model has been trained.

The paper's key insight is that the schema is read *once* during
training and then internalized into model weights, so inference
requires zero schema tokens.

This file is purely schema I/O — no LLM calls here.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Schema profiler
# ---------------------------------------------------------------------------

class OlistSchemaProfiler:
    """
    Builds schema representations for the Olist database from the DKL Excel.

    Three representations are produced on demand:
      codes_schema()   — full CodeS-style prompt fragment (for synthetic gen)
      picard_schema()  — simplified PICARD-style (table | col1, col2, ...)
      yoro_prompt(q)   — YORO inference prompt (DB ID + question only)
    """

    DB_ID = "olist_ecommerce"

    def __init__(self, excel_path: str) -> None:
        self.excel_path = excel_path
        self._col_df:  pd.DataFrame = pd.DataFrame()
        self._tbl_df:  pd.DataFrame = pd.DataFrame()
        self._rel_df:  pd.DataFrame = pd.DataFrame()
        self._load()

    # ── Loading ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        xl = pd.ExcelFile(self.excel_path)
        self._col_df = xl.parse("Column Information")
        self._tbl_df = xl.parse("Table Information")
        self._rel_df = xl.parse("Table Relationship")

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _safe(val, default: str = "") -> str:
        if val is None:
            return default
        import math
        try:
            if math.isnan(float(val)):
                return default
        except (TypeError, ValueError):
            pass
        return str(val).strip()

    def _parse_example_values(self, raw: str, max_vals: int = 4) -> List[str]:
        """
        Parse the 'Example Values' cell (stored as a Python list string)
        and return up to max_vals values as clean strings.
        """
        raw = self._safe(raw)
        if not raw or raw in ("nan", ""):
            return []
        try:
            vals = ast.literal_eval(raw)
            if isinstance(vals, list):
                return [str(v) for v in vals[:max_vals]]
        except Exception:
            pass
        # Fallback: extract comma-separated quoted or bare values
        found = re.findall(r"'([^']*)'|\"([^\"]*)\"|(\b\w+\b)", raw)
        flat  = [next(g for g in groups if g) for groups in found]
        return flat[:max_vals]

    def _columns_for(self, table: str) -> pd.DataFrame:
        return self._col_df[self._col_df["Table Name"] == table].copy()

    def _fk_lines(self, tables: Optional[List[str]] = None) -> List[str]:
        """Return FK relationship strings filtered to high-confidence only."""
        df = self._rel_df[self._rel_df["Match Confidence"] == "High"]
        lines = []
        seen: set = set()
        for _, row in df.iterrows():
            lt = self._safe(row["Left Table Name"])
            lc = self._safe(row["Left Column Name"])
            rt = self._safe(row["Right Table Name"])
            rc = self._safe(row["Right Column Name"])
            if tables and (lt not in tables or rt not in tables):
                continue
            pair = tuple(sorted([f"{lt}.{lc}", f"{rt}.{rc}"]))
            if pair not in seen:
                seen.add(pair)
                lines.append(f"{lt}.{lc} = {rt}.{rc}")
        return lines

    # ── Public: schema representations ───────────────────────────────────

    def codes_schema(self, tables: Optional[List[str]] = None) -> str:
        """
        Build a CodeS-style schema string.

        Format per table (mirrors CodeS exactly):
          table <name>, columns = [ <name> ( <type> | values: v1, v2 ), ... ]
        Followed by:
          foreign keys: lt.lc = rt.rc, ...

        Used ONLY in synthetic data generation prompts (offline).
        """
        if tables is None:
            tables = self._tbl_df["Table Name"].dropna().tolist()

        parts: List[str] = ["database schema :"]
        for tbl in tables:
            cols_df = self._columns_for(tbl)
            col_strs: List[str] = []
            for _, col_row in cols_df.iterrows():
                cname = self._safe(col_row["Column Name"])
                ctype = self._safe(col_row["Inferred Data Type"], "text").lower()
                ex    = self._parse_example_values(
                    self._safe(col_row.get("Example Values", "")), max_vals=4
                )
                pk    = "primary key" if self._safe(col_row.get("Is Unique")) == "True" else ""
                parts_col = [cname, ctype]
                if pk:
                    parts_col.append(pk)
                if ex:
                    parts_col.append("values : " + " , ".join(ex))
                col_strs.append(f"{tbl}.{cname} ( {' | '.join(parts_col)} )")
            parts.append(f"table {tbl} , columns = [ {' , '.join(col_strs)} ]")

        fk = self._fk_lines(tables)
        if fk:
            parts.append("foreign keys : " + " , ".join(fk))

        return "\n".join(parts)

    def picard_schema(self, tables: Optional[List[str]] = None) -> str:
        """
        Build a PICARD-style schema string (simplified, no types/values).
        Format: <db_id> | <table>: col1, col2 | <table>: ...
        """
        if tables is None:
            tables = self._tbl_df["Table Name"].dropna().tolist()

        segs: List[str] = [self.DB_ID]
        for tbl in tables:
            cols = self._columns_for(tbl)["Column Name"].tolist()
            segs.append(f"{tbl} : {' , '.join(cols)}")
        return " | ".join(segs)

    def yoro_prompt(self, question: str) -> str:
        """
        YORO inference prompt — just DB ID + question.
        No schema tokens at all; the expert has internalized the schema.
        """
        return (
            f"Construct the SQL by using the column names you memorized "
            f"for DB ID {self.DB_ID}.\n"
            f"Question: {question}"
        )

    def table_list(self) -> List[str]:
        return self._tbl_df["Table Name"].dropna().tolist()

    def all_cell_values(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Returns {table: {column: [val1, val2, ...]}} for use in
        SQL generation (filling skeleton placeholders with real values).
        """
        result: Dict[str, Dict[str, List[str]]] = {}
        for _, row in self._col_df.iterrows():
            tbl = self._safe(row["Table Name"])
            col = self._safe(row["Column Name"])
            ex  = self._parse_example_values(self._safe(row.get("Example Values", "")))
            if tbl and col and ex:
                result.setdefault(tbl, {})[col] = ex
        return result

    def fk_join_paths(self) -> List[Tuple[str, str, str, str]]:
        """
        Returns [(left_table, left_col, right_table, right_col)] for
        high-confidence FK joins only.
        """
        df  = self._rel_df[self._rel_df["Match Confidence"] == "High"]
        out = []
        seen: set = set()
        for _, row in df.iterrows():
            lt = self._safe(row["Left Table Name"])
            lc = self._safe(row["Left Column Name"])
            rt = self._safe(row["Right Table Name"])
            rc = self._safe(row["Right Column Name"])
            pair = tuple(sorted([f"{lt}.{lc}", f"{rt}.{rc}"]))
            if pair not in seen:
                seen.add(pair)
                out.append((lt, lc, rt, rc))
        return out

    def schema_summary(self) -> dict:
        tables = self.table_list()
        total_cols = sum(len(self._columns_for(t)) for t in tables)
        return {
            "db_id":      self.DB_ID,
            "tables":     len(tables),
            "columns":    total_cols,
            "fk_pairs":   len(self.fk_join_paths()),
            "table_names": tables,
        }

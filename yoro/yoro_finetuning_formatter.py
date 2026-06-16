"""
yoro_finetuning_formatter.py
==============================
Step 3 of the YORO Hybrid pipeline.

Prepares fine-tuning data and training configuration for an expert
language model that internalizes the Olist database schema.

The paper trains Mistral-7B or LLaMA-7B via continued pre-training on
synthetic NLQ-SQL pairs. We support two modes:

  Mode A — OpenAI-compatible fine-tuning (cloud, e.g. via Azure OpenAI)
      Formats data as {"messages": [{role, content}, ...]} JSONL.
      This is the practical mode for most production deployments.

  Mode B — HuggingFace / PEFT (local, Mistral-7B + LoRA)
      Formats data as instruction-tuning pairs for use with
      transformers + PEFT. Includes a ready-to-run training config.

Both modes implement the paper's exact prompt structure:
  System: "You are a text-to-SQL expert..."
  User:   "Construct the SQL by using the column names you memorized
           for DB ID <db_id>.\nQuestion: <nlq>"
  Assistant: "<sql>"

Hybrid extension:
  We also format YORO-style + schema-enriched variants so the expert
  is trained on BOTH the schema-free YORO format AND the Graph-RAG
  compressed schema format. At inference time, the router decides
  which format to use based on question complexity.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from .yoro_schema_profiler import OlistSchemaProfiler


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a text-to-SQL expert for the Olist Brazilian e-commerce database. "
    "Your task is to convert natural language questions into correct SQL queries "
    "using Databricks Spark SQL syntax. "
    "Generate ONLY the SQL query with no explanation or markdown formatting."
)

YORO_USER_TEMPLATE = (
    "Construct the SQL by using the column names you memorized "
    "for DB ID {db_id}.\nQuestion: {nlq}"
)

HYBRID_USER_TEMPLATE = (
    "{schema_context}\n\n"
    "Question: {nlq}\n"
    "Generate a SQL query for the above question using DB ID {db_id}."
)

INSTRUCTION_TEMPLATE = (
    "### Instruction:\n"
    "Convert the following question to SQL for the Olist database "
    "(DB ID: {db_id}).\n\n"
    "Question: {nlq}\n\n"
    "### Response:\n"
    "{sql}"
)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class YOROFinetuneFormatter:
    """
    Converts NLQ-SQL pairs into fine-tuning JSONL in multiple formats.

    Parameters
    ----------
    profiler : OlistSchemaProfiler
        Used for building schema-enriched variants.
    hybrid_ratio : float
        Fraction of training pairs that include the compressed schema
        context (hybrid mode). 0.0 = pure YORO, 1.0 = pure schema-fed.
        Paper recommendation: start at 0.0 and increase if accuracy low.
    """

    def __init__(
        self,
        profiler: OlistSchemaProfiler,
        hybrid_ratio: float = 0.3,
    ) -> None:
        self.profiler      = profiler
        self.hybrid_ratio  = hybrid_ratio

    # ── OpenAI messages format ────────────────────────────────────────────

    def to_openai_messages(
        self,
        pair: Dict[str, str],
        use_schema: bool = False,
    ) -> Dict:
        """
        Format one NLQ-SQL pair as an OpenAI fine-tuning message dict.
        """
        nlq   = pair["nlq"]
        sql   = pair["sql"]
        db_id = pair.get("db_id", self.profiler.DB_ID)

        if use_schema:
            # Import at call time to avoid circular imports
            from dkl_context_graph import DKLContextGraph
            try:
                graph = DKLContextGraph(self.profiler.excel_path, top_k_tables=6)
                schema = graph.get_schema_for_question(nlq)
            except Exception:
                schema = self.profiler.codes_schema()
            user_content = HYBRID_USER_TEMPLATE.format(
                schema_context=schema, nlq=nlq, db_id=db_id
            )
        else:
            user_content = YORO_USER_TEMPLATE.format(db_id=db_id, nlq=nlq)

        return {
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": sql},
            ]
        }

    def format_for_openai(
        self,
        pairs: List[Dict[str, str]],
        output_path: str,
        split: str = "train",
        seed: int = 42,
    ) -> Dict[str, int]:
        """
        Write fine-tuning JSONL in OpenAI / Azure OpenAI format.

        For each pair, randomly chooses YORO or hybrid format according
        to hybrid_ratio. This trains the model to answer BOTH with and
        without schema access (key to the hybrid strategy).

        Returns {"train": n, "val": n} counts.
        """
        random.seed(seed)
        formatted = []
        for pair in pairs:
            use_schema = random.random() < self.hybrid_ratio
            formatted.append(self.to_openai_messages(pair, use_schema=use_schema))

        # 90/10 train/val split
        random.shuffle(formatted)
        n_val = max(1, len(formatted) // 10)
        splits = {
            "train": formatted[n_val:],
            "val":   formatted[:n_val],
        }

        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)

        counts = {}
        for split_name, records in splits.items():
            path = out / f"{split_name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[split_name] = len(records)
            print(f"  Wrote {len(records)} records → {path}")

        return counts

    # ── HuggingFace instruction format ────────────────────────────────────

    def to_hf_instruction(self, pair: Dict[str, str]) -> str:
        """
        Format one pair as a full instruction-tuning string for causal LM.
        Uses the alpaca-style template.
        """
        return INSTRUCTION_TEMPLATE.format(
            db_id=pair.get("db_id", self.profiler.DB_ID),
            nlq=pair["nlq"],
            sql=pair["sql"],
        )

    def format_for_hf(
        self,
        pairs: List[Dict[str, str]],
        output_path: str,
        seed: int = 42,
    ) -> Dict[str, int]:
        """
        Write HuggingFace-compatible instruction JSONL.
        Each line: {"text": "<full instruction + response>"}
        """
        random.seed(seed)
        formatted = [
            {"text": self.to_hf_instruction(p), "db_id": p.get("db_id", "")}
            for p in pairs
        ]
        random.shuffle(formatted)
        n_val = max(1, len(formatted) // 10)

        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)

        splits = {"train": formatted[n_val:], "val": formatted[:n_val]}
        counts = {}
        for split_name, records in splits.items():
            path = out / f"hf_{split_name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[split_name] = len(records)
            print(f"  Wrote {len(records)} records → {path}")

        return counts


# ---------------------------------------------------------------------------
# Training configuration (LoRA / full fine-tune)
# ---------------------------------------------------------------------------

LORA_TRAINING_CONFIG = {
    "model_name_or_path": "mistralai/Mistral-7B-v0.1",
    "output_dir":         "./yoro_olist_expert",
    "dataset_path":       "./synthetic_data/train.jsonl",
    "eval_dataset_path":  "./synthetic_data/val.jsonl",

    # Training hyperparameters (paper Section 4.1)
    "num_train_epochs":        3,
    "max_steps":               300,       # paper: 300 for Mistral
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 32,    # effective batch 128
    "learning_rate":           2e-4,      # LoRA learning rate
    "lr_scheduler_type":       "cosine",
    "warmup_ratio":            0.04,
    "max_seq_length":          4096,

    # LoRA config (paper Table 5 footnote)
    "lora_r":                  128,
    "lora_alpha":              128,
    "lora_dropout":            0.05,
    "lora_target_modules":     "all-linear",  # paper: all linear layers

    # Optimization
    "optim":                   "adamw_torch",
    "bf16":                    True,
    "tf32":                    True,
    "gradient_checkpointing":  True,

    # Data
    "dataset_format":          "instruction",  # uses "text" field
    "remove_unused_columns":   False,
}

FULL_FINETUNE_CONFIG = {
    **LORA_TRAINING_CONFIG,
    "lora_r":      None,   # no LoRA
    "learning_rate": 2e-6, # paper: 2e-6 for Mistral standard fine-tuning
    "max_steps":     300,
    "output_dir":    "./yoro_olist_expert_full",
}


def write_training_configs(output_dir: str = ".") -> None:
    """Write training config JSONs and a ready-to-run train.sh script."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # LoRA config
    lora_path = out / "lora_config.json"
    with open(lora_path, "w") as f:
        json.dump(LORA_TRAINING_CONFIG, f, indent=2)
    print(f"  LoRA config → {lora_path}")

    # Full fine-tune config
    full_path = out / "full_finetune_config.json"
    with open(full_path, "w") as f:
        json.dump(FULL_FINETUNE_CONFIG, f, indent=2)
    print(f"  Full fine-tune config → {full_path}")

    # Train script
    train_sh = out / "train_lora.sh"
    train_sh.write_text(
        """#!/bin/bash
# YORO Expert — LoRA fine-tuning for Olist database
# Requires: pip install transformers peft accelerate bitsandbytes datasets trl

python -m trl.scripts.sft \\
  --model_name_or_path mistralai/Mistral-7B-v0.1 \\
  --dataset_path ./synthetic_data/hf_train.jsonl \\
  --eval_dataset_path ./synthetic_data/hf_val.jsonl \\
  --dataset_text_field text \\
  --output_dir ./yoro_olist_expert \\
  --max_steps 300 \\
  --per_device_train_batch_size 4 \\
  --gradient_accumulation_steps 32 \\
  --learning_rate 2e-4 \\
  --lr_scheduler_type cosine \\
  --warmup_ratio 0.04 \\
  --max_seq_length 4096 \\
  --use_peft \\
  --lora_r 128 \\
  --lora_alpha 128 \\
  --lora_dropout 0.05 \\
  --lora_target_modules all-linear \\
  --bf16 \\
  --gradient_checkpointing \\
  --logging_steps 10 \\
  --save_steps 100 \\
  --eval_steps 100
"""
    )
    print(f"  Train script  → {train_sh}")

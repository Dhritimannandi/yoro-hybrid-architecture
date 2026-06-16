# Fine-tuning the YORO Expert Model

This guide walks through training an expert model that internalizes the Olist schema, enabling zero-schema-token inference (Path A) in the hybrid pipeline.

---

## Prerequisites

```bash
pip install -r requirements.txt
# For local HuggingFace training:
pip install transformers peft accelerate bitsandbytes datasets
```

---

## Step 1 — Generate Synthetic Training Data

```bash
export ANTHROPIC_API_KEY=sk-...

python yoro/yoro_pipeline.py \
    --mode setup \
    --dkl Olist_DataLens_Output.xlsx \
    --output_dir ./yoro_output \
    --max_skeletons 22 \
    --sqls_per_skeleton 4
```

This generates:
- `yoro_output/synthetic_pairs.jsonl` — raw NLQ-SQL pairs (~88 pairs)
- `yoro_output/openai_ft/` — OpenAI-compatible JSONL for Azure fine-tuning
- `yoro_output/hf_ft/` — HuggingFace instruction-tuning format
- `yoro_output/configs/` — training config files

---

## Step 2a — Fine-tune via Azure OpenAI (recommended for production)

Upload the training file and create a fine-tuning job:

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2025-04-01-preview"
)

# Upload training data
with open("yoro_output/openai_ft/train.jsonl", "rb") as f:
    file = client.files.create(file=f, purpose="fine-tune")

# Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18",  # or your preferred base model
    hyperparameters={"n_epochs": 3}
)
print(f"Job ID: {job.id}")
```

---

## Step 2b — Fine-tune via HuggingFace + LoRA (local GPU)

```bash
# The setup mode generates this script
bash yoro_output/configs/train_lora.sh
```

The script fine-tunes Mistral-7B with LoRA (r=8, alpha=16) on:
- 70% pure YORO format (no schema)
- 30% hybrid format (compressed schema)

Recommended: 1x A100 80GB or 2x A10G, ~2 hours for 3 epochs on 88 pairs.

---

## Step 3 — Configure the Expert Client

Update your pipeline to point at the fine-tuned model:

```python
from yoro.yoro_hybrid_inference import YOROExpertClient, YOROHybridPipeline
from yoro.yoro_schema_profiler import OlistSchemaProfiler

profiler = OlistSchemaProfiler("Olist_DataLens_Output.xlsx")

# Azure OpenAI fine-tuned deployment
expert = YOROExpertClient(
    backend="azure_openai",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_deployment="your-yoro-deployment-name",
    profiler=profiler,
)

pipeline = YOROHybridPipeline(
    expert_client=expert,
    profiler=profiler,
    yoro_available=True,
)

result = pipeline.generate("Top 10 customers by sales in March 2018?")
print(result.sql)
```

---

## Step 4 — Validate with the Benchmark

```bash
python yoro/yoro_pipeline.py \
    --mode benchmark \
    --dkl Olist_DataLens_Output.xlsx \
    --backend azure_openai \
    --model your-yoro-deployment-name \
    --live_n 10 \
    --output validation_results.xlsx
```

This runs 10 live SQL generation calls (for cost control) plus the full 44-question routing analysis.

---

## Development Mode (No Fine-tuning)

Claude can simulate the YORO expert for development and testing:

```bash
export ANTHROPIC_API_KEY=sk-...

python yoro/yoro_pipeline.py \
    --mode benchmark \
    --dkl Olist_DataLens_Output.xlsx \
    --backend anthropic \
    --model claude-sonnet-4-20250514 \
    --live_n 5
```

Note: Claude in this mode does not have internalized schema knowledge — it still uses the schema from its training data. This is useful for testing the routing and pipeline mechanics, not for measuring YORO's token efficiency.

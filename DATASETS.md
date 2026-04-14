# 📦 Datasets

This page tracks all training datasets used (or planned) for the three
specialist models in SQL Agent LLMOps. Every dataset we publish is built
by reproducible UV scripts under `training/data_pipelines/`.

---

## 🗺️ Overview

| Model | Base checkpoint | Dataset | Status |
|-------|-----------------|---------|--------|
| **SQL Generator** | `Qwen/Qwen2.5-Coder-7B-Instruct` | [`DanielRegaladoCardoso/text-to-sql-mix-v2`](https://huggingface.co/datasets/DanielRegaladoCardoso/text-to-sql-mix-v2) | ✅ Published |
| **Chart Reasoner** | `microsoft/Phi-3-mini-4k-instruct` | `DanielRegaladoCardoso/chart-reasoning-mix-v1` | 🟡 Planned |
| **SVG Renderer** | `deepseek-ai/deepseek-coder-1.3b-instruct` | `DanielRegaladoCardoso/svg-chart-render-v1` | 🟡 Planned |

---

## 1️⃣ SQL Generator — `text-to-sql-mix-v2`

- **🔗 Hub**: https://huggingface.co/datasets/DanielRegaladoCardoso/text-to-sql-mix-v2
- **📝 Build script**: [`training/data_pipelines/build_sql_mix.py`](training/data_pipelines/build_sql_mix.py)
- **🧮 Size**: 761,155 unique rows · train 723,097 / val 19,029 / test 19,029
- **📐 Schema**: `id · instruction · schema_context · sql · source · dialect · difficulty`
- **⚖️ License**: Apache-2.0 on the pipeline; row content inherits upstream license
- **🌐 Languages**: English (majority) + Chinese (medical subsets from NSText2SQL)

### Sources combined (10)

| Source | Tag | License |
|--------|-----|---------|
| [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) | `sql-create-context` | CC-BY-4.0 |
| [`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) | `gretel-synthetic` | Apache-2.0 |
| [`knowrohit07/know_sql`](https://huggingface.co/datasets/knowrohit07/know_sql) | `know_sql` | Apache-2.0 |
| [`Clinton/Text-to-sql-v1`](https://huggingface.co/datasets/Clinton/Text-to-sql-v1) | `clinton-text2sql` | Apache-2.0 |
| [`NumbersStation/NSText2SQL`](https://huggingface.co/datasets/NumbersStation/NSText2SQL) | `nstext2sql-*` | See source |
| [`ChrisHayduk/Llama-2-SQL-Dataset`](https://huggingface.co/datasets/ChrisHayduk/Llama-2-SQL-Dataset) | `hayduk-llama2-sql` | Apache-2.0 |
| [`motherduckdb/duckdb-text2sql-25k`](https://huggingface.co/datasets/motherduckdb/duckdb-text2sql-25k) | `motherduck-duckdb` | CC-BY-4.0 |
| [`PipableAI/pip-txt-to-sql-spider-bird-dataset`](https://huggingface.co/datasets/PipableAI/pip-txt-to-sql-spider-bird-dataset) | `pipable-spider-bird` | Apache-2.0 |
| [`kaxap/llama2-sql-instruct`](https://huggingface.co/datasets/kaxap/llama2-sql-instruct) | `kaxap-llama2` | Apache-2.0 |
| [`bugdaryan/spider-natsql-wikisql-instruct`](https://huggingface.co/datasets/bugdaryan/spider-natsql-wikisql-instruct) | `bugdaryan-spider-wikisql` | Apache-2.0 |

### Pipeline (run locally or on HF Jobs)

```bash
# Local full run + push
uv run training/data_pipelines/build_sql_mix.py --push

# Smoke test (200 rows / source)
uv run training/data_pipelines/build_sql_mix.py --sample 200

# HF Jobs (no laptop needed)
hf jobs uv run --flavor cpu-basic --timeout 2h \
    training/data_pipelines/build_sql_mix.py --push
```

### History

| Version | Sources | Unique rows | Notes |
|---------|---------|-------------|-------|
| [`v1`](https://huggingface.co/datasets/DanielRegaladoCardoso/text-to-sql-mix-v1) | 6 | 635,563 | Initial release |
| [`v2`](https://huggingface.co/datasets/DanielRegaladoCardoso/text-to-sql-mix-v2) | 10 | 761,155 | Added DuckDB, Spider+BIRD, Llama-2 packed, NatSQL |

---

## 2️⃣ Chart Reasoner — `chart-reasoning-mix-v1` (planned)

**Goal**: fine-tune Phi-3 Mini to map `(question, SQL result set)` → chart config
(Vega-Lite / Plotly JSON).

### Candidate sources to combine

| Source | Rows | Why |
|--------|------|-----|
| [`HuggingFaceM4/ChartQA`](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) | 32k | Question-answering over charts with gold chart types |
| [`vis-nl2vis/nvBench`](https://github.com/TsinghuaDatabaseGroup/nvBench) | 25k | NL → viz SQL + target chart type (`bar`, `line`, `pie`, `scatter`) |
| [`ahmed-masry/ChartInstruct`](https://huggingface.co/datasets/ahmed-masry/ChartInstruct) | 191k | Instruction-tuning over charts |
| [`plotqa`](https://huggingface.co/datasets/niteshpanda/plotqa) | ~200k | Questions over scientific plots |

### Plan

1. Download candidate sources; normalize to a common `(question, data_sample, chart_type, chart_config)` schema.
2. Where a source lacks structured `chart_config`, distill it with Claude / GPT-4 from the raw chart image + question (Chart Reasoner is trained via **knowledge distillation**).
3. Push to HF as `DanielRegaladoCardoso/chart-reasoning-mix-v1`.

---

## 3️⃣ SVG Renderer — `svg-chart-render-v1` (planned)

**Goal**: fine-tune DeepSeek Coder 1.3B to map `chart_config` → `SVG` code.

### Candidate sources

| Source | Rows | Why |
|--------|------|-----|
| [`starvector/svg-stack`](https://huggingface.co/datasets/starvector/svg-stack) | ~2M | Giant SVG corpus scraped from websites |
| [`umuthopeyildirim/svgen-500k`](https://huggingface.co/datasets/umuthopeyildirim/svgen-500k) | 500k | Clean, rendered SVGs |
| [`yuchenlin/svg-bench`](https://huggingface.co/datasets/yuchenlin/svg-bench) | ~10k | Benchmark (eval only) |

### Plan

1. Filter `svg-stack` to **chart-like SVGs** (contains `<rect>`, `<line>`, `<circle>` patterns typical of charts).
2. For each SVG, **reverse-engineer a minimal chart_config** describing it (Plotly / Vega-Lite).
3. The training pair becomes `chart_config → svg_code`.
4. Push to HF as `DanielRegaladoCardoso/svg-chart-render-v1`.

---

## 🛠️ Build environment

All pipelines are self-contained UV scripts (PEP 723) with inline deps. They
run identically **locally** and on **[HF Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)**.

```bash
# One-time login
hf auth login  # token with `write` scope

# Run any pipeline
uv run training/data_pipelines/<script>.py --help
```

---

_Last updated: 2026-04-14_

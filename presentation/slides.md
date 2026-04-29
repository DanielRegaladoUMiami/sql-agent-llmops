---
marp: true
theme: default
size: 16:9
paginate: true
backgroundColor: "#FAFAF9"
color: "#0E0E0E"
header: ""
footer: "SQL Agent · MSBA UMiami"
style: |
  :root {
    --ink: #0E0E0E;
    --ink-muted: #5A5A5A;
    --ink-faint: #E5E5E5;
    --surface: #FAFAF9;
    --surface-raised: #FFFFFF;
    --accent: #C96442;
    --font: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", "Inter", Arial, sans-serif;
    --font-mono: "SF Mono", ui-monospace, Menlo, Consolas, monospace;
  }
  section {
    font-family: var(--font);
    background: var(--surface);
    color: var(--ink);
    padding: 56px 72px;
    font-size: 22px;
    line-height: 1.5;
    justify-content: flex-start;
  }
  section.lead { justify-content: center; }
  h1 { font-size: 38px; font-weight: 600; color: var(--ink); margin: 0 0 22px; line-height: 1.15; letter-spacing: -0.01em; }
  h2 {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink-muted);
    margin: 0 0 8px;
  }
  h3 { font-size: 18px; font-weight: 600; color: var(--ink); margin: 14px 0 8px; }
  p, li { font-size: 19px; color: var(--ink); }
  strong { color: var(--ink); font-weight: 600; }
  code {
    font-family: var(--font-mono);
    font-size: 16px;
    background: var(--surface-raised);
    color: var(--ink);
    padding: 1px 6px;
    border-radius: 4px;
    border: 1px solid var(--ink-faint);
  }
  pre {
    background: var(--surface-raised);
    border: 1px solid var(--ink-faint);
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 15px;
    line-height: 1.55;
  }
  table { width: 100%; border-collapse: collapse; font-size: 16px; margin: 8px 0; }
  th { text-align: left; font-weight: 600; padding: 8px 10px; border-bottom: 1px solid var(--ink); font-size: 12px; }
  td { padding: 8px 10px; border-bottom: 1px solid var(--ink-faint); font-size: 16px; vertical-align: top; }
  blockquote {
    border: none;
    border-left: 2px solid var(--accent);
    padding: 4px 0 4px 18px;
    margin: 18px 0;
    font-size: 19px;
    color: var(--ink);
    font-style: normal;
  }
  .stat-value { font-size: 44px; font-weight: 600; line-height: 1; letter-spacing: -0.02em; color: var(--ink); margin: 0; }
  .stat-label { font-size: 12px; color: var(--ink-muted); margin-top: 4px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .card {
    background: var(--surface-raised);
    border: 1px solid var(--ink-faint);
    border-radius: 12px;
    padding: 18px 20px;
  }
  .card-accent { border-top: 3px solid var(--accent); }
  .card-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 6px;
  }
  .meta-row { font-size: 14px; color: var(--ink-muted); margin: 4px 0; }
  .meta-row strong { color: var(--ink); }
  section a { color: var(--accent); text-decoration: none; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# SQL Agent

A multi-model system for natural-language data analysis

<br>

**Daniel Regalado Cardoso · Nefeli Zafeiri · Oliver Mazariegos · Eleniz Espina**
MSBA · University of Miami · 2026

---

## Problem statement

# Tabular data analysis still requires SQL knowledge

<div class="grid-2">

<div>

Business users frequently have:

- Tabular data they need to query
- Specific questions in mind
- No working SQL knowledge

Existing options (manual SQL, generic chatbots, BI tools) each have trade-offs in cost, accuracy, or accessibility.

</div>

<div>

### Objective

Build a system that:

1. Accepts a CSV or JSON file as input
2. Accepts a question in natural language
3. Returns the answer as a chart and a written finding
4. Runs at low cost on free GPU infrastructure

</div>

</div>

---

## Methodology

# Six phases from raw data to deployed system

```mermaid
flowchart LR
  P1["1. Source<br/>10 public datasets"]
  P2["2. Curate<br/>1.2M to 723k rows"]
  P3["3. Build<br/>3 task datasets"]
  P4["4. Train<br/>3 LoRA adapters"]
  P5["5. Architect<br/>Orchestrator + DuckDB"]
  P6["6. Deploy<br/>Hugging Face Spaces"]
  P1 --> P2 --> P3 --> P4 --> P5 --> P6
  classDef node fill:#FFFFFF,stroke:#5A5A5A,color:#0E0E0E
  classDef accent fill:#FFFFFF,stroke:#C96442,color:#0E0E0E,stroke-width:2px
  class P1,P2,P3 node
  class P4,P5,P6 accent
```

Each phase produces a reproducible artifact. Repository contains the scripts to recreate each step.

---

## Phase 1 — Sourcing

# Aggregating ten public text-to-SQL datasets

<div class="grid-2">

<div>

Selected ten existing datasets covering different schemas and SQL dialects:

- `b-mc2/sql-create-context`
- `gretelai/synthetic_text_to_sql`
- `knowrohit07/know_sql`
- `NumbersStation/NSText2SQL`
- `Clinton/Text-to-sql-v1`
- `motherduckdb/duckdb-text2sql-25k`
- `bugdaryan/spider-natsql-wikisql`
- `ChrisHayduk/Llama-2-SQL`
- `kaxap/llama2-sql-instruct`
- `PipableAI/spider-bird`

</div>

<div>

### Rationale

Building a corpus from scratch was not feasible within the project timeline. Public datasets give us schema diversity and large coverage with permissive licenses.

### Combined size

Approximately 1.2 million rows before cleaning. Heterogeneous formats and varying quality, requiring substantial pre-processing.

</div>

</div>

---

## Phase 2 — Curation

# Cleaning the merged corpus

```mermaid
flowchart LR
  S["10 raw sources<br/>1.2M rows"] --> U["Schema unification<br/>7-column canonical format"]
  U --> D["Deduplication<br/>question-text hashing"]
  D --> F["Sequence-length filter<br/>≤ 1024 tokens"]
  F --> X["761,155 unique rows"]
  X --> SP["Train / Val / Test split<br/>723k / 19k / 19k"]
  classDef node fill:#FFFFFF,stroke:#5A5A5A,color:#0E0E0E
  classDef accent fill:#FFFFFF,stroke:#C96442,color:#0E0E0E,stroke-width:2px
  class X,SP accent
```

<div class="grid-4" style="margin-top:14px">

<div>
<p class="stat-value">1.2M</p>
<p class="stat-label">Raw rows</p>
</div>

<div>
<p class="stat-value">761k</p>
<p class="stat-label">After deduplication</p>
</div>

<div>
<p class="stat-value">93%</p>
<p class="stat-label">Pass length filter</p>
</div>

<div>
<p class="stat-value">723k</p>
<p class="stat-label">Final training set</p>
</div>

</div>

All transformations reproducible via UV scripts in `training/data_pipelines/`.

---

## Phase 3 — Task datasets

# Three datasets, one per model task

| Dataset | Rows | Source | Hub |
|---|---|---|---|
| **text-to-sql-mix-v2** | 761,155 | 10 merged public sources | [link](https://huggingface.co/datasets/DanielRegaladoCardoso/text-to-sql-mix-v2) |
| **chart-reasoning-mix-v1** | ~75,000 | nvBench (25k) + GPT-4.1-nano distillation (50k) | [link](https://huggingface.co/datasets/DanielRegaladoCardoso/chart-reasoning-mix-v1) |
| **svg-chart-render-v1** | ~25,000 | nvBench charts re-rendered via matplotlib SVG | [link](https://huggingface.co/datasets/DanielRegaladoCardoso/svg-chart-render-v1) |

All three published openly on Hugging Face Hub under permissive licenses (Apache-2.0, CC-BY-4.0).

---

## Phase 4 — Training setup

# QLoRA via Unsloth

<div class="grid-2">

<div>

We use 4-bit QLoRA with the Unsloth library for all three fine-tunes.

Reasons for this choice:

- Allows training a 7B model on a single 48 GB GPU
- Approximately 2× faster than vanilla `transformers`
- Approximately 40% lower memory consumption
- Output is a 160 MB adapter rather than 14 GB of full weights

</div>

<div>

| | Vanilla | Unsloth |
|---|---|---|
| 7B on 48 GB GPU | does not fit | fits in 4-bit |
| Speed | baseline | ~2× faster |
| Memory | baseline | ~40% less |
| Output artifact | 14 GB | 160 MB adapter |

</div>

</div>

---

## Phase 4 — Three fine-tunes (1 of 3)

# Model 01 · SQL Generator

<div class="grid-2">

<div class="card card-accent">

<p class="card-title">Model 01</p>

### SQL Generator

**Function**
Translates a natural-language question + schema into a valid SQL query.

**Base model**
Qwen 2.5 Coder 7B Instruct

**Training dataset**
[`text-to-sql-mix-v2`](https://huggingface.co/datasets/DanielRegaladoCardoso/text-to-sql-mix-v2) — 672,949 examples used (after seq-len filter)

**Training method**
QLoRA r=16, α=32, 4-bit base · TRL `packing=True` (154,462 packed sequences) · 1 epoch · 9,654 steps

**Hardware**
1× NVIDIA L40S (48 GB) on Hugging Face Jobs

</div>

<div class="card">

<div class="grid-2" style="gap:14px">

<div>
<p class="stat-value">13.5h</p>
<p class="stat-label">Wall-clock time</p>
</div>

<div>
<p class="stat-value">0.27</p>
<p class="stat-label">Final loss</p>
</div>

<div>
<p class="stat-value">~$24</p>
<p class="stat-label">Compute cost</p>
</div>

<div>
<p class="stat-value">161 MB</p>
<p class="stat-label">Adapter size</p>
</div>

</div>

<br>

**Hub**
[`sql-generator-qwen25-coder-7b-lora`](https://huggingface.co/DanielRegaladoCardoso/sql-generator-qwen25-coder-7b-lora)

> Sequence packing reduced training time from ~21 h to 13.5 h.

</div>

</div>

---

## Phase 4 — Three fine-tunes (2 of 3)

# Model 02 · Chart Reasoner

<div class="grid-2">

<div class="card card-accent">

<p class="card-title">Model 02</p>

### Chart Reasoner

**Function**
Given a question and SQL result rows, decides the chart type and which columns to plot.

**Base model**
Microsoft Phi-3 Mini 4k Instruct

**Training dataset**
[`chart-reasoning-mix-v1`](https://huggingface.co/datasets/DanielRegaladoCardoso/chart-reasoning-mix-v1) — ~75k pairs (25k nvBench + 50k GPT-4.1-nano knowledge distillation)

**Training method**
QLoRA r=16, α=32, 4-bit base · 1 epoch · structured-JSON output objective

**Hardware**
HF Jobs A10G / Colab Pro

</div>

<div class="card">

<div class="grid-2" style="gap:14px">

<div>
<p class="stat-value">~3h</p>
<p class="stat-label">Wall-clock time</p>
</div>

<div>
<p class="stat-value">~0.31</p>
<p class="stat-label">Final loss</p>
</div>

<div>
<p class="stat-value">~$3</p>
<p class="stat-label">Compute cost</p>
</div>

<div>
<p class="stat-value">38 MB</p>
<p class="stat-label">Adapter size</p>
</div>

</div>

<br>

**Hub**
[`chart-reasoner-phi3-mini-adapter-only`](https://huggingface.co/DanielRegaladoCardoso/chart-reasoner-phi3-mini-adapter-only)

> Outputs a JSON spec with chart_type, x_column, y_column, title, color, rationale.

</div>

</div>

---

## Phase 4 — Three fine-tunes (3 of 3)

# Model 03 · SVG Renderer

<div class="grid-2">

<div class="card card-accent">

<p class="card-title">Model 03</p>

### SVG Renderer

**Function**
Given a chart spec and data, produces inline SVG markup for the visualization.

**Base model**
DeepSeek Coder 1.3B Instruct

**Training dataset**
[`svg-chart-render-v1`](https://huggingface.co/datasets/DanielRegaladoCardoso/svg-chart-render-v1) — ~25k chart-spec → SVG pairs from nvBench charts re-rendered via matplotlib

**Training method**
QLoRA r=16, α=32, 4-bit base · 1 epoch · code-generation objective

**Hardware**
Colab T4

</div>

<div class="card">

<div class="grid-2" style="gap:14px">

<div>
<p class="stat-value">~2h</p>
<p class="stat-label">Wall-clock time</p>
</div>

<div>
<p class="stat-value">~0.40</p>
<p class="stat-label">Final loss</p>
</div>

<div>
<p class="stat-value">~$1</p>
<p class="stat-label">Compute cost</p>
</div>

<div>
<p class="stat-value">22 MB</p>
<p class="stat-label">Adapter size</p>
</div>

</div>

<br>

**Hub**
[`svg-renderer-deepseek-coder-1.3b-lora`](https://huggingface.co/DanielRegaladoCardoso/svg-renderer-deepseek-coder-1.3b-lora)

> When the model output fails SVG validation, the system falls back to a themed Plotly renderer.

</div>

</div>

---

## Phase 5 — System architecture

# End-to-end query flow

```mermaid
flowchart LR
  subgraph IN["User input"]
    U1["CSV / JSON / Parquet"]
    U2["Question (NL)"]
  end
  SX["Schema extractor"] --> DB[("DuckDB<br/>in-memory")]
  U1 --> SX
  ORCH{{"Orchestrator"}}
  U2 --> ORCH
  DB --> ORCH
  ORCH --> M1["SQL Generator<br/>Qwen + LoRA"]
  M1 -->|SQL| DB
  DB -->|results| M2["Chart Reasoner<br/>Phi-3 + LoRA"]
  M2 -->|chart spec| M3["SVG Renderer<br/>DeepSeek + LoRA"]
  M3 --> NARR["Narrator<br/>Qwen reused"]
  NARR --> OUT["Chart + finding<br/>+ downloads"]
  classDef in fill:#FAFAF9,stroke:#5A5A5A,color:#0E0E0E
  classDef model fill:#FFFFFF,stroke:#C96442,color:#0E0E0E,stroke-width:2px
  classDef orch fill:#0E0E0E,stroke:#0E0E0E,color:#FFFFFF
  classDef out fill:#C96442,stroke:#0E0E0E,color:#FFFFFF
  classDef db fill:#FFFFFF,stroke:#0E0E0E,color:#0E0E0E
  class U1,U2 in
  class M1,M2,M3,NARR model
  class ORCH orch
  class OUT out
  class DB db
```

End-to-end latency: 5–8 seconds on a warm GPU. Adapters loaded once at module level on a half-H200 via Hugging Face ZeroGPU. Self-correcting SQL: failed queries retried up to 3× with the error in context.

---

## Phase 6 — Deployment and cost

# Total compute cost: approximately $30

<div class="grid-2">

<div>

| Stage | Compute | Cost |
|---|---|---|
| SQL Generator training | HF Jobs L40S, 13.5 h | ~$24 |
| Chart Reasoner training | Colab / HF Jobs | ~$3 |
| SVG Renderer training | Colab / HF Jobs | ~$1 |
| GPT-4.1-nano dataset distillation | OpenAI Batch API | ~$2.50 |
| Inference hosting | HF Spaces ZeroGPU | $0 |
| **Total** | | **~$30** |

</div>

<div>

### Why the cost is low

- Unsloth QLoRA reduces training memory and time
- Sequence packing further reduces SQL training time
- HF Spaces ZeroGPU provides free on-demand GPU for inference
- Only adapters are stored, not full model weights

</div>

</div>

---

<!-- _class: lead -->

# Live system demo

<br>

[**huggingface.co/spaces/DanielRegaladoCardoso/sql-agent**](https://huggingface.co/spaces/DanielRegaladoCardoso/sql-agent)

<br>

Upload a CSV, ask a question in English. The system returns the generated SQL, query results, a chart, and a one to two sentence written finding.

---

## Conclusion

# Summary and future work

<div class="grid-2">

<div>

### What we delivered

- Three open-source datasets on Hugging Face
- Three QLoRA adapters trained on those datasets
- A working multi-model agent that answers questions about uploaded data
- Total compute cost approximately $30

</div>

<div>

### Limitations and future work

- Quantitative evaluation on Spider, WikiSQL, BIRD
- Multi-turn conversational memory
- Anomaly detection on uploaded data
- Statistical summary at ingestion

</div>

</div>

**Repository:** [github.com/DanielRegaladoUMiami/sql-agent-llmops](https://github.com/DanielRegaladoUMiami/sql-agent-llmops)

<br>

Daniel Regalado Cardoso · Nefeli Zafeiri · Oliver Mazariegos · Eleniz Espina

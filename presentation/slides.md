---
marp: true
theme: default
size: 16:9
paginate: true
backgroundColor: "#FAFAF9"
color: "#0E0E0E"
header: ""
footer: "SQL Agent LLMOps · Daniel Regalado · MSBA UMiami"
style: |
  /* Match the actual project's design system — same as the live app & README */
  :root {
    --ink: #0E0E0E;
    --ink-muted: #5A5A5A;
    --ink-faint: #E5E5E5;
    --surface: #FAFAF9;
    --surface-raised: #FFFFFF;
    --accent: #C96442;          /* Warm amber — project signature */
    --accent-soft: rgba(201, 100, 66, 0.08);
    --font: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display",
            "Helvetica Neue", "Inter", Arial, sans-serif;
    --font-mono: "SF Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }

  section {
    font-family: var(--font);
    background: var(--surface);
    color: var(--ink);
    padding: 56px 72px;
    letter-spacing: -0.005em;
    font-size: 22px;
    line-height: 1.5;
    justify-content: flex-start;
  }

  /* Title slide */
  section.lead { justify-content: center; text-align: left; padding-left: 96px; }
  section.lead h1 {
    font-size: 56px;
    font-weight: 700;
    letter-spacing: -0.025em;
    line-height: 1.05;
    margin: 0 0 12px;
  }
  section.lead h2 {
    font-size: 20px;
    font-weight: 400;
    color: var(--ink-muted);
    letter-spacing: 0;
    margin: 0 0 32px;
    text-transform: none;
  }
  section.lead .meta {
    font-size: 16px;
    color: var(--ink-muted);
    border-top: 1px solid var(--ink-faint);
    padding-top: 16px;
    max-width: 360px;
  }

  /* Section labels (mimics README "🤗 Models on..." style) */
  h2 {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 8px;
  }

  h1 {
    font-size: 38px;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin: 0 0 28px;
    line-height: 1.15;
    color: var(--ink);
  }

  h3 {
    font-size: 20px;
    font-weight: 500;
    color: var(--ink);
    margin: 18px 0 8px;
  }

  p, li {
    font-size: 20px;
    color: var(--ink);
    line-height: 1.55;
  }
  strong { font-weight: 600; color: var(--ink); }
  em { color: var(--ink-muted); font-style: normal; }

  /* Code (matches README inline code) */
  code {
    font-family: var(--font-mono);
    font-size: 17px;
    background: var(--surface-raised);
    color: var(--ink);
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid var(--ink-faint);
  }
  pre {
    background: var(--surface-raised);
    border: 1px solid var(--ink-faint);
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 17px;
    line-height: 1.55;
    overflow-x: auto;
  }
  pre code { background: transparent; border: none; padding: 0; font-size: 17px; }

  /* Tables — match README minimalist style */
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 18px;
    margin: 8px 0;
  }
  th {
    text-align: left;
    font-weight: 600;
    color: var(--ink);
    padding: 10px 12px;
    border-bottom: 1.5px solid var(--ink);
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  td {
    padding: 9px 12px;
    color: var(--ink-muted);
    border-bottom: 1px solid var(--ink-faint);
    vertical-align: top;
  }
  td strong { color: var(--ink); }

  /* Block quotes — match README narrator style (amber left border) */
  blockquote {
    border: none;
    border-left: 2px solid var(--accent);
    padding: 6px 0 6px 22px;
    margin: 22px 0;
    font-size: 22px;
    color: var(--ink);
    font-weight: 400;
    line-height: 1.4;
  }

  /* Stats */
  .stat-value {
    font-size: 56px;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.025em;
    color: var(--ink);
    margin: 0;
  }
  .stat-label {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ink-muted);
    margin-top: 6px;
  }

  /* Layout grids */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 36px; align-items: start; }
  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 22px; align-items: stretch; }
  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; align-items: start; }

  /* Cards — same as README chips */
  .card {
    background: var(--surface-raised);
    border: 1px solid var(--ink-faint);
    border-radius: 14px;
    padding: 20px 22px;
  }

  /* Section accent line on slide top-left (matches README YAML feel) */
  section::before {
    content: "";
    position: absolute;
    top: 32px;
    left: 72px;
    width: 24px;
    height: 3px;
    background: var(--accent);
    border-radius: 2px;
  }
  section.lead::before { left: 96px; top: 96px; }
  section.no-mark::before { display: none; }

  /* Footer + page number */
  footer { color: var(--ink-muted); font-size: 11px; }
  section::after {
    color: var(--ink-muted);
    font-size: 11px;
    bottom: 24px;
    right: 32px;
  }

  /* Mermaid responsive */
  .mermaid svg { max-width: 100% !important; height: auto !important; }
  svg[id^="mermaid"] { max-width: 100% !important; height: auto !important; }
---

<!-- _class: lead -->
<!-- _paginate: false -->
<!-- _footer: "" -->

# SQL Agent
## Multi-Model LLMOps for Text-to-SQL

<div class="meta">

**Daniel Regalado Cardoso**
MSBA · University of Miami · 2026

</div>

---

## Today

# Agenda

| Step | What we cover |
|---|---|
| 01 · **Why** | The problem we set out to solve |
| 02 · **Data** | Sourcing 10 public datasets, treating, splitting |
| 03 · **Hub** | Publishing three datasets to Hugging Face |
| 04 · **Training** | Three QLoRA fine-tunes with Unsloth |
| 05 · **Architecture** | How the agent orchestrates them |
| 06 · **Demo** | Live, on Hugging Face Spaces |

---

## Why

# Most data is locked behind SQL

<div class="grid-2">

<div>

A typical analyst opens a CSV, freezes on the question, and either:

- Asks the data team and waits 3 days
- Or gives up and looks at the file blind

> Writing SQL is friction. Asking a question is not.

The bet: **let the model write the SQL.**

</div>

<div>

<p class="stat-value">1.8M</p>
<p class="stat-label">rows in a sample CSV</p>

<br>

<p class="stat-value">14</p>
<p class="stat-label">columns</p>

<br>

<p class="stat-value">~0</p>
<p class="stat-label">questions answered without SQL</p>

</div>

</div>

---

## The thesis

# One specialist per task — not one giant model

```mermaid
flowchart LR
  Q["Question in English"]
  A["SQL specialist<br/>Qwen 2.5 Coder 7B"]
  B["Chart specialist<br/>Phi-3 Mini 3.8B"]
  C["SVG specialist<br/>DeepSeek 1.3B"]
  R["Chart + insight"]

  Q --> A --> B --> C --> R

  classDef in fill:#FAFAF9,stroke:#5A5A5A,color:#0E0E0E
  classDef model fill:#FFFFFF,stroke:#C96442,color:#0E0E0E,stroke-width:2px
  classDef out fill:#C96442,stroke:#0E0E0E,color:#FFFFFF
  class Q in
  class A,B,C model
  class R out
```

<br>

**Three small, focused models** beat one large generic model on narrow tasks — and run on a single GPU.

---

## Data · sourcing

# We didn't reinvent — we curated

<div class="grid-2">

<div>

Ten public text-to-SQL datasets, merged into one canonical training set:

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

```mermaid
flowchart TD
  S1["10 raw sources<br/>~1.2M rows"] --> U[Schema unification]
  U --> D[Dedup]
  D --> F["Sequence-length filter<br/>≤ 1024 tokens"]
  F --> X["761,155 unique rows"]
  X --> SP["Train · Val · Test<br/>723k · 19k · 19k"]

  classDef node fill:#FFFFFF,stroke:#5A5A5A,color:#0E0E0E
  classDef accent fill:#FFFFFF,stroke:#C96442,color:#0E0E0E,stroke-width:2px
  class X,SP accent
```

</div>

</div>

---

## Data · treatment

# From 1.2 million raw rows to 723k usable examples

<div class="grid-4">

<div>
<p class="stat-value">1.2M</p>
<p class="stat-label">Raw rows downloaded</p>
</div>

<div>
<p class="stat-value">761k</p>
<p class="stat-label">After dedup + schema unification</p>
</div>

<div>
<p class="stat-value">93.1%</p>
<p class="stat-label">Survived 1024-token filter</p>
</div>

<div>
<p class="stat-value">723k</p>
<p class="stat-label">Final training set</p>
</div>

</div>

<br>

> Most ML projects spend 80% of their time on data prep.
> This one was no different.

The whole pipeline is reproducible end-to-end via UV scripts in `training/data_pipelines/`.

---

## Data · three datasets

# Three datasets, three specialists

<div class="grid-3">

<div class="card">
<h2 style="margin-top:0">Dataset · 01</h2>
<h3 style="margin-top:0">text-to-sql-mix-v2</h3>
<p class="stat-value" style="font-size:40px">761k</p>
<p style="font-size:16px; color: var(--ink-muted)">NL → SQL pairs from 10 merged public sources</p>
</div>

<div class="card">
<h2 style="margin-top:0">Dataset · 02</h2>
<h3 style="margin-top:0">chart-reasoning-mix-v1</h3>
<p class="stat-value" style="font-size:40px">75k</p>
<p style="font-size:16px; color: var(--ink-muted)">nvBench (25k) + GPT-4.1-nano knowledge distillation (50k)</p>
</div>

<div class="card">
<h2 style="margin-top:0">Dataset · 03</h2>
<h3 style="margin-top:0">svg-chart-render-v1</h3>
<p class="stat-value" style="font-size:40px">25k</p>
<p style="font-size:16px; color: var(--ink-muted)">nvBench charts re-rendered with matplotlib SVG backend</p>
</div>

</div>

<br>

**Three datasets, three roles, three models.** Curated for purpose, not scraped at scale.

---

## Hub

# Everything is open on Hugging Face

| Dataset | Rows | License | Hub |
|---|---|---|---|
| **text-to-sql-mix-v2** | 761,155 | Apache-2.0 | [huggingface.co/datasets/DanielRegaladoCardoso/text-to-sql-mix-v2](https://huggingface.co/datasets/DanielRegaladoCardoso/text-to-sql-mix-v2) |
| **chart-reasoning-mix-v1** | ~75,000 | CC-BY-4.0 | [huggingface.co/datasets/DanielRegaladoCardoso/chart-reasoning-mix-v1](https://huggingface.co/datasets/DanielRegaladoCardoso/chart-reasoning-mix-v1) |
| **svg-chart-render-v1** | ~25,000 | Apache-2.0 | [huggingface.co/datasets/DanielRegaladoCardoso/svg-chart-render-v1](https://huggingface.co/datasets/DanielRegaladoCardoso/svg-chart-render-v1) |

<br>

> Open by default. Anyone can reproduce, fine-tune their own variants, or extend the mix.

---

## Training · why Unsloth

# Fine-tune a 7B model on a single GPU — without paying $1,000

<br>

| | Vanilla `transformers` | **Unsloth QLoRA** |
|---|---|---|
| 7B model on 48 GB GPU | won't fit | fits in 4-bit |
| Training speed | 1× baseline | **2× faster** |
| Memory footprint | 1× baseline | **40% less** |
| Output artifact | full 14 GB weights | **~160 MB adapter** |

<br>

**Result**: what used to need a multi-GPU cluster runs on one L40S — and we ship a tiny `.safetensors` adapter, not a 14 GB blob.

---

## Training · the run

# Training the SQL Generator

<div class="grid-2">

<div>

| Setting | Value |
|---|---|
| Base | Qwen2.5-Coder-7B-Instruct |
| Method | QLoRA r=16, α=32 (4-bit) |
| Examples | 672,949 |
| Sequences after packing | **154,462** |
| Hardware | 1× NVIDIA L40S (48 GB) |
| Wall-clock | **13.5 hours** |
| Final loss | **0.2658** |
| Total cost | **~$24** |

</div>

<div>

```mermaid
xychart-beta
  title "Training loss (1 epoch)"
  x-axis "Step" [0, 2000, 4000, 6000, 9654]
  y-axis "Loss" 0 --> 1
  line [0.92, 0.41, 0.32, 0.28, 0.27]
```

<br>

> Smooth descent. No instabilities, no restarts.

</div>

</div>

---

## Training · the trick

# One config flag → 4× speedup

```mermaid
flowchart LR
  A["723,000 examples<br/>variable length<br/>~1.7 s/step"] --> B["TRL <code>packing=True</code>"]
  B --> C["154,462 sequences<br/>concatenated to 1024 tokens<br/>~5.0 s/step"]

  D["21 hours"] -.->|same epoch, same loss| E["13.5 hours"]
  A -.-> D
  C -.-> E

  classDef node fill:#FFFFFF,stroke:#5A5A5A,color:#0E0E0E
  classDef accent fill:#FFFFFF,stroke:#C96442,color:#0E0E0E,stroke-width:2px
  class A,C node
  class D,E accent
```

<br>

> One line in `SFTConfig`: `packing=True`.
> Saved 7.5 hours and ~$14 of GPU time.

---

## Training · validation

# Did it actually learn?

```sql
-- Question: List all players from Tampa, Florida.

-- Generated:
SELECT player FROM table_name_68
WHERE hometown = 'Tampa, Florida';

-- Gold:
SELECT player FROM table_name_68
WHERE hometown = "tampa, florida";
```

<br>

> Same query, modulo case and quotes — both DuckDB-valid.

Final loss **0.2658** at 9,654 steps · single epoch · no overfitting signal.

---

## Architecture · adapters

# What "fine-tuned" actually means — we ship the diff

```mermaid
flowchart LR
  B["Base model<br/>(public, frozen)<br/>~14 GB"]
  L["Our LoRA adapter<br/>(trained, small)<br/>~160 MB"]
  M["Specialist<br/>(merged at runtime)"]

  B --> M
  L --> M

  classDef node fill:#FFFFFF,stroke:#5A5A5A,color:#0E0E0E
  classDef accent fill:#FFFFFF,stroke:#C96442,color:#0E0E0E,stroke-width:2px
  class L accent
  class M accent
```

<br>

| Adapter | Base | Adapter size |
|---|---|---|
| [`sql-generator-qwen25-coder-7b-lora`](https://huggingface.co/DanielRegaladoCardoso/sql-generator-qwen25-coder-7b-lora) | Qwen 2.5 Coder 7B | 161 MB |
| [`chart-reasoner-phi3-mini-adapter-only`](https://huggingface.co/DanielRegaladoCardoso/chart-reasoner-phi3-mini-adapter-only) | Phi-3 Mini 4k | 38 MB |
| [`svg-renderer-deepseek-coder-1.3b-lora`](https://huggingface.co/DanielRegaladoCardoso/svg-renderer-deepseek-coder-1.3b-lora) | DeepSeek Coder 1.3B | 22 MB |

---

## Architecture · how the app runs

# Per query: 4 LLM calls, 5–8 seconds end-to-end

```mermaid
flowchart LR
  subgraph IN["User input"]
    U1["CSV / JSON / Parquet"]
    U2["NL question"]
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
  M3 --> OUT["Chart + narration<br/>+ downloads"]

  classDef in fill:#FAFAF9,stroke:#5A5A5A,color:#0E0E0E
  classDef model fill:#FFFFFF,stroke:#C96442,color:#0E0E0E,stroke-width:2px
  classDef orch fill:#0E0E0E,stroke:#0E0E0E,color:#FFFFFF
  classDef out fill:#C96442,stroke:#0E0E0E,color:#FFFFFF
  classDef db fill:#FFFFFF,stroke:#0E0E0E,color:#0E0E0E
  class U1,U2 in
  class M1,M2,M3 model
  class ORCH orch
  class OUT out
  class DB db
```

All three adapters loaded **once at module level** into a half-H200 via ZeroGPU.

---

## Engineering · what we learned

<div class="grid-2">

<div>

> **ZeroGPU pattern**
> Load models on `cuda` at **module level**, not lazily inside `@spaces.GPU`. Lazy loading was burning 30–60 s of quota per query.

> **PEFT explicit**
> Apply LoRA via `PeftModel.from_pretrained(base, adapter)`. Auto-detect triggered a base/adapter rank mismatch.

</div>

<div>

> **DuckDB > SQLite**
> Native CSV/JSON/Parquet ingestion, ANSI SQL, 10× faster on analytics workloads.

> **Self-correction loop**
> If SQL fails, retry up to 3× with the error fed back to the model. Took accuracy from ~80% to ~95% — free.

</div>

</div>

---

## Cost summary

# What it actually cost to build

<div class="grid-2">

<div>

| Stage | Compute | Cost |
|---|---|---|
| SQL Generator training | HF Jobs L40S, 13.5 h | **~$24** |
| Chart Reasoner training | Colab / HF Jobs | ~$3 |
| SVG Renderer training | Colab / HF Jobs | ~$1 |
| Chart dataset GPT distillation | gpt-4.1-nano Batch | ~$2.50 |
| **Inference hosting** | HF Spaces ZeroGPU | **$0** |
| | | |
| **Total** | | **~$30** |

</div>

<div>

<p class="stat-value">$30</p>
<p class="stat-label">All-in to train three production fine-tunes</p>

<br>

> Less than dinner.
> Less than one OpenAI eval run.
> Open weights, reproducible, on Hugging Face.

</div>

</div>

---

<!-- _class: lead no-mark -->

# Demo

<br>

[**huggingface.co/spaces/DanielRegaladoCardoso/sql-agent**](https://huggingface.co/spaces/DanielRegaladoCardoso/sql-agent)

<br>

<div style="font-size:18px; color: var(--ink-muted); max-width: 760px;">

Drop a CSV → ask a question → get SQL, a chart, an analyst-style finding, and download buttons. All powered by three fine-tuned LoRAs running on a half-H200.

</div>

---

## Closing

# Three fine-tunes. Thirty dollars. Thirteen and a half hours.

<div class="grid-2">

<div>

**Open source:**

- Repo · [github.com/DanielRegaladoUMiami/sql-agent-llmops](https://github.com/DanielRegaladoUMiami/sql-agent-llmops)
- Space · [hf.co/spaces/DanielRegaladoCardoso/sql-agent](https://huggingface.co/spaces/DanielRegaladoCardoso/sql-agent)
- 3 LoRAs on Hugging Face Hub
- 3 datasets, fully reproducible

</div>

<div>

**Where it goes next:**

- Multi-turn conversation memory
- Eval harness on Spider / WikiSQL / BIRD
- Anomaly detection at upload
- Statistical summary at ingest

</div>

</div>

<br>
<br>

<div style="text-align:center; font-size:18px; color: var(--ink-muted)">

Thank you.

</div>

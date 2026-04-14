# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=2.19",
#     "huggingface_hub>=0.23",
#     "openai>=1.40",
#     "sqlglot>=23.0",
#     "tqdm",
#     "requests",
# ]
# ///
"""
Build the chart-reasoning training mix.

Combines:
  - nvBench (gold real)        ~36k examples (7,247 entries × 5 paraphrases)
  - OpenAI-synthesized data    ~50k examples (gpt-4.1-nano via Batch API)

Common schema:
{
  "id":           str,
  "instruction":  str,        # NL question
  "data_profile": {
      "columns":  [{"name": "x", "type": "string|number|date|category"}],
      "row_count_estimate": int,
      "sample_rows": [{...}, ...]
  },
  "chart_spec":   {           # storytelling-grade chart specification
      "chart_type": str,
      "encoding":   {"x": ..., "y": ..., "color": ..., "size": ..., "facet": ...},
      "title":      str,
      "subtitle":   str | null,
      "annotations": [{"target": ..., "text": ...}],
      "sort":       {"by": ..., "order": "asc|desc|natural"},
      "color_strategy": "highlight|categorical|sequential|diverging",
      "highlight_value": str | null,
      "axis_format": {"y_scale": "linear|log", "y_label": ..., "x_label": ...},
      "rationale":  str
  },
  "source":     str,
  "difficulty": "easy|medium|hard|unknown"
}

USAGE — runs as a multi-stage pipeline:

  # Stage 1: build nvBench portion
  uv run build_chart_mix.py nvbench --out data/nvbench.jsonl

  # Stage 2: prepare OpenAI batch from SQL mix v2
  uv run build_chart_mix.py synth-prepare --n 50000 \
    --batch-out data/openai_batch_in.jsonl

  # Stage 3: upload batch to OpenAI
  uv run build_chart_mix.py synth-submit \
    --batch-in data/openai_batch_in.jsonl

  # Stage 4: check status (poll)
  uv run build_chart_mix.py synth-status --batch-id batch_xxx

  # Stage 5: when complete, fetch + convert
  uv run build_chart_mix.py synth-fetch --batch-id batch_xxx \
    --out data/synth.jsonl

  # Stage 6: combine + push to HF
  uv run build_chart_mix.py combine-push \
    --inputs data/nvbench.jsonl data/synth.jsonl \
    --repo DanielRegaladoCardoso/chart-reasoning-mix-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import requests
import sqlglot
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Config / constants
# ----------------------------------------------------------------------------

NVBENCH_URL = (
    "https://raw.githubusercontent.com/TsinghuaDatabaseGroup/"
    "nvBench/main/NVBench.json"
)

OPENAI_MODEL = "gpt-4.1-nano"
OPENAI_BATCH_ENDPOINT = "/v1/chat/completions"
SQL_MIX_REPO = "DanielRegaladoCardoso/text-to-sql-mix-v2"
DEFAULT_OUT_REPO = "DanielRegaladoCardoso/chart-reasoning-mix-v1"

SEED = 42
TRAIN_RATIO = 0.95
VAL_RATIO = 0.025

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chart-mix")


# ----------------------------------------------------------------------------
# Storytelling system prompt — used in synthesis
# ----------------------------------------------------------------------------

STORYTELLING_PROMPT = """\
You design chart specs that follow Tufte/Knaflic/Few principles.
Given a question + SQL result columns, return ONE JSON object:

{
 "chart_type": one of bar|line|scatter|donut|histogram|boxplot|area|heatmap|sankey|funnel,
 "encoding": {"x": <col>, "y": <col>, "color": <col|null>, "size": <col|null>, "facet": <col|null>},
 "title": INSIGHT-DRIVEN ("Sales grew 47% in Q4" NOT "Sales by month"),
 "sort": {"by": <col>, "order": "asc"|"desc"|"natural"},
 "color_strategy": "highlight"|"categorical"|"sequential"|"diverging",
 "rationale": one sentence explaining the choice
}

Rules:
 categories→bar (horizontal if labels long), trend→line, parts≤6→donut,
 correlation→scatter, ranking→bar desc. Never pie>6, no 3D, no rainbow.
 Return JSON only, no prose.
"""


# ----------------------------------------------------------------------------
# Common record dataclass
# ----------------------------------------------------------------------------

@dataclass
class ChartExample:
    id: str
    instruction: str
    data_profile: dict
    chart_spec: dict
    source: str
    difficulty: str

    def to_dict(self) -> dict:
        return asdict(self)


def make_id(source: str, instruction: str, salt: str = "") -> str:
    h = hashlib.md5(f"{source}||{instruction}||{salt}".encode()).hexdigest()
    return f"{source}-{h[:16]}"


# ----------------------------------------------------------------------------
# Stage 1 — nvBench loader
# ----------------------------------------------------------------------------

def _infer_type_from_values(values) -> str:
    if not values:
        return "string"
    sample = values[0]
    if isinstance(sample, bool):
        return "category"
    if isinstance(sample, (int, float)):
        return "number"
    if isinstance(sample, str):
        # Heuristic: looks like date?
        if re.match(r"^\d{4}[-/]\d{1,2}([-/]\d{1,2})?$", sample):
            return "date"
        return "string"
    return "string"


NVBENCH_TYPE_MAP = {
    "scatter": "scatter",
    "bar": "bar",
    "stacked bar": "bar",
    "grouping bar": "bar",
    "grouping line": "line",
    "grouping scatter": "scatter",
    "line": "line",
    "pie": "donut",  # we standardize to donut for >6 slices
    "histogram": "histogram",
    "area": "area",
}

NVBENCH_HARDNESS = {
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
    "extra hard": "hard",
}


def load_nvbench(json_path: Path) -> list[ChartExample]:
    log.info(f"Loading nvBench JSON from {json_path}...")
    if not json_path.exists():
        log.info(f"  not found locally — downloading from {NVBENCH_URL}")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(NVBENCH_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(json_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
        log.info(f"  downloaded → {json_path.stat().st_size / 1e6:.1f} MB")

    raw = json.load(open(json_path))
    log.info(f"  {len(raw):,} nvBench entries")

    out: list[ChartExample] = []
    for entry_id, entry in tqdm(raw.items(), desc="nvbench"):
        try:
            vis_obj = entry["vis_obj"]
            chart_raw = (vis_obj.get("chart") or entry.get("chart", "")).lower()
            chart_type = NVBENCH_TYPE_MAP.get(chart_raw, chart_raw or "bar")

            x_name = vis_obj.get("x_name", "x")
            y_name = vis_obj.get("y_name", "y")
            x_data = (vis_obj.get("x_data") or [[]])[0]
            y_data = (vis_obj.get("y_data") or [[]])[0]
            classify = vis_obj.get("classify") or []
            color_name = None
            if classify and isinstance(classify, list) and classify:
                color_name = "category"  # nvBench uses generic class names

            x_type = _infer_type_from_values(x_data)
            y_type = _infer_type_from_values(y_data)

            sample_rows = []
            for i in range(min(5, len(x_data))):
                row = {x_name: x_data[i] if i < len(x_data) else None,
                       y_name: y_data[i] if i < len(y_data) else None}
                if color_name and i < len(classify):
                    row[color_name] = classify[i]
                sample_rows.append(row)

            data_profile = {
                "columns": [
                    {"name": x_name, "type": x_type},
                    {"name": y_name, "type": y_type},
                ] + ([{"name": "category", "type": "category"}] if color_name else []),
                "row_count_estimate": len(x_data),
                "sample_rows": sample_rows,
            }

            difficulty = NVBENCH_HARDNESS.get(
                (entry.get("hardness") or "").lower(), "unknown"
            )

            chart_spec = {
                "chart_type": chart_type,
                "encoding": {
                    "x": x_name,
                    "y": y_name,
                    "color": color_name,
                    "size": None,
                    "facet": None,
                },
                "title": None,        # nvBench doesn't supply storytelling titles
                "subtitle": None,
                "annotations": [],
                "sort": {"by": y_name, "order": "desc"} if chart_type == "bar" else {
                    "by": x_name, "order": "natural"
                },
                "color_strategy": "categorical" if color_name else "highlight",
                "highlight_value": None,
                "axis_format": {"y_scale": "linear", "y_label": None, "x_label": None},
                "rationale": None,
            }

            for i, q in enumerate(entry.get("nl_queries", [])):
                if not q:
                    continue
                out.append(ChartExample(
                    id=make_id("nvbench", q, salt=f"{entry_id}-{i}"),
                    instruction=q.strip(),
                    data_profile=data_profile,
                    chart_spec=chart_spec,
                    source="nvbench",
                    difficulty=difficulty,
                ))
        except Exception as e:
            log.debug(f"skipped nvbench entry {entry_id}: {e}")

    log.info(f"nvBench → {len(out):,} examples")
    return out


# ----------------------------------------------------------------------------
# Stage 2 — synthesis: build OpenAI batch JSONL
# ----------------------------------------------------------------------------

def _profile_from_sql(sql: str, schema_context: str) -> dict | None:
    """Parse SQL projections and infer result columns + types from CREATE TABLE."""
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    if not parsed:
        return None

    # Build {column_name: type} from schema CREATE TABLEs
    type_map: dict[str, str] = {}
    try:
        for stmt in sqlglot.parse(schema_context, dialect="sqlite") or []:
            if not stmt or not getattr(stmt, "this", None):
                continue
            for col in stmt.find_all(sqlglot.exp.ColumnDef):
                col_name = col.alias_or_name.lower()
                t = col.args.get("kind")
                t_str = (t.sql() if t else "TEXT").upper()
                if any(k in t_str for k in ("INT", "REAL", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC")):
                    type_map[col_name] = "number"
                elif any(k in t_str for k in ("DATE", "TIME")):
                    type_map[col_name] = "date"
                else:
                    type_map[col_name] = "string"
    except Exception:
        pass

    columns: list[dict] = []
    for proj in parsed.expressions or []:
        alias = proj.alias_or_name or "col"
        # Detect aggregations
        if proj.find(sqlglot.exp.AggFunc):
            ctype = "number"
        else:
            base = (proj.find(sqlglot.exp.Column) or proj).alias_or_name or ""
            ctype = type_map.get(base.lower(), "string")
        columns.append({"name": alias, "type": ctype})

    if not columns:
        return None

    return {
        "columns": columns,
        "row_count_estimate": None,  # we don't execute
        "sample_rows": [],
    }


def synth_prepare(n: int, out_path: Path, lookup_path: Path, sql_repo: str = SQL_MIX_REPO):
    """Sample N rows from the SQL mix and produce an OpenAI batch JSONL.

    Also writes a sidecar ``lookup_path`` JSONL mapping custom_id → (instruction,
    data_profile). OpenAI batch strips arbitrary metadata, so we keep the join
    table local.
    """
    log.info(f"Loading SQL mix from {sql_repo}...")
    ds = load_dataset(sql_repo, split="train")
    log.info(f"  {len(ds):,} rows available")

    random.seed(SEED)
    indices = random.sample(range(len(ds)), min(n * 3, len(ds)))  # oversample for failures

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lookup_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with open(out_path, "w") as f, open(lookup_path, "w") as flu:
        for idx in tqdm(indices, desc="synth-prepare"):
            if written >= n:
                break
            row = ds[idx]
            if not row.get("schema_context") or not row.get("instruction"):
                skipped += 1
                continue
            profile = _profile_from_sql(row["sql"], row["schema_context"])
            if not profile or len(profile["columns"]) == 0:
                skipped += 1
                continue

            # OpenAI custom_id has 64-char limit; sql mix ids are short enough but be safe
            cid = row["id"][:60]

            user_content = (
                f"Question: {row['instruction']}\n\n"
                f"SQL result columns: {json.dumps(profile['columns'])}\n\n"
                "Design the ideal chart for this question and result. "
                "Return only valid JSON per the system schema."
            )

            req = {
                "custom_id": cid,
                "method": "POST",
                "url": OPENAI_BATCH_ENDPOINT,
                "body": {
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": STORYTELLING_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "max_tokens": 400,
                    "response_format": {"type": "json_object"},
                    "temperature": 0.3,
                },
            }
            f.write(json.dumps(req) + "\n")

            flu.write(json.dumps({
                "custom_id": cid,
                "instruction": row["instruction"],
                "data_profile": profile,
                "source_id": row["id"],
                "source": row.get("source", ""),
            }) + "\n")
            written += 1

    log.info(f"synth-prepare → {written:,} requests → {out_path}")
    log.info(f"               lookup → {lookup_path}  (skipped {skipped:,})")
    return written


# ----------------------------------------------------------------------------
# Stage 3 — submit batch to OpenAI
# ----------------------------------------------------------------------------

def synth_submit(batch_in: Path) -> str:
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set in environment")
        sys.exit(1)
    client = OpenAI()
    log.info(f"Uploading {batch_in} to OpenAI...")
    file = client.files.create(file=open(batch_in, "rb"), purpose="batch")
    log.info(f"  file id: {file.id}")
    batch = client.batches.create(
        input_file_id=file.id,
        endpoint=OPENAI_BATCH_ENDPOINT,
        completion_window="24h",
        metadata={"project": "sql-agent-chart-reasoner"},
    )
    log.info(f"✅ Batch submitted: {batch.id}  status={batch.status}")
    log.info(f"   Track with: uv run build_chart_mix.py synth-status --batch-id {batch.id}")
    return batch.id


def synth_status(batch_id: str):
    from openai import OpenAI
    client = OpenAI()
    b = client.batches.retrieve(batch_id)
    log.info(f"Batch {batch_id}")
    log.info(f"  status:    {b.status}")
    log.info(f"  requests:  {b.request_counts}")
    log.info(f"  in file:   {b.input_file_id}")
    log.info(f"  out file:  {b.output_file_id}")
    log.info(f"  err file:  {b.error_file_id}")


# ----------------------------------------------------------------------------
# Stage 5 — fetch results, convert to common schema
# ----------------------------------------------------------------------------

def _heuristic_difficulty(chart_spec: dict) -> str:
    enc = chart_spec.get("encoding", {})
    extras = sum(1 for k in ("color", "size", "facet") if enc.get(k))
    if extras >= 2 or chart_spec.get("annotations"):
        return "hard"
    if extras == 1:
        return "medium"
    return "easy"


def synth_fetch(batch_id: str, out_path: Path, lookup_path: Path):
    from openai import OpenAI
    client = OpenAI()
    b = client.batches.retrieve(batch_id)
    if b.status != "completed":
        log.error(f"Batch not completed (status={b.status})")
        sys.exit(1)
    if not b.output_file_id:
        log.error("No output file id")
        sys.exit(1)

    # Load lookup table built during prepare
    log.info(f"Loading lookup table from {lookup_path}...")
    lookup: dict[str, dict] = {}
    with open(lookup_path) as f:
        for line in f:
            r = json.loads(line)
            lookup[r["custom_id"]] = r
    log.info(f"  {len(lookup):,} entries indexed")

    log.info(f"Downloading output file {b.output_file_id}...")
    raw = client.files.content(b.output_file_id).text

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    invalid_json = 0
    missing_lookup = 0
    with open(out_path, "w") as fout:
        for line in raw.strip().split("\n"):
            try:
                rec = json.loads(line)
                custom_id = rec["custom_id"]
                content = rec["response"]["body"]["choices"][0]["message"]["content"]
            except Exception:
                skipped += 1
                continue
            try:
                spec = json.loads(content)
            except Exception:
                invalid_json += 1
                continue

            meta = lookup.get(custom_id)
            if not meta:
                missing_lookup += 1
                continue

            ex = ChartExample(
                id=make_id("synth", meta["source_id"]),
                instruction=meta["instruction"],
                data_profile=meta["data_profile"],
                chart_spec=spec,
                source="synth-openai-gpt41nano",
                difficulty=_heuristic_difficulty(spec),
            )
            fout.write(json.dumps(ex.to_dict()) + "\n")
            written += 1

    log.info(f"synth-fetch → {written:,} converted")
    log.info(f"             skipped: {skipped:,}  invalid_json: {invalid_json:,}  "
             f"missing_lookup: {missing_lookup:,}")


# ----------------------------------------------------------------------------
# Stage 6 — combine + push
# ----------------------------------------------------------------------------

SOURCE_LICENSES = {
    "nvbench": ("nvBench", "MIT", "https://github.com/TsinghuaDatabaseGroup/nvBench",
                "Tsinghua DB Group · 7,247 entries × 5 NL paraphrases"),
    "synth-openai-gpt41nano": (
        "OpenAI gpt-4.1-nano synthesis", "Apache-2.0",
        "https://huggingface.co/datasets/" + SQL_MIX_REPO,
        "Synthesized chart specs from text-to-sql-mix-v2 questions via OpenAI Batch API.",
    ),
}


def deduplicate(examples: list[ChartExample]) -> list[ChartExample]:
    seen = set()
    out = []
    for ex in examples:
        key = hashlib.md5(
            f"{ex.instruction.lower()}||{ex.chart_spec.get('chart_type','')}".encode()
        ).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def split(examples: list[ChartExample]):
    random.seed(SEED)
    random.shuffle(examples)
    n = len(examples)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * (TRAIN_RATIO + VAL_RATIO))
    return {
        "train": examples[:n_train],
        "validation": examples[n_train:n_val],
        "test": examples[n_val:],
    }


def to_hf_dataset(examples: list[ChartExample]) -> Dataset:
    """Serialize nested ``data_profile`` and ``chart_spec`` as JSON strings to
    avoid Arrow type-inference errors caused by heterogeneous sample-row
    values (str / int / float / date) across rows."""
    rows = []
    for e in examples:
        rows.append({
            "id": e.id,
            "instruction": e.instruction,
            "data_profile": json.dumps(e.data_profile, ensure_ascii=False, default=str),
            "chart_spec": json.dumps(e.chart_spec, ensure_ascii=False, default=str),
            "source": e.source,
            "difficulty": e.difficulty,
        })
    return Dataset.from_list(rows)


def build_card(stats: dict, repo: str) -> str:
    lines = []
    lines.append("---")
    lines.append("language:\n  - en")
    lines.append("license: apache-2.0")
    lines.append("task_categories:\n  - text-generation")
    lines.append("tags:\n  - chart-generation\n  - data-visualization\n  - storytelling-with-data\n  - chart-spec\n  - sql\n  - llm-distillation")
    lines.append("pretty_name: Chart Reasoning Mix v1")
    lines.append("size_categories:\n  - 10K<n<100K")
    lines.append("---")
    lines.append("")
    lines.append("# 📊 Chart Reasoning Mix · v1")
    lines.append("")
    lines.append("Training data for fine-tuning compact LLMs (Phi-3 Mini, Qwen 2.5 3B,")
    lines.append("DeepSeek Coder 1.3B) to map **`(natural-language question + SQL result")
    lines.append("schema) → storytelling-grade chart specification`**.")
    lines.append("")
    lines.append("> Powers the Chart Reasoner in the [SQL Agent LLMOps]"
                 "(https://github.com/DanielRegaladoUMiami/sql-agent-llmops) project.")
    lines.append("")
    lines.append(f"| 📦 Total | 🎨 Storytelling fields | 🧪 Format |")
    lines.append(f"|---------|------------------------|-----------|")
    lines.append(f"| **{stats['total']:,}** rows | title, subtitle, annotations, sort, color_strategy, rationale | JSON-validated |")
    lines.append("")
    lines.append("## 📐 Schema")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({
        "id": "string",
        "instruction": "string (NL question)",
        "data_profile": {
            "columns": [{"name": "string", "type": "string|number|date|category"}],
            "row_count_estimate": "int|null",
            "sample_rows": [{"col": "value"}]
        },
        "chart_spec": {
            "chart_type": "bar|line|scatter|donut|histogram|boxplot|area|heatmap|map|sankey|funnel",
            "encoding": {"x": "col", "y": "col", "color": "col|null", "size": "col|null", "facet": "col|null"},
            "title": "insight-driven title",
            "subtitle": "context|null",
            "annotations": [{"target": "col_or_value", "text": "callout"}],
            "sort": {"by": "col", "order": "asc|desc|natural"},
            "color_strategy": "highlight|categorical|sequential|diverging",
            "highlight_value": "value|null",
            "axis_format": {"y_scale": "linear|log", "y_label": "label", "x_label": "label"},
            "rationale": "why this chart"
        },
        "source": "string",
        "difficulty": "easy|medium|hard|unknown"
    }, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 📦 Splits")
    lines.append("")
    lines.append("| Split | Rows |")
    lines.append("|-------|------|")
    for split_name, n in stats["splits"].items():
        lines.append(f"| `{split_name}` | {n:,} |")
    lines.append("")
    lines.append("## 📊 Sources")
    lines.append("")
    lines.append("| Source | Rows | License | Link | Notes |")
    lines.append("|--------|------|---------|------|-------|")
    for src, n in sorted(stats["by_source"].items(), key=lambda x: -x[1]):
        meta = SOURCE_LICENSES.get(src, (src, "Unknown", "", ""))
        name, lic, url, notes = meta
        link = f"[link]({url})" if url else "—"
        lines.append(f"| {name} (`{src}`) | {n:,} | {lic} | {link} | {notes} |")
    lines.append("")
    lines.append("## 🎯 Storytelling principles")
    lines.append("")
    lines.append("Synthesized rows are generated with a system prompt distilling:")
    lines.append("- **Edward Tufte** — data-ink ratio, integrity, small multiples")
    lines.append("- **Cole Nussbaumer Knaflic** — clutter elimination, action-driven titles")
    lines.append("- **Stephen Few** — perceptual encoding, dashboard hygiene")
    lines.append("")
    lines.append("Models trained on this dataset learn not just **chart type** but:")
    lines.append("- ✅ Insight-driven titles (\"Sales grew 47%\", not \"Sales over time\")")
    lines.append("- ✅ Highlight-color strategy (one accent, gray background)")
    lines.append("- ✅ Smart sorting (value-desc for rankings)")
    lines.append("- ✅ Annotation choices (call out the answer)")
    lines.append("- ✅ Rationale (the model can explain its choice)")
    lines.append("")
    lines.append("## 🚀 Usage")
    lines.append("")
    lines.append("```python")
    lines.append("from datasets import load_dataset")
    lines.append(f'ds = load_dataset("{repo}")')
    lines.append("ex = ds['train'][0]")
    lines.append("print(ex['instruction'])")
    lines.append("print(ex['chart_spec']['title'])")
    lines.append("print(ex['chart_spec']['rationale'])")
    lines.append("```")
    lines.append("")
    lines.append("## 🛠 Build pipeline")
    lines.append("")
    lines.append("Open-source pipeline: [`training/data_pipelines/build_chart_mix.py`]"
                "(https://github.com/DanielRegaladoUMiami/sql-agent-llmops/blob/main/training/data_pipelines/build_chart_mix.py)")
    lines.append("")
    lines.append("## 📝 Citation")
    lines.append("")
    lines.append("```bibtex")
    lines.append("@dataset{regalado2026chartmix,")
    lines.append("  author = {Regalado Cardoso, Daniel},")
    lines.append("  title  = {Chart Reasoning Mix v1},")
    lines.append("  year   = {2026},")
    lines.append(f"  url    = {{https://huggingface.co/datasets/{repo}}}")
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("Plus original sources — see Sources table.")
    lines.append("")
    return "\n".join(lines)


def combine_push(input_paths: list[Path], repo: str, push: bool, save_local: Path | None):
    examples: list[ChartExample] = []
    for p in input_paths:
        log.info(f"Reading {p}...")
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                examples.append(ChartExample(**d))
    log.info(f"Loaded {len(examples):,} total examples")

    examples = deduplicate(examples)
    log.info(f"After dedup: {len(examples):,}")

    splits = split(examples)
    log.info(
        f"Splits: train={len(splits['train']):,}  "
        f"val={len(splits['validation']):,}  test={len(splits['test']):,}"
    )

    dsd = DatasetDict({k: to_hf_dataset(v) for k, v in splits.items()})

    by_source = Counter(e.source for e in examples)
    by_diff = Counter(e.difficulty for e in examples)
    stats = {
        "total": len(examples),
        "splits": {k: len(v) for k, v in splits.items()},
        "by_source": dict(by_source),
        "by_difficulty": dict(by_diff),
    }

    if save_local:
        log.info(f"Saving locally to {save_local}")
        dsd.save_to_disk(str(save_local))

    if push:
        log.info(f"Pushing to {repo} (public)...")
        dsd.push_to_hub(repo, private=False)

        card = build_card(stats, repo)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
            commit_message="Initial dataset card",
        )
        log.info(f"✅ Done: https://huggingface.co/datasets/{repo}")
    else:
        log.info("Skipping push (--push not set). Stats:")
        log.info(json.dumps(stats, indent=2))


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Build chart-reasoning training mix")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("nvbench", help="Build nvBench portion → JSONL")
    p1.add_argument("--out", type=Path, default=Path("data/nvbench.jsonl"))
    p1.add_argument("--cache", type=Path, default=Path("data/cache/NVBench.json"))

    p2 = sub.add_parser("synth-prepare", help="Build OpenAI Batch input JSONL from SQL mix")
    p2.add_argument("--n", type=int, default=50000)
    p2.add_argument("--batch-out", type=Path, default=Path("data/openai_batch_in.jsonl"))
    p2.add_argument("--lookup-out", type=Path, default=Path("data/openai_batch_lookup.jsonl"))
    p2.add_argument("--sql-repo", default=SQL_MIX_REPO)

    p3 = sub.add_parser("synth-submit", help="Upload prepared batch to OpenAI")
    p3.add_argument("--batch-in", type=Path, default=Path("data/openai_batch_in.jsonl"))

    p4 = sub.add_parser("synth-status", help="Check OpenAI Batch status")
    p4.add_argument("--batch-id", required=True)

    p5 = sub.add_parser("synth-fetch", help="Fetch completed batch + convert")
    p5.add_argument("--batch-id", required=True)
    p5.add_argument("--out", type=Path, default=Path("data/synth.jsonl"))
    p5.add_argument("--lookup", type=Path, default=Path("data/openai_batch_lookup.jsonl"))

    p6 = sub.add_parser("combine-push", help="Combine JSONL inputs + push to HF")
    p6.add_argument("--inputs", nargs="+", type=Path, required=True)
    p6.add_argument("--repo", default=DEFAULT_OUT_REPO)
    p6.add_argument("--push", action="store_true")
    p6.add_argument("--save-local", type=Path, default=None)

    args = p.parse_args()

    if args.cmd == "nvbench":
        examples = load_nvbench(args.cache)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            for e in examples:
                f.write(json.dumps(e.to_dict()) + "\n")
        log.info(f"✅ Wrote {len(examples):,} → {args.out}")

    elif args.cmd == "synth-prepare":
        synth_prepare(args.n, args.batch_out, args.lookup_out, args.sql_repo)

    elif args.cmd == "synth-submit":
        bid = synth_submit(args.batch_in)
        # Save batch id for convenience
        Path("data/last_batch_id.txt").write_text(bid)

    elif args.cmd == "synth-status":
        synth_status(args.batch_id)

    elif args.cmd == "synth-fetch":
        synth_fetch(args.batch_id, args.out, args.lookup)

    elif args.cmd == "combine-push":
        combine_push(args.inputs, args.repo, args.push, args.save_local)


if __name__ == "__main__":
    main()

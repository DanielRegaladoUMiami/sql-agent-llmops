# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=2.19",
#     "huggingface_hub>=0.23",
#     "sqlglot>=23.0",
#     "tqdm",
#     "pyyaml",
# ]
# ///
"""
Build the text-to-SQL training mix for SQL Agent LLMOps.

Combines multiple high-quality text-to-SQL datasets from HuggingFace,
normalizes them to a common schema, deduplicates, quality-filters with
sqlglot, splits train/val/test and pushes to HuggingFace Hub as a
public dataset with a full dataset card and source attribution.

Runs as a UV script locally OR on HF Jobs (cpu-basic is enough).

Usage (local):
    uv run training/data_pipelines/build_sql_mix.py
    uv run training/data_pipelines/build_sql_mix.py --sample 1000  # smoke test
    uv run training/data_pipelines/build_sql_mix.py --push

Usage (HF Jobs):
    hf jobs uv run --flavor cpu-basic --timeout 2h \
        --secrets HF_TOKEN \
        training/data_pipelines/build_sql_mix.py --push
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Callable

import sqlglot
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

OUTPUT_REPO = "DanielRegaladoCardoso/text-to-sql-mix-v2"
SEED = 42
TRAIN_RATIO = 0.95
VAL_RATIO = 0.025
# test = 1 - train - val

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sql-mix")


# ----------------------------------------------------------------------------
# Common schema
# ----------------------------------------------------------------------------

@dataclass
class Example:
    id: str
    instruction: str
    schema_context: str
    sql: str
    source: str
    dialect: str
    difficulty: str  # easy | medium | hard | unknown

    def validate(self) -> bool:
        return bool(self.instruction and self.sql)


def make_id(source: str, instruction: str, sql: str) -> str:
    h = hashlib.md5(f"{source}||{instruction}||{sql}".encode("utf-8")).hexdigest()
    return f"{source}-{h[:16]}"


# ----------------------------------------------------------------------------
# Source loaders — each returns a list[Example]
# ----------------------------------------------------------------------------

def load_sql_create_context(sample: int | None) -> list[Example]:
    """b-mc2/sql-create-context — 78k. Schema + question + SQL."""
    log.info("Loading b-mc2/sql-create-context...")
    ds = load_dataset("b-mc2/sql-create-context", split="train")
    if sample:
        ds = ds.select(range(min(sample, len(ds))))
    out = []
    for ex in tqdm(ds, desc="sql-create-context"):
        out.append(Example(
            id=make_id("sql-create-context", ex["question"], ex["answer"]),
            instruction=ex["question"],
            schema_context=ex.get("context", ""),
            sql=ex["answer"],
            source="sql-create-context",
            dialect="generic",
            difficulty="unknown",
        ))
    return out


def load_gretel(sample: int | None) -> list[Example]:
    """gretelai/synthetic_text_to_sql — ~105k. Synthetic, diverse domains."""
    log.info("Loading gretelai/synthetic_text_to_sql...")
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    if sample:
        ds = ds.select(range(min(sample, len(ds))))
    out = []
    for ex in tqdm(ds, desc="gretel"):
        complexity = (ex.get("sql_complexity") or "").lower()
        difficulty = {
            "basic sql": "easy",
            "aggregation": "medium",
            "single join": "medium",
            "multiple_joins": "hard",
            "window functions": "hard",
            "subqueries": "hard",
            "set operations": "hard",
            "cte": "hard",
        }.get(complexity, "medium")
        out.append(Example(
            id=make_id("gretel", ex["sql_prompt"], ex["sql"]),
            instruction=ex["sql_prompt"],
            schema_context=ex.get("sql_context", ""),
            sql=ex["sql"],
            source="gretel-synthetic",
            dialect="postgres",
            difficulty=difficulty,
        ))
    return out


def load_know_sql(sample: int | None) -> list[Example]:
    """knowrohit07/know_sql — compact & clean."""
    log.info("Loading knowrohit07/know_sql...")
    ds = load_dataset("knowrohit07/know_sql", split="validation")
    if sample:
        ds = ds.select(range(min(sample, len(ds))))
    out = []
    for ex in tqdm(ds, desc="know_sql"):
        out.append(Example(
            id=make_id("know_sql", ex["question"], ex["answer"]),
            instruction=ex["question"],
            schema_context=ex.get("context", ""),
            sql=ex["answer"],
            source="know_sql",
            dialect="generic",
            difficulty="unknown",
        ))
    return out


def load_nstext2sql(sample: int | None) -> list[Example]:
    """NumbersStation/NSText2SQL — ~290k. Multi-dialect, volume."""
    log.info("Loading NumbersStation/NSText2SQL...")
    ds = load_dataset("NumbersStation/NSText2SQL", split="train")
    if sample:
        ds = ds.select(range(min(sample, len(ds))))
    out = []
    # This dataset has: instruction (schema + Q) + output (SQL) + source
    for ex in tqdm(ds, desc="nstext2sql"):
        instruction_field = ex.get("instruction", "")
        output = ex.get("output", "")
        src = ex.get("source", "ns")
        # Try to split schema from question — schema typically ends with blank line
        schema, question = _split_schema_and_question(instruction_field)
        out.append(Example(
            id=make_id(f"ns-{src}", question, output),
            instruction=question,
            schema_context=schema,
            sql=output,
            source=f"nstext2sql-{src}",
            dialect="generic",
            difficulty="unknown",
        ))
    return out


def load_clinton(sample: int | None) -> list[Example]:
    """Clinton/Text-to-sql-v1 — large instruct format."""
    log.info("Loading Clinton/Text-to-sql-v1...")
    ds = load_dataset("Clinton/Text-to-sql-v1", split="train")
    if sample:
        ds = ds.select(range(min(sample, len(ds))))
    out = []
    for ex in tqdm(ds, desc="clinton"):
        instruction = ex.get("instruction", "")
        inp = ex.get("input", "")
        output = ex.get("response") or ex.get("output", "")
        # Some rows pack schema into input
        out.append(Example(
            id=make_id("clinton", instruction, output),
            instruction=instruction,
            schema_context=inp,
            sql=output,
            source="clinton-text2sql",
            dialect="generic",
            difficulty="unknown",
        ))
    return out


HAYDUK_INSTRUCTION_RE = re.compile(r"### Instruction:\s*(.*?)\s*### Input:", re.DOTALL)
HAYDUK_INPUT_RE = re.compile(r"### Input:\s*(.*?)\s*### Response:", re.DOTALL)


def load_chrishayduk(sample: int | None) -> list[Example]:
    """ChrisHayduk/Llama-2-SQL-Dataset — Alpaca-packed prompt + SQL output."""
    log.info("Loading ChrisHayduk/Llama-2-SQL-Dataset...")
    ds = load_dataset("ChrisHayduk/Llama-2-SQL-Dataset", split="train")
    if sample:
        ds = ds.select(range(min(sample, len(ds))))
    out = []
    for ex in tqdm(ds, desc="chrishayduk"):
        raw_input = ex.get("input", "")
        output = ex.get("output", "")
        m_inst = HAYDUK_INSTRUCTION_RE.search(raw_input)
        m_schema = HAYDUK_INPUT_RE.search(raw_input)
        instruction = m_inst.group(1).strip() if m_inst else ""
        schema = m_schema.group(1).strip() if m_schema else ""
        if not instruction:
            continue
        out.append(Example(
            id=make_id("hayduk", instruction, output),
            instruction=instruction,
            schema_context=schema,
            sql=output,
            source="hayduk-llama2-sql",
            dialect="generic",
            difficulty="unknown",
        ))
    return out


KAXAP_SCHEMA_RE = re.compile(r"CREATE TABLE.*?\);", re.DOTALL)
KAXAP_Q_RE = re.compile(r'following question:\s*"([^"]+)"', re.DOTALL)
KAXAP_SQL_RE = re.compile(r"\[/INST\]\s*(.*?)\s*</s>", re.DOTALL)

BUGDARYAN_INST_RE = re.compile(
    r"### Instruction:\s*(.*?)\s*### Response:", re.DOTALL
)
BUGDARYAN_RESP_RE = re.compile(r"### Response:\s*(.*?)$", re.DOTALL)
CREATE_TABLE_RE = re.compile(r"CREATE TABLE[\s\S]*?\)", re.MULTILINE)


def load_motherduck(sample: int | None) -> list[Example]:
    """motherduckdb/duckdb-text2sql-25k — DuckDB dialect with inline schema."""
    log.info("Loading motherduckdb/duckdb-text2sql-25k...")
    ds = load_dataset("motherduckdb/duckdb-text2sql-25k", split="train")
    if sample:
        ds = ds.select(range(min(sample, len(ds))))
    out = []
    for ex in tqdm(ds, desc="motherduck"):
        out.append(Example(
            id=make_id("motherduck", ex["prompt"], ex["query"]),
            instruction=ex["prompt"],
            schema_context=ex.get("schema", ""),
            sql=ex["query"],
            source="motherduck-duckdb",
            dialect="duckdb",
            difficulty="unknown",
        ))
    return out


def load_pipable_spider_bird(sample: int | None) -> list[Example]:
    """PipableAI/pip-txt-to-sql-spider-bird-dataset — Spider+BIRD with schemas."""
    log.info("Loading PipableAI/pip-txt-to-sql-spider-bird-dataset...")
    ds = load_dataset("PipableAI/pip-txt-to-sql-spider-bird-dataset", split="train")
    if sample:
        ds = ds.select(range(min(sample, len(ds))))
    out = []
    for ex in tqdm(ds, desc="pipable-spider-bird"):
        out.append(Example(
            id=make_id("pip-spider-bird", ex["question"], ex["query"]),
            instruction=ex["question"],
            schema_context=ex.get("schema", ""),
            sql=ex["query"],
            source="pipable-spider-bird",
            dialect="sqlite",
            difficulty="hard",  # Spider/BIRD are complex by design
        ))
    return out


def load_kaxap(sample: int | None) -> list[Example]:
    """kaxap/llama2-sql-instruct — packed Llama-2 [INST] format."""
    log.info("Loading kaxap/llama2-sql-instruct...")
    ds = load_dataset("kaxap/llama2-sql-instruct", split="train")
    if sample:
        ds = ds.select(range(min(sample, len(ds))))
    out = []
    for ex in tqdm(ds, desc="kaxap"):
        text = ex.get("text", "")
        m_schema = KAXAP_SCHEMA_RE.search(text)
        m_q = KAXAP_Q_RE.search(text)
        m_sql = KAXAP_SQL_RE.search(text)
        if not (m_q and m_sql):
            continue
        schema = m_schema.group(0) if m_schema else ""
        question = m_q.group(1).strip()
        sql = m_sql.group(1).strip().rstrip(";") + ";" if m_sql.group(1).strip() else ""
        out.append(Example(
            id=make_id("kaxap", question, sql),
            instruction=question,
            schema_context=schema,
            sql=sql,
            source="kaxap-llama2",
            dialect="generic",
            difficulty="unknown",
        ))
    return out


def load_bugdaryan(sample: int | None) -> list[Example]:
    """bugdaryan/spider-natsql-wikisql-instruct — Alpaca packed."""
    log.info("Loading bugdaryan/spider-natsql-wikisql-instruct...")
    ds = load_dataset("bugdaryan/spider-natsql-wikisql-instruct", split="train")
    if sample:
        ds = ds.select(range(min(sample, len(ds))))
    out = []
    for ex in tqdm(ds, desc="bugdaryan"):
        text = ex.get("text", "")
        m_inst = BUGDARYAN_INST_RE.search(text)
        m_resp = BUGDARYAN_RESP_RE.search(text)
        if not (m_inst and m_resp):
            continue
        inst_block = m_inst.group(1).strip()
        sql = m_resp.group(1).strip()
        # Inside inst_block the CREATE TABLE is inline; split it out
        schemas = CREATE_TABLE_RE.findall(inst_block)
        schema = "\n\n".join(schemas) if schemas else ""
        # Remove schemas from inst_block to get the NL question
        question = CREATE_TABLE_RE.sub("", inst_block).strip()
        # Remove prefix like "Convert text to SQLite query:"
        question = re.sub(r"^[^:]+:\s*", "", question, count=1).strip()
        if not question:
            continue
        out.append(Example(
            id=make_id("bugdaryan", question, sql),
            instruction=question,
            schema_context=schema,
            sql=sql,
            source="bugdaryan-spider-wikisql",
            dialect="sqlite",
            difficulty="unknown",
        ))
    return out


SOURCES: dict[str, Callable[[int | None], list[Example]]] = {
    "sql-create-context": load_sql_create_context,
    "gretel": load_gretel,
    "know_sql": load_know_sql,
    "nstext2sql": load_nstext2sql,
    "clinton": load_clinton,
    "chrishayduk": load_chrishayduk,
    "motherduck": load_motherduck,
    "pipable-spider-bird": load_pipable_spider_bird,
    "kaxap": load_kaxap,
    "bugdaryan": load_bugdaryan,
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

SCHEMA_HINTS = re.compile(r"(create\s+table|schema\s*:|tables?\s*:)", re.IGNORECASE)


def _split_schema_and_question(text: str) -> tuple[str, str]:
    """Best-effort split of a combined schema+question blob."""
    if not text:
        return "", ""
    # If it contains a blank line, take everything before last blank line
    # as schema, the rest as question.
    parts = re.split(r"\n\s*\n", text.strip())
    if len(parts) >= 2:
        schema = "\n\n".join(parts[:-1])
        question = parts[-1]
        if SCHEMA_HINTS.search(schema):
            return schema, question
    return "", text.strip()


def is_parseable_sql(sql: str) -> bool:
    if not sql or len(sql.strip()) < 5:
        return False
    try:
        sqlglot.parse_one(sql)
        return True
    except Exception:
        return False


def infer_difficulty(sql: str, current: str) -> str:
    if current != "unknown":
        return current
    s = sql.lower()
    if any(k in s for k in ["with ", "partition by", "over(", "over ("]):
        return "hard"
    if s.count("join") >= 2 or "having" in s or s.count("select") > 1:
        return "hard"
    if "join" in s or "group by" in s:
        return "medium"
    return "easy"


def filter_and_enrich(examples: list[Example]) -> list[Example]:
    log.info("Quality filter + enrichment...")
    out = []
    for ex in tqdm(examples, desc="filter"):
        if not ex.validate():
            continue
        if len(ex.sql) > 4000 or len(ex.instruction) > 2000:
            continue
        if not is_parseable_sql(ex.sql):
            continue
        ex.difficulty = infer_difficulty(ex.sql, ex.difficulty)
        ex.instruction = ex.instruction.strip()
        ex.schema_context = (ex.schema_context or "").strip()
        ex.sql = ex.sql.strip()
        out.append(ex)
    log.info(f"  kept {len(out):,} / {len(examples):,}")
    return out


def deduplicate(examples: list[Example]) -> list[Example]:
    log.info("Deduplicating...")
    seen = set()
    out = []
    for ex in examples:
        key = hashlib.md5(
            f"{ex.instruction.lower()}||{ex.sql.lower()}".encode()
        ).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    log.info(f"  kept {len(out):,} / dedup input")
    return out


def split_examples(examples: list[Example]) -> dict[str, list[Example]]:
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


def to_hf_dataset(examples: list[Example]) -> Dataset:
    return Dataset.from_list([asdict(e) for e in examples])


# ----------------------------------------------------------------------------
# Dataset card
# ----------------------------------------------------------------------------

SOURCE_ATTRIBUTION = [
    ("b-mc2/sql-create-context", "sql-create-context", "CC-BY-4.0",
     "https://huggingface.co/datasets/b-mc2/sql-create-context",
     "Builds on Spider and WikiSQL; adds CREATE TABLE schema context."),
    ("gretelai/synthetic_text_to_sql", "gretel-synthetic", "Apache-2.0",
     "https://huggingface.co/datasets/gretelai/synthetic_text_to_sql",
     "105k synthetic SQL examples across 11 domains (finance, healthcare, retail, etc.)."),
    ("knowrohit07/know_sql", "know_sql", "Apache-2.0",
     "https://huggingface.co/datasets/knowrohit07/know_sql",
     "Compact and clean text-to-SQL pairs."),
    ("Clinton/Text-to-sql-v1", "clinton-text2sql", "Apache-2.0",
     "https://huggingface.co/datasets/Clinton/Text-to-sql-v1",
     "Large instruction-tuned SQL dataset."),
    ("NumbersStation/NSText2SQL", "nstext2sql-*", "See source",
     "https://huggingface.co/datasets/NumbersStation/NSText2SQL",
     "290k examples from 20+ sources, multi-dialect."),
    ("ChrisHayduk/Llama-2-SQL-Dataset", "hayduk-llama2-sql", "Apache-2.0",
     "https://huggingface.co/datasets/ChrisHayduk/Llama-2-SQL-Dataset",
     "Llama-2 instruction format text-to-SQL."),
    ("motherduckdb/duckdb-text2sql-25k", "motherduck-duckdb", "CC-BY-4.0",
     "https://huggingface.co/datasets/motherduckdb/duckdb-text2sql-25k",
     "25k DuckDB-dialect SQL examples by the MotherDuck team."),
    ("PipableAI/pip-txt-to-sql-spider-bird-dataset", "pipable-spider-bird", "Apache-2.0",
     "https://huggingface.co/datasets/PipableAI/pip-txt-to-sql-spider-bird-dataset",
     "Spider + BIRD benchmarks with inline CREATE TABLE schemas (complex queries)."),
    ("kaxap/llama2-sql-instruct", "kaxap-llama2", "Apache-2.0",
     "https://huggingface.co/datasets/kaxap/llama2-sql-instruct",
     "Llama-2 [INST]-formatted SQL instructions."),
    ("bugdaryan/spider-natsql-wikisql-instruct", "bugdaryan-spider-wikisql", "Apache-2.0",
     "https://huggingface.co/datasets/bugdaryan/spider-natsql-wikisql-instruct",
     "Spider + NatSQL + WikiSQL packed into Alpaca instruct format."),
]


def build_dataset_card(stats: dict) -> str:
    lines = []
    lines.append("---")
    lines.append("language:")
    lines.append("  - en")
    lines.append("license: apache-2.0")
    lines.append("task_categories:")
    lines.append("  - text-generation")
    lines.append("  - text2text-generation")
    lines.append("tags:")
    lines.append("  - sql")
    lines.append("  - text-to-sql")
    lines.append("  - code-generation")
    lines.append("  - instruction-tuning")
    lines.append("pretty_name: Text-to-SQL Training Mix v1")
    lines.append("size_categories:")
    lines.append("  - 100K<n<1M")
    lines.append("---")
    lines.append("")
    lines.append("# Text-to-SQL Training Mix v1")
    lines.append("")
    lines.append("A curated, deduplicated and quality-filtered mix of six high-quality")
    lines.append("text-to-SQL datasets from HuggingFace, designed for fine-tuning code LLMs")
    lines.append("(Qwen 2.5 Coder, DeepSeek Coder, Llama-3, etc.) on SQL generation.")
    lines.append("")
    lines.append("This dataset powers the SQL Generator in the [SQL Agent LLMOps]"
                 "(https://github.com/DanielRegaladoUMiami/sql-agent-llmops) project.")
    lines.append("")
    lines.append("## Schema")
    lines.append("")
    lines.append("| Field | Type | Description |")
    lines.append("|-------|------|-------------|")
    lines.append("| `id` | string | Stable hash-based identifier |")
    lines.append("| `instruction` | string | Natural language question / instruction |")
    lines.append("| `schema_context` | string | Database schema (CREATE TABLE or prose) — may be empty |")
    lines.append("| `sql` | string | Target SQL query (parseable via sqlglot) |")
    lines.append("| `source` | string | Original dataset tag |")
    lines.append("| `dialect` | string | SQL dialect hint (generic, postgres, mysql, sqlite) |")
    lines.append("| `difficulty` | string | easy / medium / hard (heuristic) |")
    lines.append("")
    lines.append("## Splits")
    lines.append("")
    lines.append("| Split | Examples |")
    lines.append("|-------|----------|")
    for split, n in stats["splits"].items():
        lines.append(f"| {split} | {n:,} |")
    lines.append("")
    lines.append("Split ratios: 95% train / 2.5% validation / 2.5% test")
    lines.append("")
    lines.append("## Source attribution")
    lines.append("")
    lines.append("This dataset is a derivative work combining the following sources:")
    lines.append("")
    lines.append("| Source | Tag in `source` | License | Link | Notes |")
    lines.append("|--------|-----------------|---------|------|-------|")
    for name, tag, lic, url, notes in SOURCE_ATTRIBUTION:
        lines.append(f"| `{name}` | `{tag}` | {lic} | [link]({url}) | {notes} |")
    lines.append("")
    lines.append("## Per-source statistics")
    lines.append("")
    lines.append("| Source | Examples |")
    lines.append("|--------|----------|")
    for src, n in sorted(stats["by_source"].items(), key=lambda x: -x[1]):
        lines.append(f"| `{src}` | {n:,} |")
    lines.append("")
    lines.append("## Difficulty distribution")
    lines.append("")
    lines.append("| Difficulty | Examples |")
    lines.append("|------------|----------|")
    for diff, n in sorted(stats["by_difficulty"].items(), key=lambda x: -x[1]):
        lines.append(f"| `{diff}` | {n:,} |")
    lines.append("")
    lines.append("## Pipeline")
    lines.append("")
    lines.append("1. Download each source from the HuggingFace Hub.")
    lines.append("2. Normalize into the common schema shown above.")
    lines.append("3. Filter: reject rows where SQL is unparseable by `sqlglot`, or")
    lines.append("   where instruction/SQL exceed length thresholds.")
    lines.append("4. Deduplicate by MD5 hash of `(lower(instruction), lower(sql))`.")
    lines.append("5. Heuristic difficulty tagging based on SQL complexity signals.")
    lines.append("6. Stratified shuffle + 95/2.5/2.5 split.")
    lines.append("")
    lines.append("## Usage")
    lines.append("")
    lines.append("```python")
    lines.append("from datasets import load_dataset")
    lines.append(f'ds = load_dataset("{OUTPUT_REPO}")')
    lines.append("")
    lines.append("def format_example(ex):")
    lines.append("    ctx = ex['schema_context']")
    lines.append("    prompt = (")
    lines.append('        f"You are a SQL expert. Given the schema below, write a SQL query.\\n\\n"')
    lines.append('        f"### Schema:\\n{ctx}\\n\\n"')
    lines.append('        f"### Question:\\n{ex[\'instruction\']}\\n\\n"')
    lines.append('        f"### SQL:\\n"')
    lines.append("    )")
    lines.append("    return {'prompt': prompt, 'completion': ex['sql']}")
    lines.append("")
    lines.append("ds = ds.map(format_example)")
    lines.append("```")
    lines.append("")
    lines.append("## Citation")
    lines.append("")
    lines.append("If you use this dataset, please cite the original sources and this mix:")
    lines.append("")
    lines.append("```bibtex")
    lines.append("@dataset{regalado2026textsqlmix,")
    lines.append("  author = {Regalado Cardoso, Daniel},")
    lines.append(f"  title  = {{{OUTPUT_REPO.split('/')[-1]}}},")
    lines.append("  year   = {2026},")
    lines.append("  url    = {https://huggingface.co/datasets/" + OUTPUT_REPO + "}")
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("## License")
    lines.append("")
    lines.append("This mix is released under **Apache-2.0**. Individual source licenses")
    lines.append("remain attached to their respective rows — see the source attribution")
    lines.append("table above. Please review the license of each upstream dataset before")
    lines.append("using this mix in commercial applications.")
    lines.append("")
    return "\n".join(lines)


def compute_stats(splits: dict[str, list[Example]]) -> dict:
    by_source: dict[str, int] = defaultdict(int)
    by_difficulty: dict[str, int] = defaultdict(int)
    split_sizes = {}
    for name, exs in splits.items():
        split_sizes[name] = len(exs)
        for ex in exs:
            by_source[ex.source] += 1
            by_difficulty[ex.difficulty] += 1
    return {
        "splits": split_sizes,
        "by_source": dict(by_source),
        "by_difficulty": dict(by_difficulty),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Take only N rows from each source (for smoke testing)")
    parser.add_argument("--sources", nargs="+", default=list(SOURCES.keys()),
                        choices=list(SOURCES.keys()))
    parser.add_argument("--push", action="store_true",
                        help="Push the resulting dataset to HuggingFace Hub")
    parser.add_argument("--repo", default=OUTPUT_REPO)
    parser.add_argument("--save-local", default=None,
                        help="Also save to this local path")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("Text-to-SQL Mix — Build Pipeline")
    log.info(f"Sources: {args.sources}")
    if args.sample:
        log.info(f"Sample mode: {args.sample} per source")
    log.info("=" * 70)

    all_examples: list[Example] = []
    for key in args.sources:
        try:
            all_examples.extend(SOURCES[key](args.sample))
        except Exception as e:
            log.error(f"Source '{key}' failed: {e}")

    log.info(f"Total raw: {len(all_examples):,}")

    all_examples = filter_and_enrich(all_examples)
    all_examples = deduplicate(all_examples)
    log.info(f"After filter+dedup: {len(all_examples):,}")

    splits = split_examples(all_examples)
    log.info(
        f"Split sizes: train={len(splits['train']):,}  "
        f"val={len(splits['validation']):,}  test={len(splits['test']):,}"
    )

    dsd = DatasetDict({k: to_hf_dataset(v) for k, v in splits.items()})

    stats = compute_stats(splits)

    if args.save_local:
        log.info(f"Saving locally to {args.save_local}")
        dsd.save_to_disk(args.save_local)

    if args.push:
        log.info(f"Pushing to {args.repo} (public)...")
        dsd.push_to_hub(args.repo, private=False)

        card = build_dataset_card(stats)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
            commit_message="Add dataset card with source attribution",
        )
        log.info(f"✅ Done. https://huggingface.co/datasets/{args.repo}")
    else:
        log.info("Skipping push (--push not set). Stats:")
        log.info(stats)


if __name__ == "__main__":
    main()

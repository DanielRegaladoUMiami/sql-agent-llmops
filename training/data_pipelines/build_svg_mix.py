# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=2.19",
#     "huggingface_hub>=0.23",
#     "matplotlib>=3.7",
#     "tqdm",
#     "requests",
# ]
# ///
"""
Build the SVG-rendering training mix for the SVG Renderer model.

The SVG Renderer (DeepSeek Coder 1.3B) takes a chart specification (JSON) and
emits inline SVG code that visually represents that chart. To train it well we
need clean ``(chart_spec → svg_code)`` pairs.

This pipeline assembles three complementary sources:

1. **synth-charts**  — programmatically render thousands of charts from
   nvBench's gold chart configs (real chart types, real data shapes) with
   matplotlib's SVG backend. Produces *perfect* (spec, svg) pairs.

2. **synth-augment** — synthetic data variations on top of the nvBench
   charts: change titles, sizes, colors, point counts. Multiplies the
   training signal without inventing new chart shapes.

3. **svgen-chartlike** — load ``umuthopeyildirim/svgen-500k`` and filter to
   *chart-shaped* SVGs (multi-element, has rect/line/group structure). These
   teach the model general SVG syntax fluency.

Common schema:

    {
      "id":         str,
      "chart_spec": {                 # input
          "chart_type":  str,
          "data":        [{"x":..., "y":..., "color":...}],
          "encoding":    {"x": "col", "y": "col", "color": "col|null"},
          "title":       str|null,
          "x_label":     str|null,
          "y_label":     str|null,
      },
      "svg_code":   str,              # target output
      "source":     str,
      "metadata":   {
          "chart_type":     str,
          "num_points":     int,
          "svg_size_bytes": int,
      }
    }

Stages:
  uv run build_svg_mix.py synth-charts --out data/svg_synth.jsonl
  uv run build_svg_mix.py svgen        --out data/svg_svgen.jsonl --max 15000
  uv run build_svg_mix.py combine-push --inputs data/svg_*.jsonl  --push
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import requests
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

NVBENCH_URL = (
    "https://raw.githubusercontent.com/TsinghuaDatabaseGroup/"
    "nvBench/main/NVBench.json"
)

DEFAULT_OUT_REPO = "DanielRegaladoCardoso/svg-chart-render-v1"
SVGEN_REPO = "umuthopeyildirim/svgen-500k"

SEED = 42
TRAIN_RATIO = 0.95
VAL_RATIO = 0.025

MAX_SVG_BYTES = 50_000  # drop pathological outputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("svg-mix")


# ----------------------------------------------------------------------------
# Common record
# ----------------------------------------------------------------------------

@dataclass
class SvgExample:
    id: str
    chart_spec: dict
    svg_code: str
    source: str
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


def make_id(source: str, key: str) -> str:
    h = hashlib.md5(f"{source}||{key}".encode()).hexdigest()
    return f"{source}-{h[:16]}"


# ----------------------------------------------------------------------------
# Stage 1 — synth-charts: render nvBench configs via matplotlib
# ----------------------------------------------------------------------------

NVBENCH_CHART_MAP = {
    "bar":              "bar",
    "stacked bar":      "bar",
    "grouping bar":     "bar",
    "line":             "line",
    "grouping line":    "line",
    "scatter":          "scatter",
    "grouping scatter": "scatter",
    "pie":              "donut",
    "histogram":        "histogram",
    "area":             "area",
}

# Tasteful, accessible default palette (Tableau 10-style)
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
           "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"]


def _ensure_nvbench(cache: Path) -> dict:
    if not cache.exists():
        log.info(f"Downloading nvBench from {NVBENCH_URL}...")
        cache.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(NVBENCH_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(cache, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
    return json.load(open(cache))


def _render_svg(chart_type: str, data: list[dict], title: str | None,
                x_label: str | None, y_label: str | None,
                color_idx: int = 0) -> str | None:
    """Render a chart with matplotlib and return SVG string, or None on failure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not data:
        return None

    xs = [d["x"] for d in data]
    ys = [d["y"] for d in data]
    colors_field = [d.get("color") for d in data]
    has_color = any(c is not None for c in colors_field)

    plt.rcParams["svg.fonttype"] = "none"  # text stays as text
    fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
    color = PALETTE[color_idx % len(PALETTE)]

    try:
        if chart_type == "bar":
            if has_color:
                groups = sorted(set(colors_field))
                width = 0.8 / max(1, len(groups))
                x_unique = list(dict.fromkeys(xs))
                x_pos = {x: i for i, x in enumerate(x_unique)}
                for gi, g in enumerate(groups):
                    g_xs = [x_pos[d["x"]] + gi * width for d in data
                            if d.get("color") == g]
                    g_ys = [d["y"] for d in data if d.get("color") == g]
                    ax.bar(g_xs, g_ys, width=width,
                           color=PALETTE[gi % len(PALETTE)], label=str(g))
                ax.set_xticks([i + width * (len(groups) - 1) / 2 for i in range(len(x_unique))])
                ax.set_xticklabels([str(x) for x in x_unique], rotation=30, ha="right")
                ax.legend(fontsize=8, frameon=False)
            else:
                ax.bar(range(len(xs)), ys, color=color)
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels([str(x) for x in xs], rotation=30, ha="right")
        elif chart_type == "line":
            if has_color:
                groups = sorted(set(colors_field))
                for gi, g in enumerate(groups):
                    g_xs = [d["x"] for d in data if d.get("color") == g]
                    g_ys = [d["y"] for d in data if d.get("color") == g]
                    ax.plot(g_xs, g_ys, marker="o", color=PALETTE[gi % len(PALETTE)],
                            label=str(g))
                ax.legend(fontsize=8, frameon=False)
            else:
                ax.plot(xs, ys, marker="o", color=color)
        elif chart_type == "scatter":
            if has_color:
                groups = sorted(set(colors_field))
                for gi, g in enumerate(groups):
                    g_xs = [d["x"] for d in data if d.get("color") == g]
                    g_ys = [d["y"] for d in data if d.get("color") == g]
                    ax.scatter(g_xs, g_ys, color=PALETTE[gi % len(PALETTE)],
                               label=str(g), s=40, alpha=0.8)
                ax.legend(fontsize=8, frameon=False)
            else:
                ax.scatter(xs, ys, color=color, s=40, alpha=0.8)
        elif chart_type == "donut":
            sizes = [abs(float(y)) for y in ys]
            if sum(sizes) == 0:
                return None
            wedges, _ = ax.pie(sizes, labels=[str(x) for x in xs],
                               colors=PALETTE, wedgeprops={"width": 0.45})
            ax.set_aspect("equal")
        elif chart_type == "histogram":
            ax.hist(ys, bins=min(20, max(5, len(ys) // 3)), color=color, edgecolor="white")
        elif chart_type == "area":
            ax.fill_between(range(len(xs)), ys, color=color, alpha=0.5)
            ax.plot(range(len(xs)), ys, color=color)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([str(x) for x in xs], rotation=30, ha="right")
        else:
            return None  # unsupported

        if title:
            ax.set_title(title, fontsize=11)
        if x_label and chart_type != "donut":
            ax.set_xlabel(x_label, fontsize=9)
        if y_label and chart_type != "donut":
            ax.set_ylabel(y_label, fontsize=9)
        if chart_type != "donut":
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        buf = io.StringIO()
        fig.tight_layout()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        svg = buf.getvalue()
        # Strip XML preamble for compactness
        svg = re.sub(r"^<\?xml[^>]+\?>\s*", "", svg)
        svg = re.sub(r"<!DOCTYPE[^>]+>\s*", "", svg)
        return svg
    except Exception:
        return None
    finally:
        plt.close(fig)


def synth_charts(out_path: Path, cache: Path, max_per_entry: int = 3,
                 augment_titles: bool = True) -> int:
    """Render nvBench chart configs to SVG via matplotlib."""
    raw = _ensure_nvbench(cache)
    log.info(f"Rendering charts from {len(raw):,} nvBench entries...")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    rendered_per_type = Counter()
    skipped = 0

    with open(out_path, "w") as f:
        for entry_id, entry in tqdm(raw.items(), desc="render"):
            try:
                vis = entry["vis_obj"]
                chart_raw = (vis.get("chart") or "").lower()
                chart_type = NVBENCH_CHART_MAP.get(chart_raw)
                if not chart_type:
                    skipped += 1
                    continue

                x_data = (vis.get("x_data") or [[]])[0]
                y_data = (vis.get("y_data") or [[]])[0]
                classify = vis.get("classify") or []
                if not x_data or not y_data:
                    skipped += 1
                    continue

                data = []
                for i in range(min(len(x_data), len(y_data))):
                    d = {"x": x_data[i], "y": y_data[i]}
                    if classify and i < len(classify):
                        d["color"] = classify[i]
                    data.append(d)

                x_name = vis.get("x_name", "x")
                y_name = vis.get("y_name", "y")

                # Title candidates: nl_queries (paraphrases) become titles
                titles = []
                if augment_titles:
                    titles = [None] + [q for q in entry.get("nl_queries", [])[:max_per_entry - 1]
                                       if q and len(q) < 100]
                else:
                    titles = [None]

                for ti, title in enumerate(titles):
                    svg = _render_svg(chart_type, data, title, x_name, y_name,
                                      color_idx=ti)
                    if not svg or len(svg) > MAX_SVG_BYTES:
                        continue

                    spec = {
                        "chart_type": chart_type,
                        "data": data,
                        "encoding": {
                            "x": x_name,
                            "y": y_name,
                            "color": "color" if classify else None,
                        },
                        "title": title,
                        "x_label": x_name,
                        "y_label": y_name,
                    }
                    ex = SvgExample(
                        id=make_id("synth-mpl", f"{entry_id}-{ti}"),
                        chart_spec=spec,
                        svg_code=svg,
                        source="synth-matplotlib",
                        metadata={
                            "chart_type": chart_type,
                            "num_points": len(data),
                            "svg_size_bytes": len(svg),
                        },
                    )
                    f.write(json.dumps(ex.to_dict()) + "\n")
                    written += 1
                    rendered_per_type[chart_type] += 1
            except Exception:
                skipped += 1
                continue

    log.info(f"synth-charts → {written:,} pairs   (skipped {skipped:,})")
    log.info(f"  by type: {dict(rendered_per_type)}")
    return written


# ----------------------------------------------------------------------------
# Stage 2 — svgen-chartlike: filter svgen-500k to chart-shaped SVGs
# ----------------------------------------------------------------------------

CHART_HINT_TAGS = ("<rect", "<line", "<g ", "<text")


def _is_chartlike(svg: str) -> bool:
    """Heuristic: chart SVGs have multiple element types, not just one path."""
    if not svg or len(svg) < 200 or len(svg) > MAX_SVG_BYTES:
        return False
    distinct = sum(1 for tag in CHART_HINT_TAGS if tag in svg)
    if distinct < 2:
        return False
    # Reject pure icons (single big path)
    paths = svg.count("<path")
    rects = svg.count("<rect")
    lines = svg.count("<line")
    if paths >= 1 and rects == 0 and lines == 0:
        return False
    return True


def svgen_chartlike(out_path: Path, max_n: int = 15000) -> int:
    """Stream svgen-500k, keep chart-shaped SVGs."""
    log.info(f"Streaming {SVGEN_REPO} and filtering to chart-like SVGs...")
    ds = load_dataset(SVGEN_REPO, split="train", streaming=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    seen = 0

    with open(out_path, "w") as f:
        for row in tqdm(ds, desc="svgen-filter"):
            seen += 1
            svg = row.get("output") or ""
            if not _is_chartlike(svg):
                continue
            label = row.get("input") or "chart"
            description = row.get("description") or ""
            spec = {
                "chart_type": "unknown",
                "data": [],
                "encoding": {"x": None, "y": None, "color": None},
                "title": label,
                "x_label": None,
                "y_label": None,
                "_freeform_description": description,
            }
            ex = SvgExample(
                id=make_id("svgen", f"{seen}"),
                chart_spec=spec,
                svg_code=svg,
                source=f"svgen500k-{(row.get('source') or 'unknown').replace(' ', '_')[:30]}",
                metadata={
                    "chart_type": "unknown",
                    "num_points": 0,
                    "svg_size_bytes": len(svg),
                },
            )
            f.write(json.dumps(ex.to_dict()) + "\n")
            kept += 1
            if kept >= max_n:
                break
    log.info(f"svgen-chartlike → kept {kept:,} / scanned {seen:,}")
    return kept


# ----------------------------------------------------------------------------
# Stage 3 — combine + push
# ----------------------------------------------------------------------------

def deduplicate(examples: list[SvgExample]) -> list[SvgExample]:
    seen = set()
    out = []
    for ex in examples:
        key = hashlib.md5(ex.svg_code.encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def split(examples: list[SvgExample]):
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


def to_hf_dataset(examples: list[SvgExample]) -> Dataset:
    """Serialize ``chart_spec`` and ``metadata`` as JSON strings to avoid Arrow
    type-inference errors (x-values can be str/int/float/date across rows)."""
    rows = []
    for e in examples:
        rows.append({
            "id": e.id,
            "chart_spec": json.dumps(e.chart_spec, ensure_ascii=False),
            "svg_code": e.svg_code,
            "source": e.source,
            "metadata": json.dumps(e.metadata, ensure_ascii=False),
        })
    return Dataset.from_list(rows)


def build_card(stats: dict, repo: str) -> str:
    L = []
    L.append("---")
    L.append("language:\n  - en")
    L.append("license: apache-2.0")
    L.append("task_categories:\n  - text-generation")
    L.append("tags:\n  - svg\n  - chart-rendering\n  - data-visualization\n  - code-generation\n  - chart-spec-to-svg")
    L.append("pretty_name: SVG Chart Render Mix v1")
    L.append("size_categories:\n  - 10K<n<100K")
    L.append("---")
    L.append("")
    L.append("# 🎨 SVG Chart Render Mix · v1")
    L.append("")
    L.append("Training data for fine-tuning a small code model (e.g. DeepSeek Coder 1.3B)")
    L.append("to map **`(chart specification JSON) → SVG code`**.")
    L.append("")
    L.append("> Powers the SVG Renderer in the [SQL Agent LLMOps]"
             "(https://github.com/DanielRegaladoUMiami/sql-agent-llmops) project.")
    L.append("")
    L.append(f"| 📦 Total | 🎨 Chart-spec inputs | 🖼 SVG outputs |")
    L.append(f"|---------|----------------------|----------------|")
    L.append(f"| **{stats['total']:,}** rows | structured JSON specs | rendered, validated SVG strings |")
    L.append("")
    L.append("## 📐 Schema")
    L.append("")
    L.append("```json")
    L.append(json.dumps({
        "id": "string",
        "chart_spec": {
            "chart_type": "bar|line|scatter|donut|histogram|area|unknown",
            "data": [{"x": "value", "y": "value", "color": "value|null"}],
            "encoding": {"x": "col", "y": "col", "color": "col|null"},
            "title": "string|null",
            "x_label": "string|null",
            "y_label": "string|null"
        },
        "svg_code": "string  (full <svg>…</svg>)",
        "source": "string",
        "metadata": {"chart_type": "string", "num_points": "int", "svg_size_bytes": "int"}
    }, indent=2))
    L.append("```")
    L.append("")
    L.append("## 📦 Splits")
    L.append("")
    L.append("| Split | Rows |")
    L.append("|-------|------|")
    for k, n in stats["splits"].items():
        L.append(f"| `{k}` | {n:,} |")
    L.append("")
    L.append("## 📊 Sources")
    L.append("")
    L.append("| Source | Rows | License | Notes |")
    L.append("|--------|------|---------|-------|")
    L.append("| `synth-matplotlib` (rendered from nvBench configs) | "
             f"{stats['by_source'].get('synth-matplotlib', 0):,} | Apache-2.0 | "
             "Programmatically rendered: real chart configs from nvBench, "
             "rendered with matplotlib (Agg/SVG backend). Perfect (spec, svg) pairs.|")
    svgen_total = sum(n for src, n in stats["by_source"].items() if src.startswith("svgen500k"))
    L.append(f"| `svgen500k-*` (filtered) | {svgen_total:,} | "
             "Per upstream | Subset of [`umuthopeyildirim/svgen-500k`]"
             "(https://huggingface.co/datasets/umuthopeyildirim/svgen-500k) "
             "filtered to chart-shaped SVGs (multi-element, has rect/line/group). "
             "Provides general SVG syntax fluency.|")
    L.append("")
    L.append("## 🛠 Pipeline")
    L.append("")
    L.append("1. **synth-charts** — load nvBench (Tsinghua DB Group) chart configs.")
    L.append("   For each entry, replay the chart in matplotlib (bar / line / scatter /")
    L.append("   donut / histogram / area), augment with each NL paraphrase as a")
    L.append("   candidate title, save as SVG.")
    L.append("2. **svgen-chartlike** — stream `umuthopeyildirim/svgen-500k`, keep only")
    L.append("   SVGs with chart-shaped structure (multiple `<rect>` / `<line>` / `<g>`).")
    L.append("3. **combine-push** — dedup by SVG hash, 95/2.5/2.5 split, push to HF.")
    L.append("")
    L.append("Build script: [`training/data_pipelines/build_svg_mix.py`]"
             "(https://github.com/DanielRegaladoUMiami/sql-agent-llmops/blob/main/training/data_pipelines/build_svg_mix.py)")
    L.append("")
    L.append("## 🚀 Usage")
    L.append("")
    L.append("```python")
    L.append("from datasets import load_dataset")
    L.append(f'ds = load_dataset("{repo}")')
    L.append("ex = ds['train'][0]")
    L.append("print(ex['chart_spec']['chart_type'])")
    L.append("print(ex['svg_code'][:200])")
    L.append("```")
    L.append("")
    L.append("### SFT format suggestion")
    L.append("")
    L.append("```python")
    L.append("def to_sft(row):")
    L.append("    return {")
    L.append('        "messages": [')
    L.append('            {"role": "system", "content": "You render chart specifications as inline SVG."},')
    L.append('            {"role": "user", "content": "Render this chart spec as SVG:\\n\\n" + json.dumps(row["chart_spec"])},')
    L.append('            {"role": "assistant", "content": row["svg_code"]},')
    L.append('        ]')
    L.append("    }")
    L.append("```")
    L.append("")
    L.append("## ⚠️ Limitations")
    L.append("")
    L.append("- SVGs from `synth-matplotlib` carry matplotlib's stylistic defaults")
    L.append("  (font, axes, ticks). The model trained on this dataset will produce")
    L.append("  matplotlib-flavored SVGs.")
    L.append("- `svgen-chartlike` rows have no structured `chart_spec` — only the")
    L.append("  free-form `title` and `_freeform_description` are populated.")
    L.append("- SVGs are bounded to ≤ 50 KB to keep training tractable.")
    L.append("")
    L.append("## 📝 Citation")
    L.append("")
    L.append("```bibtex")
    L.append("@dataset{regalado2026svgmix,")
    L.append("  author = {Regalado Cardoso, Daniel},")
    L.append("  title  = {SVG Chart Render Mix v1},")
    L.append("  year   = {2026},")
    L.append(f"  url    = {{https://huggingface.co/datasets/{repo}}}")
    L.append("}")
    L.append("```")
    L.append("")
    return "\n".join(L)


def combine_push(input_paths: list[Path], repo: str, push: bool, save_local: Path | None):
    examples: list[SvgExample] = []
    for p in input_paths:
        log.info(f"Reading {p}...")
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                examples.append(SvgExample(**d))
    log.info(f"Loaded {len(examples):,} total")

    examples = deduplicate(examples)
    log.info(f"After dedup: {len(examples):,}")

    splits = split(examples)
    log.info(f"Splits: train={len(splits['train']):,}  val={len(splits['validation']):,}  test={len(splits['test']):,}")

    dsd = DatasetDict({k: to_hf_dataset(v) for k, v in splits.items()})

    by_source = Counter(e.source for e in examples)
    by_type = Counter(e.metadata.get("chart_type", "unknown") for e in examples)
    stats = {
        "total": len(examples),
        "splits": {k: len(v) for k, v in splits.items()},
        "by_source": dict(by_source),
        "by_chart_type": dict(by_type),
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
    p = argparse.ArgumentParser(description="Build SVG-rendering training mix")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("synth-charts", help="Render nvBench charts via matplotlib")
    p1.add_argument("--out", type=Path, default=Path("data/svg_synth.jsonl"))
    p1.add_argument("--cache", type=Path, default=Path("data/cache/NVBench.json"))
    p1.add_argument("--max-per-entry", type=int, default=3,
                    help="Max title-augmented variants per nvBench entry")
    p1.add_argument("--no-augment", action="store_true")

    p2 = sub.add_parser("svgen", help="Filter svgen-500k to chart-like SVGs")
    p2.add_argument("--out", type=Path, default=Path("data/svg_svgen.jsonl"))
    p2.add_argument("--max", type=int, default=15000)

    p3 = sub.add_parser("combine-push", help="Combine + push to HF")
    p3.add_argument("--inputs", nargs="+", type=Path, required=True)
    p3.add_argument("--repo", default=DEFAULT_OUT_REPO)
    p3.add_argument("--push", action="store_true")
    p3.add_argument("--save-local", type=Path, default=None)

    args = p.parse_args()

    if args.cmd == "synth-charts":
        synth_charts(args.out, args.cache,
                     max_per_entry=args.max_per_entry,
                     augment_titles=not args.no_augment)
    elif args.cmd == "svgen":
        svgen_chartlike(args.out, args.max)
    elif args.cmd == "combine-push":
        combine_push(args.inputs, args.repo, args.push, args.save_local)


if __name__ == "__main__":
    main()

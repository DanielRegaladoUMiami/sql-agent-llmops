"""
Fill the null titles on nvBench rows with a meaningful placeholder derived
from the question + chart structure. Re-push the dataset as v1.1.

Rationale: leaving 72% of rows with title=null risks teaching Phi-3 to skip
titles. Using the NL question is a free, coherent upgrade — not storytelling
but a lot better than None.
"""
import json
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
from collections import Counter

REPO = "DanielRegaladoCardoso/chart-reasoning-mix-v1"


RATIONALE_TEMPLATES = {
    "bar":       "Bar chart compares {y} across categories of {x}.",
    "line":      "Line chart shows how {y} changes over {x}.",
    "scatter":   "Scatter plot reveals the relationship between {x} and {y}.",
    "donut":     "Donut chart shows the share of each {x} in total {y}.",
    "histogram": "Histogram shows the distribution of {y}.",
    "boxplot":   "Box plot compares the distribution of {y} across {x}.",
    "area":      "Area chart emphasizes cumulative {y} across {x}.",
    "heatmap":   "Heatmap displays {y} intensity across {x} dimensions.",
    "pie":       "Pie chart shows proportion of {y} across {x}.",
}


def _title_from_question(q: str) -> str:
    q = (q or "").strip()
    if not q:
        return "Chart overview"
    # Strip trailing punctuation and chart-type hints
    for noise in [" in a bar chart", " in a line chart", " in a scatter chart",
                  " in a pie chart", " on a bar chart", " using a bar chart",
                  ". Show", " Show bar chart", " Show me a",
                  "Show me ", "Show ", "show ", "Visualize "]:
        q = q.replace(noise, "")
    q = q.rstrip(".?! ")
    return q[:120]


def enrich(row):
    spec = json.loads(row["chart_spec"]) if isinstance(row["chart_spec"], str) else row["chart_spec"]
    changed = False

    # Only backfill where title is missing
    if not spec.get("title"):
        spec["title"] = _title_from_question(row["instruction"])
        changed = True

    if not spec.get("rationale"):
        ctype = (spec.get("chart_type") or "bar").lower()
        enc = spec.get("encoding") or {}
        x = enc.get("x") or "the x-axis"
        y = enc.get("y") or "the y-axis"
        tmpl = RATIONALE_TEMPLATES.get(ctype,
              "{ctype} chart visualizes {y} across {x}.").format(x=x, y=y, ctype=ctype)
        spec["rationale"] = tmpl
        changed = True

    row["chart_spec"] = json.dumps(spec, ensure_ascii=False)
    return row


print("Loading dataset...")
ds = load_dataset(REPO)

print("Enriching splits...")
enriched = DatasetDict({
    k: v.map(enrich, desc=f"enrich-{k}") for k, v in ds.items()
})

# Quick audit
sample = enriched["train"][0]
print("\n=== Sample after enrichment ===")
print("instruction:", sample["instruction"][:100])
print("source    :", sample["source"])
spec = json.loads(sample["chart_spec"])
print("title     :", spec.get("title"))
print("rationale :", spec.get("rationale"))

# Count how many had null titles (audit)
null_before = sum(1 for r in ds["train"] if not json.loads(r["chart_spec"]).get("title"))
null_after  = sum(1 for r in enriched["train"] if not json.loads(r["chart_spec"]).get("title"))
print(f"\nNull titles before: {null_before:,}")
print(f"Null titles after : {null_after:,}")

print("\nPushing to HF (overwrite v1)...")
enriched.push_to_hub(REPO, private=False)
print(f"✅ Done: https://huggingface.co/datasets/{REPO}")

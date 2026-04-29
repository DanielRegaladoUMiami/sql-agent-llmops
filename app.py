"""
SQL Agent — Gradio app for Hugging Face Spaces (ZeroGPU).

Apple x Claude minimalist design with progressive feedback during the
multi-step pipeline.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Generator, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

import gradio as gr  # noqa: E402

try:
    import spaces  # type: ignore
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

    class _SpacesShim:
        @staticmethod
        def GPU(duration: int = 60):  # noqa: N802
            def decorator(fn):
                return fn
            return decorator

    spaces = _SpacesShim()  # type: ignore

# CRITICAL: load all 3 models on cuda at module level per ZeroGPU best
# practice. PyTorch CUDA emulation handles this when no real GPU is present;
# inside @spaces.GPU calls, the real GPU is used and inference is fast.
logger.info("Loading models at module level...")
from src.models.sql_generator import SQLGenerator  # noqa: E402
from src.models.chart_reasoner import ChartReasoner  # noqa: E402
from src.models.svg_renderer import SVGRenderer  # noqa: E402
from src.orchestrator.pipeline import SQLAgentOrchestrator  # noqa: E402

_SQL_GEN = SQLGenerator()
_CHART_REASONER = ChartReasoner()
_SVG_RENDERER = SVGRenderer()
logger.info("All models loaded")


# ============================================================ THEME / CSS
THEME_CSS = """
:root {
  --ink: #0E0E0E;
  --ink-muted: #5A5A5A;
  --ink-faint: #E5E5E5;
  --surface: #FAFAF9;
  --surface-raised: #FFFFFF;
  --accent: #C96442;
  --accent-soft: rgba(201, 100, 66, 0.08);
  --radius: 16px;
  --radius-sm: 10px;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md: 0 6px 24px rgba(0,0,0,0.08);
  --font: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display",
          "Helvetica Neue", Arial, sans-serif;
  --font-mono: "SF Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

@media (prefers-color-scheme: dark) {
  :root {
    --ink: #F4F4F2;
    --ink-muted: #8A8A8A;
    --ink-faint: #2A2A2A;
    --surface: #0E0E0E;
    --surface-raised: #161616;
    --accent: #E8866A;
    --accent-soft: rgba(232, 134, 106, 0.10);
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.25);
    --shadow-md: 0 6px 24px rgba(0,0,0,0.45);
  }
}

/* Gradio container reset (Gradio 5 selectors) */
gradio-app, .gradio-container, .main, .app, .contain, .wrap {
  background: var(--surface) !important;
  color: var(--ink) !important;
  font-family: var(--font) !important;
}
.gradio-container {
  max-width: 1600px !important;
  width: 100% !important;
  margin: 0 auto !important;
  padding: 32px 40px 60px !important;
  min-height: 920px !important;
  box-sizing: border-box;
}

/* Two-column rectangular layout — 16:9-ish aspect, fixed feel */
.split-layout {
  display: grid;
  grid-template-columns: minmax(360px, 460px) 1fr;
  gap: 36px;
  align-items: start;
  min-height: 820px;
}
@media (max-width: 900px) {
  .split-layout { grid-template-columns: 1fr; gap: 22px; min-height: auto; }
  .gradio-container { padding: 24px 18px 60px !important; min-height: auto !important; }
}
.split-left { position: sticky; top: 24px; }
.split-right { min-height: 720px; display: flex; flex-direction: column; }
.split-right > .panel-label { flex-shrink: 0; }

.panel-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--ink-muted) !important;
  margin: 0 0 10px 2px;
}
footer { display: none !important; }
.show-api { display: none !important; }

/* Header */
.app-header {
  margin-bottom: 28px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--ink-faint);
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 16px;
}
.app-title { font-size: 18px; font-weight: 600; letter-spacing: -0.015em; color: var(--ink) !important; }
.app-subtitle { font-size: 13px; color: var(--ink-muted) !important; }

/* HF Login button — make it Apple-style */
[data-testid="login-button"], .login-button, button[aria-label*="login"], button[aria-label*="Login"] {
  background: var(--ink) !important;
  color: var(--surface) !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 500 !important;
  padding: 8px 14px !important;
  font-size: 13px !important;
  margin-bottom: 18px !important;
}

/* File upload — compact, Apple-style */
.upload-row { margin-bottom: 14px; }
.upload-row .gr-file, .upload-row .file-preview { background: transparent !important; }
.upload-row [data-testid="file"] {
  border: 1.5px dashed var(--ink-faint) !important;
  border-radius: var(--radius) !important;
  padding: 20px 16px !important;
  background: transparent !important;
  transition: all 200ms ease !important;
  min-height: 90px !important;
}
.upload-row [data-testid="file"]:hover {
  border-color: var(--accent) !important;
  background: var(--accent-soft) !important;
}
.upload-row [data-testid="file"] *,
.upload-row .upload-text,
.upload-row svg { color: var(--ink-muted) !important; fill: var(--ink-muted) !important; }
.upload-row .file-preview, .upload-row [class*="FilePreview"] { display: none !important; }

/* File chip (after upload) */
.file-chip {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 8px 14px 8px 12px;
  background: var(--surface-raised);
  border: 1px solid var(--ink-faint);
  border-radius: 999px;
  font-size: 13px;
  color: var(--ink) !important;
}
.file-chip span { color: var(--ink) !important; }
.file-chip-meta { color: var(--ink-muted) !important; font-size: 12px; }
.file-chip-dot {
  width: 6px; height: 6px;
  background: var(--accent);
  border-radius: 50%;
  flex-shrink: 0;
}

/* Hide the giant gr.File "uploaded file" display — we have our own chip */
.upload-row [data-testid="file"] .file-preview,
.upload-row .file-preview-holder {
  display: none !important;
}

/* Question input */
.question-row { margin: 14px 0 8px; }
textarea, .gr-text-input textarea, [data-testid="textbox"] textarea {
  background: var(--surface-raised) !important;
  border: 1px solid var(--ink-faint) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--ink) !important;
  font-family: var(--font) !important;
  font-size: 15px !important;
  padding: 14px 16px !important;
  box-shadow: none !important;
  transition: border-color 150ms ease !important;
  line-height: 1.5 !important;
}
textarea:focus, [data-testid="textbox"] textarea:focus {
  border-color: var(--accent) !important;
  outline: none !important;
  box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
textarea::placeholder { color: var(--ink-muted) !important; }
.kb-hint {
  font-size: 11px;
  color: var(--ink-muted) !important;
  margin: 4px 4px 0;
  background: transparent !important;
  padding: 0 !important;
}

/* Question group container — make sure it's not boxed/dark */
.question-row, .question-row > div, .question-row .gr-block, .question-row .gr-form {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}

/* Buttons */
button.primary, button[variant="primary"], .gr-button.primary {
  background: var(--ink) !important;
  color: var(--surface) !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--font) !important;
  font-weight: 500 !important;
  font-size: 14px !important;
  padding: 10px 18px !important;
  transition: opacity 150ms ease !important;
  box-shadow: none !important;
}
button.primary:hover { opacity: 0.85 !important; }
button.secondary, button[variant="secondary"] {
  background: transparent !important;
  color: var(--ink) !important;
  border: 1px solid var(--ink-faint) !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 500 !important;
  padding: 10px 18px !important;
}

/* Conversation */
.turn { margin: 32px 0; }
.turn:first-child { margin-top: 16px; }
.turn-question {
  font-size: 16px;
  color: var(--ink);
  font-weight: 500;
  margin-bottom: 14px;
  letter-spacing: -0.01em;
  line-height: 1.5;
}
.turn-progress {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
  color: var(--ink-muted);
  padding: 12px 16px;
  background: var(--surface-raised);
  border: 1px solid var(--ink-faint);
  border-radius: var(--radius-sm);
  margin: 6px 0;
}
.turn-progress::before {
  content: "";
  width: 8px; height: 8px;
  background: var(--accent);
  border-radius: 50%;
  animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 0.3; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.3); }
}
.turn-error {
  background: var(--accent-soft);
  border-left: 3px solid var(--accent);
  color: var(--accent);
  padding: 12px 14px;
  border-radius: var(--radius-sm);
  font-size: 13px;
  margin: 6px 0;
  font-family: var(--font-mono);
}

.chart-wrap {
  background: var(--surface-raised);
  border: 1px solid var(--ink-faint);
  border-radius: var(--radius);
  padding: 24px;
  margin: 8px 0 14px;
  box-shadow: var(--shadow-sm);
}
.chart-wrap svg { width: 100% !important; height: auto !important; display: block; }

/* Code blocks */
.sql-block {
  background: var(--surface-raised);
  border: 1px solid var(--ink-faint);
  border-radius: var(--radius-sm);
  font-family: var(--font-mono);
  font-size: 12.5px;
  color: var(--ink);
  padding: 14px 16px;
  overflow-x: auto;
  white-space: pre-wrap;
  margin: 6px 0 0;
  line-height: 1.6;
}

/* Details / collapsibles */
details {
  margin: 8px 0;
  border: 1px solid var(--ink-faint);
  border-radius: var(--radius-sm);
  background: var(--surface-raised);
}
details summary {
  cursor: pointer;
  padding: 10px 14px;
  font-size: 12.5px;
  color: var(--ink-muted);
  list-style: none;
  user-select: none;
  font-weight: 500;
}
details summary::-webkit-details-marker { display: none; }
details summary::before {
  content: "›";
  display: inline-block;
  width: 12px;
  margin-right: 4px;
  transition: transform 150ms ease;
  color: var(--ink-muted);
}
details[open] summary::before { transform: rotate(90deg); }
details > *:not(summary) { padding: 0 14px 14px; }

/* Data table */
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  font-family: var(--font);
}
.data-table th {
  text-align: left;
  font-weight: 600;
  color: var(--ink);
  padding: 8px 10px;
  border-bottom: 1px solid var(--ink-faint);
  white-space: nowrap;
}
.data-table td {
  padding: 7px 10px;
  color: var(--ink-muted);
  border-bottom: 1px solid var(--ink-faint);
}
.data-table tr:last-child td { border-bottom: none; }
.data-table-meta { font-size: 11px; color: var(--ink-muted); margin-top: 8px; padding: 0 4px; }

/* Empty state */
.empty {
  padding: 40px 0 8px;
  text-align: center;
}
.empty-title {
  font-size: 15px;
  color: var(--ink);
  font-weight: 500;
  margin-bottom: 6px;
}
.empty-sub {
  font-size: 13px;
  color: var(--ink-muted);
  margin-bottom: 28px;
}
.example-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
  max-width: 580px;
  margin: 0 auto;
}
.example-card {
  text-align: left;
  padding: 14px 16px;
  background: var(--surface-raised);
  border: 1px solid var(--ink-faint);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: all 150ms ease;
}
.example-card:hover {
  border-color: var(--accent);
  background: var(--accent-soft);
  transform: translateY(-1px);
}
.example-card-title {
  font-size: 13px;
  font-weight: 500;
  color: var(--ink);
  margin-bottom: 4px;
}
.example-card-meta {
  font-size: 11px;
  color: var(--ink-muted);
}

/* Suggestions */
.suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 14px 0 0;
}
.suggestion-chip {
  font-size: 12px;
  padding: 6px 12px;
  background: var(--surface-raised);
  border: 1px solid var(--ink-faint);
  border-radius: 999px;
  cursor: pointer;
  color: var(--ink-muted);
  transition: all 150ms ease;
}
.suggestion-chip:hover {
  border-color: var(--accent);
  color: var(--ink);
}

/* Hide labels Gradio adds */
.gr-form > label, label.svelte-1gfkn6j, .label-wrap { display: none !important; }

/* Narration — analyst-style finding under the chart */
.narration {
  margin: 12px 2px 4px;
  font-size: 14px;
  line-height: 1.55;
  color: var(--ink) !important;
  letter-spacing: -0.005em;
  padding-left: 12px;
  border-left: 2px solid var(--accent);
}

/* Download links below chart */
.downloads {
  display: flex;
  gap: 6px;
  margin: 8px 0 4px;
  flex-wrap: wrap;
}
.download-link {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  padding: 6px 12px;
  background: var(--surface-raised);
  border: 1px solid var(--ink-faint);
  border-radius: var(--radius-sm);
  color: var(--ink-muted);
  text-decoration: none;
  transition: all 150ms ease;
  cursor: pointer;
}
.download-link:hover {
  border-color: var(--accent);
  color: var(--ink);
  background: var(--accent-soft);
}
.download-link .icon {
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1;
}

/* Schema preview after upload */
.schema-preview {
  margin: 14px 0;
  padding: 14px 16px;
  background: var(--surface-raised);
  border: 1px solid var(--ink-faint);
  border-radius: var(--radius-sm);
}
.schema-preview-header {
  font-size: 12px;
  font-weight: 600;
  color: var(--ink);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 10px;
}
.schema-cols {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.schema-col {
  display: inline-flex;
  align-items: baseline;
  gap: 5px;
  padding: 4px 10px;
  background: var(--surface);
  border: 1px solid var(--ink-faint);
  border-radius: 6px;
  font-size: 12px;
}
.schema-col-name { color: var(--ink); font-family: var(--font-mono); font-size: 12px; }
.schema-col-type { color: var(--ink-muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em; }
"""


# ===================================================== ORCHESTRATOR (lazy)
_AGENT: Optional[SQLAgentOrchestrator] = None


def get_agent() -> SQLAgentOrchestrator:
    global _AGENT
    if _AGENT is None:
        _AGENT = SQLAgentOrchestrator(_SQL_GEN, _CHART_REASONER, _SVG_RENDERER)
    return _AGENT


# =================================================== EXAMPLE DATA (built-in)
def _make_titanic_csv() -> Path:
    """Tiny embedded Titanic-like sample so first-time users can play with no upload."""
    p = ROOT / "_examples" / "titanic.csv"
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "passenger_id": range(1, 21),
        "survived": [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        "pclass": [3, 1, 3, 1, 3, 3, 1, 3, 3, 2, 3, 1, 3, 3, 3, 2, 3, 2, 3, 3],
        "sex": ["male","female","female","female","male","male","male","male","female","female",
                "female","female","male","male","female","female","male","female","male","female"],
        "age": [22,38,26,35,35,None,54,2,27,14,4,58,20,39,14,55,2,None,31,None],
        "fare": [7.25,71.28,7.92,53.10,8.05,8.46,51.86,21.07,11.13,30.07,
                 16.70,26.55,8.05,31.27,7.85,16.00,29.13,13.00,18.00,7.23],
        "embarked": ["S","C","S","S","S","Q","S","S","S","C","S","S","S","S","Q","S","Q","S","S","Q"],
    })
    df.to_csv(p, index=False)
    return p


def _suggest_questions(table: str, schema: list[dict]) -> list[str]:
    """Generate question suggestions tailored to the loaded dataset's columns."""
    if not schema:
        return []

    NUMERIC = {"INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "REAL", "INT", "SMALLINT"}
    DATE = {"DATE", "TIMESTAMP", "DATETIME", "TIME"}
    STRING = {"VARCHAR", "STRING", "TEXT", "CHAR"}

    def kind(t: str) -> str:
        t = (t or "").upper().split("(")[0]
        if any(k in t for k in NUMERIC): return "num"
        if any(k in t for k in DATE):    return "date"
        if any(k in t for k in STRING):  return "str"
        return "other"

    cols = [(c["name"], kind(c.get("type", ""))) for c in schema]
    nums = [n for n, k in cols if k == "num"]
    dates = [n for n, k in cols if k == "date"]
    strs = [n for n, k in cols if k == "str"]

    qs: list[str] = []
    if nums:
        qs.append(f"Top 10 rows by {nums[0]}")
    if strs and nums:
        qs.append(f"{nums[0].capitalize()} grouped by {strs[0]}")
    if strs:
        qs.append(f"Count of rows by {strs[0]}")
    if dates and nums:
        qs.append(f"{nums[0].capitalize()} over time ({dates[0]})")
    if len(nums) >= 2:
        qs.append(f"Compare {nums[0]} vs {nums[1]}")
    if not qs:
        qs.append(f"Show me the first 10 rows of {table}")

    return qs[:4]


# =================================================== HTML render helpers
def _file_chip_html(filename: str, rows: int, cols: int) -> str:
    return (
        '<div class="file-chip">'
        '<span class="file-chip-dot"></span>'
        f'<span>{filename}</span>'
        f'<span class="file-chip-meta">{rows:,} rows · {cols} cols</span>'
        '</div>'
    )


def _schema_preview_html(table: str, schema: list[dict]) -> str:
    """Render the column names + types of the loaded table."""
    if not schema:
        return ""
    cols = "".join(
        f'<span class="schema-col">'
        f'<span class="schema-col-name">{c["name"]}</span>'
        f'<span class="schema-col-type">{c["type"]}</span>'
        f'</span>'
        for c in schema
    )
    return (
        '<div class="schema-preview">'
        f'<div class="schema-preview-header">{table} · {len(schema)} columns</div>'
        f'<div class="schema-cols">{cols}</div>'
        '</div>'
    )


def _download_links_html(sql: str, results: list[dict], svg: str) -> str:
    """Inline data-URL download links for CSV (results) and SVG (chart)."""
    import base64
    import csv
    import io as _io

    parts = []

    # CSV download
    if results:
        buf = _io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results:
            writer.writerow({k: ("" if v is None else v) for k, v in r.items()})
        csv_b64 = base64.b64encode(buf.getvalue().encode("utf-8")).decode("ascii")
        parts.append(
            f'<a class="download-link" '
            f'href="data:text/csv;base64,{csv_b64}" '
            f'download="query-result.csv">'
            f'<span class="icon">↓</span> CSV ({len(results):,} rows)</a>'
        )

    # SVG download (standalone version: explicit dims, white bg, XML prolog)
    if svg and "<svg" in svg.lower():
        from src.visualization.svg_theme import to_standalone_svg
        standalone = to_standalone_svg(svg)
        svg_b64 = base64.b64encode(standalone.encode("utf-8")).decode("ascii")
        parts.append(
            f'<a class="download-link" '
            f'href="data:image/svg+xml;base64,{svg_b64}" '
            f'download="chart.svg">'
            f'<span class="icon">↓</span> SVG</a>'
        )

    if not parts:
        return ""
    return f'<div class="downloads">{"".join(parts)}</div>'


def _suggestions_html(qs: list[str]) -> str:
    if not qs:
        return ""
    chips = "".join(
        f'<span class="suggestion-chip" onclick="document.querySelector(\'textarea\').value=this.textContent;document.querySelector(\'textarea\').focus();">{q}</span>'
        for q in qs
    )
    return f'<div class="suggestions">{chips}</div>'


def _data_table_html(rows: list[dict], max_rows: int = 10) -> str:
    if not rows:
        return '<div class="empty-sub" style="padding:8px 0">No rows.</div>'
    df = pd.DataFrame(rows[:max_rows])
    cols = df.columns.tolist()
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{('' if r.get(c) is None else r.get(c, ''))}</td>" for c in cols
        ) + "</tr>"
        for r in rows[:max_rows]
    )
    note = (
        f'<div class="data-table-meta">Showing {min(max_rows, len(rows))} of {len(rows):,} rows</div>'
        if len(rows) > max_rows else ""
    )
    return f'<table class="data-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{note}'


def _turn_html_progress(question: str, status: str) -> str:
    """Render a turn that's still in progress (with animated indicator)."""
    return (
        '<div class="turn">'
        f'<div class="turn-question">{question}</div>'
        f'<div class="turn-progress">{status}</div>'
        '</div>'
    )


def _turn_html_complete(result: dict) -> str:
    """Render a finished turn."""
    parts: list[str] = [f'<div class="turn-question">{result["question"]}</div>']

    if result.get("error"):
        parts.append(f'<div class="turn-error">{result["error"]}</div>')

    if result.get("svg"):
        parts.append(f'<div class="chart-wrap">{result["svg"]}</div>')

    # Narration: 1-2 sentence finding from the analyst persona
    if result.get("narration"):
        parts.append(f'<div class="narration">{result["narration"]}</div>')

    # Inline download links for CSV + SVG
    parts.append(_download_links_html(
        result.get("sql") or "",
        result.get("results") or [],
        result.get("svg") or "",
    ))

    if result.get("sql"):
        parts.append(
            '<details><summary>SQL query</summary>'
            f'<pre class="sql-block">{result["sql"]}</pre>'
            '</details>'
        )

    if result.get("results"):
        parts.append(
            '<details><summary>Data</summary>'
            f'{_data_table_html(result["results"])}'
            '</details>'
        )

    return f'<div class="turn">{"".join(parts)}</div>'


def _conversation_html(history: list[dict], in_progress: tuple[str, str] | None = None) -> str:
    """Conversation HTML reacts to: data loaded? history? in-progress?"""
    has_data = bool(get_agent().list_tables())
    has_turns = bool(history) or in_progress is not None

    if not has_data and not has_turns:
        return _empty_state_html()
    if has_data and not has_turns:
        return _ready_state_html()

    out = "".join(_turn_html_complete(t) for t in history)
    if in_progress:
        out += _turn_html_progress(in_progress[0], in_progress[1])
    return out


def _empty_state_html() -> str:
    return (
        '<div class="empty">'
        '<div class="empty-title">No data loaded yet</div>'
        '<div class="empty-sub">Drop a CSV, JSON, Parquet or Excel file above, '
        'or click <strong>Load demo dataset</strong> to play with sample data.</div>'
        '</div>'
    )


def _ready_state_html() -> str:
    """Shown when data is loaded but no queries asked yet.
    Suggestions are derived from the actual loaded table's columns."""
    agent = get_agent()
    tables = agent.list_tables()
    suggestions: list[str] = []
    if tables:
        schema = agent.executor.get_table_schema(tables[0])
        suggestions = _suggest_questions(tables[0], schema)
    return (
        '<div class="empty">'
        '<div class="empty-title">Ready</div>'
        '<div class="empty-sub">Ask a question above, or try one of these:</div>'
        f'{_suggestions_html(suggestions)}'
        '</div>'
    )


# ============================================================ EVENT HANDLERS
def _build_schema_html(table: str) -> str:
    agent = get_agent()
    schema = agent.executor.get_table_schema(table)
    return _schema_preview_html(table, schema)


def on_upload(file):
    """Returns: (chip, schema_html, conversation, history_state)."""
    if file is None:
        return "", "", _conversation_html([]), []
    agent = get_agent()
    agent.reset()
    try:
        path = Path(file.name if hasattr(file, "name") else file)
        table = agent.load_data(path)
        rows = agent.executor.con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        cols = len(agent.executor.get_table_schema(table))
        chip = _file_chip_html(path.name, rows, cols)
        schema = _build_schema_html(table)
        return chip, schema, _conversation_html([]), []
    except Exception as e:
        logger.exception("upload failed")
        return "", "", f'<div class="turn-error">Could not load file: {e}</div>', []


def on_load_demo():
    """Returns: (chip, schema_html, conversation, history_state)."""
    agent = get_agent()
    agent.reset()
    try:
        p = _make_titanic_csv()
        table = agent.load_data(p)
        rows = agent.executor.con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        cols = len(agent.executor.get_table_schema(table))
        chip = _file_chip_html("titanic.csv (demo)", rows, cols)
        schema = _build_schema_html(table)
        return chip, schema, _conversation_html([]), []
    except Exception as e:
        logger.exception("demo load failed")
        return "", "", f'<div class="turn-error">Could not load demo: {e}</div>', []


@spaces.GPU(duration=60)
def _gpu_process(question: str) -> dict:
    """Inference only — models already on cuda from module-level loading."""
    agent = get_agent()
    return agent.process(question)


def on_ask(question: str, history: list) -> Generator[Tuple[str, str, list], None, None]:
    """
    Generator: yields conversation HTML at each pipeline step so the user
    sees real-time progress instead of waiting silently.
    """
    # Single-shot mode: each Ask is a fresh query, replaces any previous result
    question = (question or "").strip()
    if not question:
        yield _conversation_html([]), "", []
        return

    if not get_agent().list_tables():
        result = {
            "question": question,
            "error": "Upload a file first or load the demo dataset.",
        }
        yield _conversation_html([result]), "", [result]
        return

    # First yield: show the question with progress indicator (no past history)
    yield _conversation_html([], in_progress=(question, "Generating SQL…")), "", []

    try:
        result = _gpu_process(question)
    except Exception as e:
        logger.exception("ask failed")
        result = {"question": question, "error": str(e)}

    # Replace history with just this single result
    yield _conversation_html([result]), "", [result]


def on_reset():
    """Clear ONLY the dataset (file, chip, schema, agent state).
    Also wipes the displayed result since it's now stale without data.
    Order: upload, chip_html, schema_html, conversation, history_state
    """
    get_agent().reset()
    return None, "", "", _conversation_html([]), []


# ====================================================================== APP
def build_app() -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Base(),
        css=THEME_CSS,
        title="SQL Agent",
        analytics_enabled=False,
    ) as demo:
        # Header
        gr.HTML(
            '<div class="app-header">'
            '<div>'
            '<div class="app-title">SQL Agent</div>'
            '<div class="app-subtitle">Ask anything about your data.</div>'
            '</div>'
            '</div>'
        )

        # Two-column layout: LEFT = data/question controls, RIGHT = results
        with gr.Row(elem_classes=["split-layout"]):
            # ---------- LEFT panel ----------
            with gr.Column(elem_classes=["split-left"], scale=0):
                gr.HTML('<div class="panel-label">Dataset</div>')
                with gr.Row(elem_classes=["upload-row"]):
                    upload = gr.File(
                        label="",
                        file_types=[".csv", ".json", ".parquet", ".xlsx", ".xls"],
                        show_label=False,
                        container=False,
                    )
                chip_html = gr.HTML("")
                schema_html = gr.HTML("")

                gr.HTML('<div class="panel-label" style="margin-top:18px">Question</div>')
                with gr.Group(elem_classes=["question-row"]):
                    question = gr.Textbox(
                        placeholder="Ask anything about your data…",
                        lines=3,
                        max_lines=10,
                        show_label=False,
                        container=False,
                        autofocus=True,
                    )
                    gr.HTML('<div class="kb-hint">Press Enter to send · Shift+Enter for newline</div>')

                with gr.Row():
                    ask_btn = gr.Button("Ask", variant="primary", size="sm")
                    reset_btn = gr.Button("Clear", variant="secondary", size="sm")
                    demo_btn = gr.Button("Demo", variant="secondary", size="sm",
                                          elem_id="load_demo_btn")

            # ---------- RIGHT panel ----------
            with gr.Column(elem_classes=["split-right"], scale=1):
                gr.HTML('<div class="panel-label">Result</div>')
                history_state = gr.State([])
                conversation = gr.HTML(_conversation_html([]))

        # ------------- events -------------
        upload.upload(
            fn=on_upload,
            inputs=upload,
            outputs=[chip_html, schema_html, conversation, history_state],
            api_name=False,
        )
        upload.clear(
            fn=lambda: ("", "", _conversation_html([]), []),
            outputs=[chip_html, schema_html, conversation, history_state],
            api_name=False,
        )
        demo_btn.click(
            fn=on_load_demo,
            outputs=[chip_html, schema_html, conversation, history_state],
            api_name=False,
        )
        ask_btn.click(
            fn=on_ask,
            inputs=[question, history_state],
            outputs=[conversation, question, history_state],
            api_name=False,
        )
        question.submit(
            fn=on_ask,
            inputs=[question, history_state],
            outputs=[conversation, question, history_state],
            api_name=False,
        )
        reset_btn.click(
            fn=on_reset,
            outputs=[upload, chip_html, schema_html, conversation, history_state],
            api_name=False,
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue(api_open=False).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_api=False,
    )

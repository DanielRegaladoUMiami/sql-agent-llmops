"""
SQL Agent — Gradio app for Hugging Face Spaces (ZeroGPU).

Apple x Claude minimalist design: single column, generous whitespace, warm
accent, monochrome ink, SF font stack, no chrome.

Entry point file at the repo root because that's the HF Spaces convention.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# Ensure src/ is importable when run from repo root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from src.orchestrator.pipeline import SQLAgentOrchestrator  # noqa: E402

# spaces is a Hugging Face SDK that exposes @spaces.GPU. It only exists on
# HF Spaces with ZeroGPU enabled, so we degrade gracefully when running
# locally.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- styling
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
  --shadow-md: 0 4px 16px rgba(0,0,0,0.06);
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
    --shadow-md: 0 4px 16px rgba(0,0,0,0.35);
  }
}

/* Reset Gradio chrome */
.gradio-container {
  max-width: 760px !important;
  margin: 0 auto !important;
  padding: 32px 24px 80px !important;
  background: var(--surface) !important;
  font-family: var(--font) !important;
  color: var(--ink) !important;
}
body, .gradio-container, .main, footer { background: var(--surface) !important; }
footer { display: none !important; }

/* Header */
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 32px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--ink-faint);
}
.app-title {
  font-size: 17px;
  font-weight: 600;
  letter-spacing: -0.015em;
  color: var(--ink);
}
.app-subtitle {
  font-size: 13px;
  color: var(--ink-muted);
  margin-top: 2px;
}

/* Cards */
.card {
  background: var(--surface-raised);
  border: 1px solid var(--ink-faint);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: 16px;
  box-shadow: var(--shadow-sm);
  transition: box-shadow 200ms ease, border-color 200ms ease;
}
.card:hover { box-shadow: var(--shadow-md); }

/* Upload area */
.upload-zone {
  border: 1.5px dashed var(--ink-faint) !important;
  border-radius: var(--radius) !important;
  background: transparent !important;
  padding: 32px 20px !important;
  text-align: center !important;
  transition: border-color 200ms ease, background 200ms ease !important;
}
.upload-zone:hover {
  border-color: var(--accent) !important;
  background: var(--accent-soft) !important;
}
.upload-zone .icon-wrap, .upload-zone svg { color: var(--ink-muted) !important; }

/* Inputs */
input, textarea, .gr-input, .gr-text-input textarea {
  background: var(--surface-raised) !important;
  border: 1px solid var(--ink-faint) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--ink) !important;
  font-family: var(--font) !important;
  font-size: 15px !important;
  padding: 14px 16px !important;
  box-shadow: none !important;
  transition: border-color 150ms ease !important;
}
input:focus, textarea:focus { border-color: var(--accent) !important; outline: none !important; }
::placeholder { color: var(--ink-muted) !important; }

/* Buttons */
button.primary, .gr-button-primary {
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
button.primary:hover, .gr-button-primary:hover { opacity: 0.85 !important; }
button.secondary, .gr-button-secondary {
  background: transparent !important;
  color: var(--ink) !important;
  border: 1px solid var(--ink-faint) !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 500 !important;
  padding: 10px 18px !important;
}

/* Conversation */
.turn { margin: 28px 0; }
.turn-question {
  font-size: 16px;
  color: var(--ink);
  font-weight: 500;
  margin-bottom: 14px;
  letter-spacing: -0.01em;
}
.turn-status {
  font-size: 13px;
  color: var(--ink-muted);
  font-style: italic;
  margin: 6px 0 12px;
}
.turn-error {
  background: var(--accent-soft);
  border-left: 2px solid var(--accent);
  color: var(--accent);
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  font-size: 13px;
  margin: 6px 0;
}
.chart-wrap {
  background: var(--surface-raised);
  border: 1px solid var(--ink-faint);
  border-radius: var(--radius);
  padding: 24px;
  margin: 8px 0 12px;
  box-shadow: var(--shadow-sm);
}
.chart-wrap svg { width: 100% !important; height: auto !important; display: block; }

/* Code blocks */
.sql-block {
  background: var(--surface-raised);
  border: 1px solid var(--ink-faint);
  border-radius: var(--radius-sm);
  font-family: var(--font-mono);
  font-size: 13px;
  color: var(--ink);
  padding: 14px 16px;
  overflow-x: auto;
  white-space: pre;
  margin: 6px 0;
  line-height: 1.55;
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
  font-size: 13px;
  color: var(--ink-muted);
  list-style: none;
  user-select: none;
  font-weight: 500;
}
details summary::-webkit-details-marker { display: none; }
details summary::before {
  content: "›";
  display: inline-block;
  margin-right: 8px;
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
}
.data-table td {
  padding: 7px 10px;
  color: var(--ink-muted);
  border-bottom: 1px solid var(--ink-faint);
}
.data-table tr:last-child td { border-bottom: none; }

/* File chip */
.file-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 14px;
  background: var(--surface-raised);
  border: 1px solid var(--ink-faint);
  border-radius: 999px;
  font-size: 13px;
  color: var(--ink);
}
.file-chip-meta { color: var(--ink-muted); }

/* Empty state */
.empty {
  text-align: center;
  color: var(--ink-muted);
  font-size: 14px;
  padding: 60px 20px;
}

/* Hide Gradio labels */
.gr-form > label, .gr-block > label, label.svelte-1gfkn6j { display: none !important; }
"""


# ---------------------------------------------------------- orchestrator
# Single global orchestrator. Models load lazily inside @spaces.GPU.
_AGENT: Optional[SQLAgentOrchestrator] = None


def get_agent() -> SQLAgentOrchestrator:
    global _AGENT
    if _AGENT is None:
        _AGENT = SQLAgentOrchestrator()
    return _AGENT


# --------------------------------------------------------------- helpers
def _file_chip_html(filename: str, rows: int, cols: int) -> str:
    return (
        f'<div class="file-chip">'
        f'<span>{filename}</span>'
        f'<span class="file-chip-meta">· {rows:,} rows · {cols} cols</span>'
        f'</div>'
    )


def _data_table_html(rows: list[dict], max_rows: int = 10) -> str:
    if not rows:
        return '<div class="empty">No rows.</div>'
    df = pd.DataFrame(rows[:max_rows])
    cols = df.columns.tolist()
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{r.get(c, '')}</td>" for c in cols) + "</tr>"
        for r in rows[:max_rows]
    )
    note = (
        f'<div style="font-size:11px;color:var(--ink-muted);'
        f'margin-top:8px">Showing {min(max_rows, len(rows))} of {len(rows):,} rows</div>'
        if len(rows) > max_rows else ""
    )
    return f'<table class="data-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{note}'


def _turn_html(result: dict) -> str:
    """Render one full agent turn: chart, then collapsible SQL + data."""
    parts: list[str] = []
    parts.append(f'<div class="turn-question">{result["question"]}</div>')

    if result.get("error"):
        parts.append(f'<div class="turn-error">{result["error"]}</div>')

    if result.get("svg"):
        parts.append(f'<div class="chart-wrap">{result["svg"]}</div>')

    if result.get("sql"):
        parts.append(
            '<details><summary>SQL</summary>'
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


def _conversation_html(history: list[dict]) -> str:
    if not history:
        return (
            '<div class="empty">'
            'Upload a CSV or JSON file and ask anything about your data.'
            '</div>'
        )
    return "".join(_turn_html(t) for t in history)


# ------------------------------------------------------------ event flow
def on_upload(file) -> Tuple[str, str]:
    """File uploaded -> register in DuckDB, return chip HTML + status."""
    if file is None:
        return "", "No file."
    agent = get_agent()
    agent.reset()
    try:
        path = Path(file.name if hasattr(file, "name") else file)
        table = agent.load_data(path)
        sample = agent.sample(table, 1)
        rows = len(agent.executor.con.execute(f'SELECT * FROM "{table}"').fetchall())
        cols = len(sample.columns)
        return _file_chip_html(path.name, rows, cols), f"Ready: table “{table}”."
    except Exception as e:
        logger.exception("upload failed")
        return "", f"Could not load file: {e}"


@spaces.GPU(duration=120)
def _gpu_process(question: str) -> dict:
    """The actual GPU-bound call. Models are loaded lazily inside this scope."""
    agent = get_agent()
    return agent.process(question)


def on_ask(question: str, history: list) -> Tuple[str, str, list]:
    """Submit a question -> run pipeline -> append to history -> render."""
    history = history or []
    question = (question or "").strip()
    if not question:
        return _conversation_html(history), "", history
    if not get_agent().list_tables():
        history.append({
            "question": question,
            "error": "Upload a CSV or JSON file first.",
        })
        return _conversation_html(history), "", history

    try:
        result = _gpu_process(question)
    except Exception as e:
        logger.exception("ask failed")
        result = {"question": question, "error": str(e)}

    history.append(result)
    return _conversation_html(history), "", history


def on_reset() -> Tuple[str, str, list]:
    get_agent().reset()
    return "", "", []


# ------------------------------------------------------------------ app
def build_app() -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Base(
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ),
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

        # Upload card
        upload = gr.File(
            label="",
            file_types=[".csv", ".json", ".parquet", ".xlsx", ".xls"],
            elem_classes=["upload-zone"],
        )
        chip_html = gr.HTML("")
        upload_status = gr.Markdown("", visible=False)

        # Question input
        question = gr.Textbox(
            placeholder="Ask anything about your data…",
            lines=2,
            max_lines=8,
            show_label=False,
            container=False,
        )
        with gr.Row():
            ask_btn = gr.Button("Ask", variant="primary", size="sm")
            reset_btn = gr.Button("Clear", variant="secondary", size="sm")

        # Conversation
        history_state = gr.State([])
        conversation = gr.HTML(_conversation_html([]))

        # Wire events. api_name=False on each event skips JSON-schema api
        # introspection which crashes on Dict[str, Any] returns in gradio<5.
        upload.upload(
            fn=on_upload,
            inputs=upload,
            outputs=[chip_html, upload_status],
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
            outputs=[chip_html, question, history_state],
            api_name=False,
        ).then(
            fn=lambda: _conversation_html([]),
            outputs=conversation,
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

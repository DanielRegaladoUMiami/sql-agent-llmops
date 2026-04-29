"""
Generate an 8-slide editable PowerPoint deck for the SQL Agent class
presentation. Uses python-pptx so every text element is a native shape
that can be edited in PowerPoint or Keynote.

Slide map:
  1. Title
  2. Problem statement + 6-phase methodology
  3. Data — sourcing, curation, three datasets
  4. Model 01 · SQL Generator
  5. Model 02 · Chart Reasoner
  6. Model 03 · SVG Renderer
  7. System architecture + cost
  8. Demo + conclusion

Run: python3 generate_pptx.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ---------------------------------------------------------------- theme
INK = RGBColor(0x0E, 0x0E, 0x0E)
INK_MUTED = RGBColor(0x5A, 0x5A, 0x5A)
INK_FAINT = RGBColor(0xE5, 0xE5, 0xE5)
SURFACE = RGBColor(0xFA, 0xFA, 0xF9)
SURFACE_RAISED = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT = RGBColor(0xC9, 0x64, 0x42)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Calibri"
FONT_MONO = "Menlo"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


# ----------------------------------------------------------- helpers
def set_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text(slide, x, y, w, h, text, size=18, bold=False, color=INK, font=FONT,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=1.4):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0); tf.margin_right = Emu(0)
    tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line_spacing
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def add_paragraphs(slide, x, y, w, h, paragraphs, size=14, color=INK, font=FONT,
                   line_spacing=1.45):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0); tf.margin_right = Emu(0)
    tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = line_spacing
        if i > 0:
            p.space_before = Pt(4)
        run = p.add_run()
        run.text = para
        run.font.name = font
        run.font.size = Pt(size)
        run.font.color.rgb = color
    return box


def add_rect(slide, x, y, w, h, fill=SURFACE_RAISED, line=INK_FAINT,
             line_w=0.75, corner=0.08):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shape.adjustments[0] = corner
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(line_w)
    shape.shadow.inherit = False
    return shape


def add_line(slide, x1, y1, x2, y2, color=INK_FAINT, weight=0.5):
    line = slide.shapes.add_connector(1, x1, y1, x2, y2)
    line.line.color.rgb = color
    line.line.width = Pt(weight)
    return line


def add_accent_bar(slide):
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.83), Inches(0.45),
        Inches(0.32), Inches(0.04)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()


def add_kicker(slide, text, x=Inches(0.83), y=Inches(0.6)):
    add_text(slide, x, y, Inches(11), Inches(0.3),
             text.upper(), size=10, bold=True, color=ACCENT,
             line_spacing=1.0)


def add_title(slide, text, x=Inches(0.83), y=Inches(0.95), w=Inches(11.67),
              h=Inches(1.0), size=28):
    add_text(slide, x, y, w, h, text, size=size, bold=True, color=INK,
             line_spacing=1.15)


def add_footer(slide, label, page_num):
    add_text(slide, Inches(0.83), Inches(7.05), Inches(8), Inches(0.3),
             label, size=9, color=INK_MUTED)
    add_text(slide, Inches(12.0), Inches(7.05), Inches(0.5), Inches(0.3),
             page_num, size=9, color=INK_MUTED, align=PP_ALIGN.RIGHT)


# ------------------------------------------------------------- build
prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
blank = prs.slide_layouts[6]


# ============================================================ 01 TITLE
slide = prs.slides.add_slide(blank)
set_bg(slide, SURFACE)
add_accent_bar(slide)

add_text(slide, Inches(1.5), Inches(2.3), Inches(10), Inches(1.5),
         "SQL Agent", size=64, bold=True, color=INK, line_spacing=1.05)
add_text(slide, Inches(1.5), Inches(3.4), Inches(10), Inches(0.6),
         "A multi-model system for natural-language data analysis",
         size=20, color=INK_MUTED)

add_line(slide, Inches(1.5), Inches(5.3), Inches(7), Inches(5.3),
         color=INK_FAINT, weight=0.5)

add_text(slide, Inches(1.5), Inches(5.45), Inches(2), Inches(0.3),
         "TEAM", size=10, bold=True, color=INK_MUTED)
add_text(slide, Inches(1.5), Inches(5.75), Inches(10), Inches(0.4),
         "Daniel Regalado Cardoso · Nefeli Zafeiri · Oliver Mazariegos · Eleniz Espina",
         size=14, color=INK)
add_text(slide, Inches(1.5), Inches(6.1), Inches(10), Inches(0.4),
         "MSBA · University of Miami · 2026",
         size=12, color=INK_MUTED)


# ============================================================ 02 PROBLEM + METHODOLOGY
slide = prs.slides.add_slide(blank)
set_bg(slide, SURFACE)
add_accent_bar(slide)
add_kicker(slide, "Problem and methodology")
add_title(slide, "Tabular data analysis still requires SQL knowledge")

# Problem
add_text(slide, Inches(0.83), Inches(2.2), Inches(11), Inches(0.5),
         "Business users have tabular data and questions, but no working SQL knowledge. Existing options (manual SQL, generic chatbots, BI tools) trade off cost, accuracy, or accessibility.",
         size=13, color=INK)

# Section divider
add_text(slide, Inches(0.83), Inches(3.15), Inches(11), Inches(0.3),
         "OUR APPROACH — SIX PHASES FROM RAW DATA TO DEPLOYED SYSTEM",
         size=10, bold=True, color=INK_MUTED)

# Six phase boxes
phases = [
    ("1", "Source", "10 public\ndatasets"),
    ("2", "Curate", "1.2M to 723k\nrows"),
    ("3", "Build", "3 task\ndatasets"),
    ("4", "Train", "3 LoRA\nadapters"),
    ("5", "Architect", "Orchestrator\n+ DuckDB"),
    ("6", "Deploy", "Hugging Face\nSpaces"),
]
box_w = Inches(1.85)
gap = Inches(0.15)
total_w = box_w * 6 + gap * 5
start_x = (SLIDE_W - total_w) / 2
for i, (num, name, sub) in enumerate(phases):
    x = start_x + (box_w + gap) * i
    accent = i >= 3
    line_color = ACCENT if accent else INK_FAINT
    line_w = 1.5 if accent else 0.75
    add_rect(slide, x, Inches(3.7), box_w, Inches(1.7),
             fill=SURFACE_RAISED, line=line_color, line_w=line_w)
    add_text(slide, x, Inches(3.85), box_w, Inches(0.3), num,
             size=11, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    add_text(slide, x, Inches(4.15), box_w, Inches(0.4), name,
             size=15, bold=True, color=INK, align=PP_ALIGN.CENTER)
    add_text(slide, x, Inches(4.6), box_w, Inches(0.7), sub,
             size=11, color=INK_MUTED, align=PP_ALIGN.CENTER)
    if i < 5:
        ax = x + box_w + Inches(0.02)
        add_line(slide, ax, Inches(4.55), ax + Inches(0.11), Inches(4.55),
                 color=INK_MUTED, weight=0.75)

# Objective box at bottom
obj_y = Inches(5.7)
add_text(slide, Inches(0.83), obj_y, Inches(11), Inches(0.3),
         "OBJECTIVE", size=10, bold=True, color=ACCENT)
add_text(slide, Inches(0.83), obj_y + Inches(0.3), Inches(11), Inches(0.7),
         "Accept a CSV/JSON file and a question in English. Return SQL, query results, a chart, and a written finding — at low cost on free GPU infrastructure.",
         size=13, color=INK)

add_footer(slide, "Problem and methodology", "02 / 08")


# ============================================================ 03 DATA
slide = prs.slides.add_slide(blank)
set_bg(slide, SURFACE)
add_accent_bar(slide)
add_kicker(slide, "Phases 1, 2 and 3 — Data")
add_title(slide, "We curated three purpose-built datasets, one per model")

# Narrative paragraph — explains WHAT we did
add_text(slide, Inches(0.83), Inches(2.0), Inches(11.67), Inches(0.5),
         "Building a corpus from scratch was not feasible. Instead, we aggregated 10 public text-to-SQL datasets, unified their schemas, removed duplicates by question hashing, and filtered out examples longer than 1,024 tokens. From the same merged corpus we then derived two more datasets — one for chart reasoning, one for SVG rendering — to train each specialist on its own task.",
         size=12, color=INK, line_spacing=1.5)

# Three dataset cards — bigger, with FUNCTION (what each trains)
ds_y = Inches(3.5)
datasets = [
    {
        "num": "01",
        "name": "text-to-sql-mix-v2",
        "rows": "761,155",
        "trains": "SQL Generator · Qwen 7B",
        "what": "Natural-language question + schema → valid SQL query.",
        "source": "Merged from 10 public datasets (sql-create-context, gretel, knowsql, NSText2SQL, Spider, WikiSQL, etc.). Schemas unified into a 7-column canonical format.",
    },
    {
        "num": "02",
        "name": "chart-reasoning-mix-v1",
        "rows": "~75,000",
        "trains": "Chart Reasoner · Phi-3 Mini",
        "what": "Question + SQL results → JSON chart spec (type, axes, title).",
        "source": "25k real NL/chart pairs from nvBench + 50k pairs synthesized via GPT-4.1-nano knowledge distillation over the SQL corpus.",
    },
    {
        "num": "03",
        "name": "svg-chart-render-v1",
        "rows": "~25,000",
        "trains": "SVG Renderer · DeepSeek 1.3B",
        "what": "Chart spec + data sample → inline SVG markup.",
        "source": "nvBench charts re-rendered through matplotlib's SVG backend, plus chart-shaped SVGs filtered from the svgen-500k collection.",
    },
]

cw = Inches(3.95); cg = Inches(0.2)
total_c = cw * 3 + cg * 2
cx = (SLIDE_W - total_c) / 2

for i, d in enumerate(datasets):
    x = cx + (cw + cg) * i
    add_rect(slide, x, ds_y, cw, Inches(3.3),
             fill=SURFACE_RAISED, line=INK_FAINT, line_w=0.75)

    # Accent stripe at top of card
    s = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, x + Inches(0.2), ds_y + Inches(0.18),
        Inches(0.4), Inches(0.05))
    s.fill.solid(); s.fill.fore_color.rgb = ACCENT; s.line.fill.background()

    inset = x + Inches(0.2)
    text_w = cw - Inches(0.4)

    # Dataset number + name
    add_text(slide, inset, ds_y + Inches(0.32), text_w, Inches(0.25),
             f"DATASET · {d['num']}", size=9, bold=True, color=ACCENT)
    add_text(slide, inset, ds_y + Inches(0.6), text_w, Inches(0.4),
             d["name"], size=14, bold=True, color=INK, font=FONT_MONO)

    # Rows (big)
    add_text(slide, inset, ds_y + Inches(1.05), text_w, Inches(0.6),
             d["rows"], size=28, bold=True, color=INK)
    add_text(slide, inset, ds_y + Inches(1.65), text_w, Inches(0.3),
             "ROWS", size=8, color=INK_MUTED)

    # Trains label
    add_text(slide, inset, ds_y + Inches(2.0), text_w, Inches(0.3),
             "TRAINS", size=8, bold=True, color=INK_MUTED)
    add_text(slide, inset, ds_y + Inches(2.22), text_w, Inches(0.4),
             d["trains"], size=11, bold=True, color=INK)

    # What
    add_text(slide, inset, ds_y + Inches(2.55), text_w, Inches(0.5),
             d["what"], size=10, color=INK)

    # Source (smaller, muted)
    add_text(slide, inset, ds_y + Inches(2.95), text_w, Inches(0.5),
             d["source"], size=9, color=INK_MUTED, line_spacing=1.35)

# Footer-line takeaway
add_text(slide, Inches(0.83), Inches(7.0), Inches(11.67), Inches(0.3),
         "All three published openly on Hugging Face Hub under permissive licenses (Apache-2.0, CC-BY-4.0).",
         size=11, color=INK_MUTED)

add_footer(slide, "Data — sourcing, curation, and three datasets", "03 / 08")


# ============================================================ 04, 05, 06 THREE MODELS
def model_slide(page, role_num, name, function, base, dataset, method,
                hardware, time, loss, cost, size, hub, note):
    slide = prs.slides.add_slide(blank)
    set_bg(slide, SURFACE)
    add_accent_bar(slide)
    add_kicker(slide, f"Phase 4 — Three fine-tunes ({role_num} of 3)")
    add_title(slide, f"Model 0{role_num} · {name}")

    # Left card
    lx = Inches(0.83); ly = Inches(2.3); lw = Inches(6.0); lh = Inches(4.5)
    add_rect(slide, lx, ly, lw, lh, fill=SURFACE_RAISED, line=INK_FAINT)
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                lx + Inches(0.25), ly + Inches(0.18),
                                Inches(0.4), Inches(0.06))
    s.fill.solid(); s.fill.fore_color.rgb = ACCENT; s.line.fill.background()

    inset_x = lx + Inches(0.25)
    add_text(slide, inset_x, ly + Inches(0.3), lw - Inches(0.5), Inches(0.3),
             f"MODEL · 0{role_num}", size=10, bold=True, color=ACCENT)
    add_text(slide, inset_x, ly + Inches(0.6), lw - Inches(0.5), Inches(0.5),
             name, size=22, bold=True, color=INK)

    cur_y = ly + Inches(1.3)

    def field(label, body, gap=0.55):
        nonlocal cur_y
        add_text(slide, inset_x, cur_y, Inches(2.0), Inches(0.3),
                 label.upper(), size=9, bold=True, color=INK_MUTED)
        add_text(slide, inset_x + Inches(2.0), cur_y, lw - Inches(2.5), Inches(0.5),
                 body, size=12, color=INK)
        cur_y += Inches(gap)

    field("Function", function)
    field("Base model", base)
    field("Dataset", dataset)
    field("Method", method, gap=0.7)
    field("Hardware", hardware)

    # Right card: metrics + hub
    rx = Inches(7.13); ry = Inches(2.3); rw = Inches(5.4); rh = Inches(4.5)
    add_rect(slide, rx, ry, rw, rh, fill=SURFACE_RAISED, line=INK_FAINT)

    stats = [(time, "Wall-clock time"),
             (loss, "Final loss"),
             (cost, "Compute cost"),
             (size, "Adapter size")]
    sw_in = (rw - Inches(0.7)) / 2
    sh_in = Inches(1.0)
    for i, (val, label) in enumerate(stats):
        col = i % 2
        row = i // 2
        sx_box = rx + Inches(0.25) + (sw_in + Inches(0.2)) * col
        sy_box = ry + Inches(0.25) + (sh_in + Inches(0.15)) * row
        add_text(slide, sx_box, sy_box, sw_in, Inches(0.6),
                 val, size=28, bold=True, color=INK)
        add_text(slide, sx_box, sy_box + Inches(0.6), sw_in, Inches(0.4),
                 label.upper(), size=9, color=INK_MUTED)

    add_text(slide, rx + Inches(0.25), ry + Inches(2.65), rw - Inches(0.5), Inches(0.3),
             "HUB", size=9, bold=True, color=INK_MUTED)
    add_text(slide, rx + Inches(0.25), ry + Inches(2.95), rw - Inches(0.5), Inches(0.5),
             hub, size=11, color=ACCENT, font=FONT_MONO)
    add_text(slide, rx + Inches(0.25), ry + Inches(3.6), rw - Inches(0.5), Inches(0.8),
             note, size=11, color=INK_MUTED)

    add_footer(slide, f"Phase 4 · Model 0{role_num}", page)


# Model 01 · SQL Generator
model_slide(
    page="04 / 08", role_num=1,
    name="SQL Generator",
    function="Translates a natural-language question and schema into a valid SQL query.",
    base="Qwen 2.5 Coder 7B Instruct",
    dataset="text-to-sql-mix-v2 (672,949 examples)",
    method="QLoRA r=16, α=32, 4-bit base · TRL packing=True (154,462 packed sequences) · 1 epoch · 9,654 steps",
    hardware="1× NVIDIA L40S (48 GB) on Hugging Face Jobs",
    time="13.5h", loss="0.27", cost="~$24", size="161 MB",
    hub="DanielRegaladoCardoso/sql-generator-qwen25-coder-7b-lora",
    note="Sequence packing reduced training time from approximately 21 h to 13.5 h.",
)

# Model 02 · Chart Reasoner
model_slide(
    page="05 / 08", role_num=2,
    name="Chart Reasoner",
    function="Given a question and SQL result rows, decides the chart type and which columns to plot.",
    base="Microsoft Phi-3 Mini 4k Instruct",
    dataset="chart-reasoning-mix-v1 (~75,000 pairs)",
    method="QLoRA r=16, α=32, 4-bit base · 1 epoch · structured-JSON output objective",
    hardware="HF Jobs A10G / Colab Pro",
    time="~3h", loss="~0.31", cost="~$3", size="38 MB",
    hub="DanielRegaladoCardoso/chart-reasoner-phi3-mini-adapter-only",
    note="Outputs a JSON spec with chart_type, x_column, y_column, title, color, and rationale.",
)

# Model 03 · SVG Renderer
model_slide(
    page="06 / 08", role_num=3,
    name="SVG Renderer",
    function="Given a chart spec and data, produces inline SVG markup for the visualization.",
    base="DeepSeek Coder 1.3B Instruct",
    dataset="svg-chart-render-v1 (~25,000 chart-spec → SVG pairs)",
    method="QLoRA r=16, α=32, 4-bit base · 1 epoch · code-generation objective",
    hardware="Colab T4",
    time="~2h", loss="~0.40", cost="~$1", size="22 MB",
    hub="DanielRegaladoCardoso/svg-renderer-deepseek-coder-1.3b-lora",
    note="When model output fails SVG validation, the system falls back to a themed Plotly renderer.",
)


# ============================================================ 07 ARCHITECTURE + COST
slide = prs.slides.add_slide(blank)
set_bg(slide, SURFACE)
add_accent_bar(slide)
add_kicker(slide, "Phases 5 and 6 — Architecture and cost")
add_title(slide, "End-to-end query flow and total compute spend")

# Architecture flow
flow = [
    ("CSV / NL Q", False, False, False),
    ("Schema +\nDuckDB", False, False, False),
    ("Orchestrator", False, True, False),
    ("SQL Generator\nQwen + LoRA", True, False, False),
    ("Chart Reasoner\nPhi-3 + LoRA", True, False, False),
    ("SVG Renderer\nDeepSeek + LoRA", True, False, False),
    ("Chart +\nfinding", False, False, True),
]
fw = Inches(1.55); fg = Inches(0.18)
total = fw * len(flow) + fg * (len(flow) - 1)
fx = (SLIDE_W - total) / 2
for i, (label, accent_border, dark_fill, accent_fill) in enumerate(flow):
    x = fx + (fw + fg) * i
    fill = INK if dark_fill else (ACCENT if accent_fill else SURFACE_RAISED)
    line = ACCENT if accent_border else INK_FAINT
    line_w = 1.5 if accent_border else 0.75
    text_color = WHITE if (dark_fill or accent_fill) else INK
    add_rect(slide, x, Inches(2.3), fw, Inches(1.4), fill=fill, line=line, line_w=line_w)
    add_text(slide, x, Inches(2.65), fw, Inches(0.9), label,
             size=11, bold=True, color=text_color, align=PP_ALIGN.CENTER)
    if i < len(flow) - 1:
        ax = x + fw + Inches(0.04)
        add_line(slide, ax, Inches(3.0), ax + Inches(0.10), Inches(3.0),
                 color=INK_MUTED, weight=0.75)

# Description below flow
add_text(slide, Inches(0.83), Inches(3.85), Inches(11.67), Inches(0.4),
         "Per query: 4 model calls + DuckDB execution. End-to-end latency 5–8 s on a warm GPU.",
         size=12, color=INK)
add_text(slide, Inches(0.83), Inches(4.2), Inches(11.67), Inches(0.4),
         "Adapters loaded once at module level on a half-H200 via Hugging Face ZeroGPU. Self-correcting SQL retries on failure.",
         size=11, color=INK_MUTED)

# Cost section
add_text(slide, Inches(0.83), Inches(4.85), Inches(11), Inches(0.3),
         "TOTAL COMPUTE COST",
         size=10, bold=True, color=ACCENT)

# Big $30 + breakdown
add_text(slide, Inches(0.83), Inches(5.2), Inches(3.5), Inches(1.4),
         "$30", size=84, bold=True, color=INK)
add_text(slide, Inches(0.83), Inches(6.55), Inches(3.5), Inches(0.3),
         "ALL-IN", size=10, bold=True, color=INK_MUTED)

# Cost table on right
cost_rows = [
    ("SQL Generator training", "L40S, 13.5 h", "~$24"),
    ("Chart Reasoner training", "Colab / HF Jobs", "~$3"),
    ("SVG Renderer training", "Colab / HF Jobs", "~$1"),
    ("GPT-4.1-nano distillation", "OpenAI Batch", "~$2.50"),
    ("Inference hosting", "ZeroGPU", "$0"),
]
ct_x = Inches(4.7); ct_y = Inches(5.2)
for i, (a, b, c) in enumerate(cost_rows):
    yy = ct_y + Inches(i * 0.35)
    add_text(slide, ct_x, yy, Inches(3.8), Inches(0.3), a, size=11, color=INK)
    add_text(slide, ct_x + Inches(3.8), yy, Inches(2.5), Inches(0.3), b, size=11, color=INK_MUTED)
    add_text(slide, ct_x + Inches(6.3), yy, Inches(1.2), Inches(0.3), c, size=11, color=INK, align=PP_ALIGN.RIGHT)

add_footer(slide, "Architecture and cost", "07 / 08")


# ============================================================ 08 DEMO + CONCLUSION
slide = prs.slides.add_slide(blank)
set_bg(slide, SURFACE)
add_accent_bar(slide)
add_kicker(slide, "Demo and conclusion")
add_title(slide, "Live system demo and summary")

# Demo strip
add_rect(slide, Inches(0.83), Inches(2.3), Inches(11.67), Inches(1.4),
         fill=SURFACE_RAISED, line=ACCENT, line_w=1.5)
add_text(slide, Inches(1.1), Inches(2.5), Inches(11.0), Inches(0.4),
         "LIVE DEMO", size=10, bold=True, color=ACCENT)
add_text(slide, Inches(1.1), Inches(2.85), Inches(11.0), Inches(0.5),
         "huggingface.co/spaces/DanielRegaladoCardoso/sql-agent",
         size=18, bold=True, color=INK)
add_text(slide, Inches(1.1), Inches(3.3), Inches(11.0), Inches(0.4),
         "Upload a CSV, ask a question. The system returns SQL, results, a chart, and a written finding.",
         size=12, color=INK_MUTED)

# Two columns
add_text(slide, Inches(0.83), Inches(4.0), Inches(5.5), Inches(0.4),
         "WHAT WE DELIVERED", size=10, bold=True, color=ACCENT)
add_paragraphs(slide, Inches(0.83), Inches(4.35), Inches(5.5), Inches(2.5), [
    "•  Three open-source datasets on Hugging Face",
    "•  Three QLoRA adapters trained on those datasets",
    "•  A working multi-model agent",
    "•  Total compute cost approximately $30",
], size=12, color=INK)

add_text(slide, Inches(7.0), Inches(4.0), Inches(5.5), Inches(0.4),
         "FUTURE WORK", size=10, bold=True, color=ACCENT)
add_paragraphs(slide, Inches(7.0), Inches(4.35), Inches(5.5), Inches(2.5), [
    "•  Quantitative evaluation on Spider, WikiSQL, BIRD",
    "•  Multi-turn conversational memory",
    "•  Anomaly detection on uploaded data",
    "•  Statistical summary at ingestion",
], size=12, color=INK)

# Footer
add_text(slide, Inches(0.83), Inches(6.7), Inches(11.67), Inches(0.3),
         "github.com/DanielRegaladoUMiami/sql-agent-llmops",
         size=11, color=ACCENT, font=FONT_MONO)
add_text(slide, Inches(0.83), Inches(7.05), Inches(11.67), Inches(0.3),
         "Daniel Regalado Cardoso · Nefeli Zafeiri · Oliver Mazariegos · Eleniz Espina · MSBA UMiami 2026",
         size=10, color=INK_MUTED)


# ------------------------------------------------------------ save
out = "slides-editable.pptx"
prs.save(out)
print(f"Saved: {out} ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)" if hasattr(prs.slides, '__iter__') else f"Saved: {out}")

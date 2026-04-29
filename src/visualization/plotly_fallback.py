"""
Plotly fallback renderer with the project's Apple/Claude theme baked in.

Returns inline SVG strings so the app can drop them into gr.HTML directly.
"""

import logging
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

logger = logging.getLogger(__name__)


# Theme constants (mirror of svg_theme.py)
ACCENT = "#C96442"
INK = "#0E0E0E"
INK_MUTED = "#5A5A5A"
INK_FAINT = "#E5E5E5"

FONT_FAMILY = (
    '-apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", '
    '"Helvetica Neue", Arial, sans-serif'
)


def _fmt(v: float) -> str:
    """Compact number formatting for axis ticks."""
    if abs(v) >= 1e9:
        return f"{v/1e9:.1f}B"
    if abs(v) >= 1e6:
        return f"{v/1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"{v/1e3:.1f}k"
    if abs(v - int(v)) < 1e-9:
        return str(int(v))
    return f"{v:.2f}"


def _theme_layout(title: str = "") -> dict:
    # Don't render Plotly's title — the user's question is already shown
    # above the chart card, this would just duplicate.
    return dict(
        title=dict(text="", font=dict(family=FONT_FAMILY, size=15, color=INK)),
        font=dict(family=FONT_FAMILY, size=12, color=INK),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=48, r=24, t=44, b=44),
        xaxis=dict(
            showgrid=True, gridcolor=INK_FAINT, zerolinecolor=INK_FAINT,
            linecolor=INK_FAINT, tickcolor=INK_FAINT,
            tickfont=dict(color=INK_MUTED, family=FONT_FAMILY, size=11),
            title_font=dict(color=INK, family=FONT_FAMILY, size=12),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=INK_FAINT, zerolinecolor=INK_FAINT,
            linecolor=INK_FAINT, tickcolor=INK_FAINT,
            tickfont=dict(color=INK_MUTED, family=FONT_FAMILY, size=11),
            title_font=dict(color=INK, family=FONT_FAMILY, size=12),
        ),
        showlegend=False,
        colorway=[ACCENT, INK, INK_MUTED, "#8B7355", "#A0826D"],
    )


class PlotlyRenderer:
    """Render a chart spec + data tuple as a themed inline SVG."""

    MAX_CATEGORIES = 30  # cap for bar/pie/scatter to keep charts readable

    def render(self, spec: Dict[str, Any], data: List[Dict[str, Any]]) -> str:
        if not data:
            return self._empty("No data returned by the query.")

        df = pd.DataFrame(data)
        chart_type = (spec.get("chart_type") or "bar").lower()
        title = spec.get("title") or ""
        x = spec.get("x_column") or (df.columns[0] if len(df.columns) >= 1 else None)
        y = spec.get("y_column") or (df.columns[1] if len(df.columns) >= 2 else None)

        # If categorical chart with too many categories: take top N by y value
        if chart_type in ("bar", "pie") and y and y in df.columns and len(df) > self.MAX_CATEGORIES:
            try:
                df = df.sort_values(by=y, ascending=False).head(self.MAX_CATEGORIES)
                title = (title or "Top values") + f" (top {self.MAX_CATEGORIES})"
            except Exception:
                df = df.head(self.MAX_CATEGORIES)

        try:
            fig = self._build(chart_type, df, x, y, title, spec)
        except Exception as e:
            logger.warning(f"Plotly build failed ({e}); rendering as table")
            fig = self._table_fig(df, title)

        fig.update_layout(**_theme_layout(title))
        return self._to_svg(fig)

    def _build(
        self,
        kind: str,
        df: pd.DataFrame,
        x: str | None,
        y: str | None,
        title: str,
        spec: Dict[str, Any],
    ) -> go.Figure:
        if kind == "table" or x is None or y is None:
            return self._table_fig(df, title)

        if kind == "bar":
            return go.Figure(go.Bar(
                x=df[x], y=df[y], marker_color=ACCENT, marker_line_width=0,
            ))
        if kind == "line":
            return go.Figure(go.Scatter(
                x=df[x], y=df[y], mode="lines+markers",
                line=dict(color=ACCENT, width=2),
                marker=dict(color=ACCENT, size=6),
            ))
        if kind == "area":
            return go.Figure(go.Scatter(
                x=df[x], y=df[y], mode="lines",
                fill="tozeroy",
                line=dict(color=ACCENT, width=2),
                fillcolor="rgba(201,100,66,0.18)",
            ))
        if kind == "scatter":
            return go.Figure(go.Scatter(
                x=df[x], y=df[y], mode="markers",
                marker=dict(color=ACCENT, size=8, opacity=0.75),
            ))
        if kind == "pie":
            return go.Figure(go.Pie(
                labels=df[x], values=df[y], hole=0.55,
                marker=dict(line=dict(color="#FAFAF9", width=2)),
                textfont=dict(family=FONT_FAMILY, color=INK),
            ))
        if kind == "histogram":
            return go.Figure(go.Histogram(x=df[x], marker_color=ACCENT))

        return go.Figure(go.Bar(x=df[x], y=df[y], marker_color=ACCENT))

    def _table_fig(self, df: pd.DataFrame, title: str) -> go.Figure:
        df = df.head(100)
        return go.Figure(go.Table(
            header=dict(
                values=[f"<b>{c}</b>" for c in df.columns],
                fill_color="rgba(0,0,0,0)",
                line_color=INK_FAINT,
                align="left",
                font=dict(family=FONT_FAMILY, color=INK, size=12),
                height=32,
            ),
            cells=dict(
                values=[df[c].astype(str).tolist() for c in df.columns],
                fill_color="rgba(0,0,0,0)",
                line_color=INK_FAINT,
                align="left",
                font=dict(family=FONT_FAMILY, color=INK_MUTED, size=11),
                height=28,
            ),
        ))

    def _to_svg(self, fig: go.Figure) -> str:
        """Try Plotly→SVG via kaleido; if that fails, hand-draw a simple SVG
        from the figure's data so the user always gets a visual."""
        try:
            return pio.to_image(fig, format="svg").decode("utf-8")
        except Exception as e:
            logger.warning(f"Plotly to_image failed ({e}); using native SVG fallback")
            return self._native_svg(fig)

    def _native_svg(self, fig: go.Figure) -> str:
        """Hand-draw a simple SVG from the figure data — no kaleido needed."""
        traces = fig.data
        if not traces:
            return self._empty("No data to plot.")

        W, H = 600, 380
        pad_l, pad_r, pad_t, pad_b = 60, 30, 40, 50

        # Pull the first trace's data
        t = traces[0]
        xs = list(getattr(t, "x", []) or [])
        ys = list(getattr(t, "y", []) or [])
        # For pies / tables we don't have x/y; fall back to text
        if not xs or not ys:
            return self._empty("Unsupported chart shape; see Data section.")

        try:
            ys_num = [float(y) if y is not None else 0.0 for y in ys]
        except (TypeError, ValueError):
            return self._empty("Non-numeric values; see Data section.")

        ymin, ymax = min(ys_num), max(ys_num)
        if ymin == ymax:
            ymin -= 1
            ymax += 1
        plot_w = W - pad_l - pad_r
        plot_h = H - pad_t - pad_b
        n = len(xs)

        def y_to_px(v: float) -> float:
            return pad_t + plot_h * (1 - (v - ymin) / (ymax - ymin))

        title = (fig.layout.title.text or "").strip() if fig.layout.title else ""

        kind = type(t).__name__.lower()  # "Bar", "Scatter", etc.
        bars: list[str] = []
        path: list[str] = []

        if "bar" in kind:
            bw = max(2.0, plot_w / max(n, 1) * 0.8)
            for i, v in enumerate(ys_num):
                cx = pad_l + (i + 0.5) * (plot_w / max(n, 1))
                top = y_to_px(v)
                bottom = y_to_px(0)
                y = min(top, bottom)
                h = abs(bottom - top)
                bars.append(
                    f'<rect x="{cx-bw/2:.1f}" y="{y:.1f}" '
                    f'width="{bw:.1f}" height="{h:.1f}" '
                    f'class="chart-accent" rx="2" />'
                )
        else:  # treat as line
            pts = [
                f"{pad_l + i * (plot_w / max(n - 1, 1)):.1f},{y_to_px(v):.1f}"
                for i, v in enumerate(ys_num)
            ]
            path.append(
                f'<polyline points="{" ".join(pts)}" fill="none" '
                f'class="chart-accent" stroke-width="2" />'
            )
            for p in pts:
                x, y = p.split(",")
                path.append(f'<circle cx="{x}" cy="{y}" r="3" class="chart-accent" />')

        # Axes labels — show min, mid, max on Y
        y_ticks = []
        for v in (ymin, (ymin + ymax) / 2, ymax):
            yp = y_to_px(v)
            y_ticks.append(
                f'<line x1="{pad_l}" y1="{yp:.1f}" x2="{W-pad_r}" y2="{yp:.1f}" class="chart-grid" />'
                f'<text x="{pad_l-8:.1f}" y="{yp+3:.1f}" text-anchor="end" '
                f'font-size="10" class="chart-muted">{_fmt(v)}</text>'
            )

        # X labels — every Nth so we don't overlap
        step = max(1, n // 8)
        x_labels = []
        for i in range(0, n, step):
            cx = pad_l + (i + 0.5) * (plot_w / max(n, 1))
            label = str(xs[i])
            if len(label) > 12:
                label = label[:11] + "…"
            x_labels.append(
                f'<text x="{cx:.1f}" y="{H-pad_b+18}" text-anchor="middle" '
                f'font-size="10" class="chart-muted">{label}</text>'
            )

        title_el = (
            f'<text x="{W/2}" y="{20}" text-anchor="middle" '
            f'font-size="13" font-weight="600" class="chart-ink">{title}</text>'
            if title else ""
        )

        return (
            f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" '
            f'style="width:100%;height:auto;display:block">'
            f'{title_el}'
            f'{"".join(y_ticks)}'
            f'{"".join(bars)}{"".join(path)}'
            f'{"".join(x_labels)}'
            f'</svg>'
        )

    def _empty(self, msg: str) -> str:
        return f'''<svg viewBox="0 0 600 200" preserveAspectRatio="xMidYMid meet"
                       style="width:100%;height:auto;display:block">
            <text x="300" y="100" text-anchor="middle"
                  font-family="{FONT_FAMILY}" font-size="14" fill="{INK_MUTED}">
                {msg}
            </text>
        </svg>'''

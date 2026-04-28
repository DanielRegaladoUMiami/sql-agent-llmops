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


def _theme_layout(title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(family=FONT_FAMILY, size=15, color=INK)),
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

    def render(self, spec: Dict[str, Any], data: List[Dict[str, Any]]) -> str:
        if not data:
            return self._empty("No data returned by the query.")

        df = pd.DataFrame(data)
        chart_type = (spec.get("chart_type") or "bar").lower()
        title = spec.get("title") or ""
        x = spec.get("x_column") or (df.columns[0] if len(df.columns) >= 1 else None)
        y = spec.get("y_column") or (df.columns[1] if len(df.columns) >= 2 else None)

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
        try:
            return pio.to_image(fig, format="svg").decode("utf-8")
        except Exception as e:
            logger.warning(f"to_image SVG failed ({e}); returning HTML")
            return fig.to_html(include_plotlyjs="cdn", full_html=False)

    def _empty(self, msg: str) -> str:
        return f'''<svg viewBox="0 0 600 200" preserveAspectRatio="xMidYMid meet"
                       style="width:100%;height:auto;display:block">
            <text x="300" y="100" text-anchor="middle"
                  font-family="{FONT_FAMILY}" font-size="14" fill="{INK_MUTED}">
                {msg}
            </text>
        </svg>'''

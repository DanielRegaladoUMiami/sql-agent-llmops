"""
SVG post-processor: enforce a consistent Apple/Claude aesthetic on any
SVG (whether produced by the trained model or the Plotly fallback).

The output should:
- Be responsive (viewBox + 100% width)
- Use a single warm accent (#C96442) plus monochrome ink (#0E0E0E / #5A5A5A)
- Use SF Pro / system font stack
- Use thin strokes (1.25-1.5px) and no chrome
- Include light grid lines instead of axes lines
"""

import re
from xml.etree import ElementTree as ET

# Theme constants — keep in sync with app CSS
ACCENT = "#C96442"
INK = "#0E0E0E"
INK_MUTED = "#5A5A5A"
INK_FAINT = "#E5E5E5"
SURFACE = "#FAFAF9"

FONT_FAMILY = (
    '-apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", '
    '"Helvetica Neue", Arial, sans-serif'
)


def is_renderable_svg(svg: str) -> bool:
    """Cheap structural validity check — does this look like a real SVG with content?"""
    if not svg or "<svg" not in svg.lower():
        return False
    if "</svg>" not in svg.lower():
        return False
    # Require at least a few drawing primitives
    primitives = sum(svg.lower().count(f"<{tag}") for tag in
                     ("rect", "path", "line", "circle", "polygon", "polyline", "text", "g "))
    return primitives >= 3


def apply_theme(svg: str) -> str:
    """Normalize an SVG to the project's visual language."""
    if not svg or "<svg" not in svg:
        return svg

    svg = _ensure_viewbox(svg)
    svg = _ensure_responsive(svg)
    svg = _normalize_fonts(svg)
    svg = _normalize_strokes(svg)
    svg = _wrap_with_theme(svg)
    return svg


def _ensure_viewbox(svg: str) -> str:
    """If width/height are present but viewBox is missing, derive it."""
    if re.search(r"viewBox\s*=", svg, re.IGNORECASE):
        return svg
    w = _attr(svg, "width") or "600"
    h = _attr(svg, "height") or "400"
    w_num = re.sub(r"[^\d.]", "", w) or "600"
    h_num = re.sub(r"[^\d.]", "", h) or "400"
    return re.sub(
        r"<svg",
        f'<svg viewBox="0 0 {w_num} {h_num}"',
        svg,
        count=1,
        flags=re.IGNORECASE,
    )


def _ensure_responsive(svg: str) -> str:
    """Strip explicit width/height so the SVG fills its container responsively."""
    svg = re.sub(r'\s(width|height)="[^"]*"', "", svg, flags=re.IGNORECASE, count=2)
    if "preserveAspectRatio" not in svg:
        svg = re.sub(
            r"<svg",
            '<svg preserveAspectRatio="xMidYMid meet" '
            'style="width:100%;height:auto;display:block"',
            svg,
            count=1,
            flags=re.IGNORECASE,
        )
    return svg


def _normalize_fonts(svg: str) -> str:
    """Force the system font stack on all text."""
    svg = re.sub(
        r'font-family\s*=\s*"[^"]*"',
        f'font-family="{FONT_FAMILY}"',
        svg,
        flags=re.IGNORECASE,
    )
    svg = re.sub(
        r"font-family\s*:\s*[^;\"']+",
        f"font-family:{FONT_FAMILY}",
        svg,
        flags=re.IGNORECASE,
    )
    return svg


def _normalize_strokes(svg: str) -> str:
    """Make all strokes thin and consistent."""
    svg = re.sub(
        r'stroke-width\s*=\s*"[^"]*"',
        'stroke-width="1.25"',
        svg,
        flags=re.IGNORECASE,
    )
    return svg


def _wrap_with_theme(svg: str) -> str:
    """
    Inject a <style> block scoped to the SVG that sets default colors using
    CSS variables that the host document can override (light/dark themes).
    """
    style = f"""<style>
      .chart-bg {{ fill: transparent; }}
      .chart-ink, text {{ fill: {INK}; font-family: {FONT_FAMILY}; }}
      .chart-muted {{ fill: {INK_MUTED}; }}
      .chart-grid {{ stroke: {INK_FAINT}; stroke-width: 0.75; }}
      .chart-accent {{ fill: {ACCENT}; stroke: {ACCENT}; }}
    </style>"""
    return re.sub(r"(<svg[^>]*>)", r"\1" + style, svg, count=1, flags=re.IGNORECASE)


def _attr(svg: str, name: str) -> str | None:
    m = re.search(rf'\s{name}\s*=\s*"([^"]+)"', svg, re.IGNORECASE)
    return m.group(1) if m else None

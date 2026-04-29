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

# Variant with single quotes for use INSIDE style="..." attributes — double
# quotes inside double-quoted attributes break XML parsing on download.
FONT_FAMILY_SINGLE = FONT_FAMILY.replace('"', "'")


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

    svg = _strip_plotly_chrome(svg)
    svg = _ensure_viewbox(svg)
    svg = _ensure_responsive(svg)
    svg = _normalize_fonts(svg)
    svg = _normalize_strokes(svg)
    svg = _wrap_with_theme(svg)
    return svg


def _strip_plotly_chrome(svg: str) -> str:
    """Remove Plotly's modebar, watermark and toolbar artifacts."""
    # Plotly draws a modebar group — strip it
    svg = re.sub(
        r'<g[^>]*class="[^"]*modebar[^"]*"[^>]*>.*?</g>',
        "",
        svg,
        flags=re.DOTALL,
    )
    # Remove explicit white backgrounds that Plotly adds
    svg = re.sub(
        r'fill="(?:rgb\(255,\s*255,\s*255\)|#fff(?:fff)?|white)"',
        'fill="transparent"',
        svg,
        flags=re.IGNORECASE,
    )
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
    """Strip explicit width/height so the SVG fills its container responsively.
    Adds preserveAspectRatio and style only if not already present, and merges
    style values to avoid duplicate style attributes (which break XML parse)."""
    # Drop explicit width/height
    svg = re.sub(r'\s(width|height)="[^"]*"', "", svg, flags=re.IGNORECASE, count=2)

    if "preserveAspectRatio" not in svg:
        svg = re.sub(
            r"<svg",
            '<svg preserveAspectRatio="xMidYMid meet"',
            svg,
            count=1,
            flags=re.IGNORECASE,
        )

    # Merge style: if existing style="" or style="..." present, append; else add
    style_value = "width:100%;height:auto;display:block"
    if re.search(r'<svg[^>]*\sstyle\s*=', svg, re.IGNORECASE):
        # Already has a style attr — merge our values into it
        def _merge(m):
            existing = m.group(2).strip().rstrip(";")
            merged = (existing + ";" if existing else "") + style_value
            return f'{m.group(1)}style="{merged}"'
        svg = re.sub(
            r'(<svg[^>]*\s)style\s*=\s*"([^"]*)"',
            _merge, svg, count=1, flags=re.IGNORECASE,
        )
    else:
        svg = re.sub(
            r"<svg",
            f'<svg style="{style_value}"',
            svg, count=1, flags=re.IGNORECASE,
        )
    return svg


def _normalize_fonts(svg: str) -> str:
    """Force the system font stack on all text. Use the single-quote variant
    when injecting INTO a style="..." attribute, double-quote when standalone."""
    # Standalone font-family attribute: font-family="..."
    svg = re.sub(
        r'font-family\s*=\s*"[^"]*"',
        f'font-family="{FONT_FAMILY}"',
        svg,
        flags=re.IGNORECASE,
    )
    # Inline in style attribute: font-family:... — must use single quotes
    # so we don't break the surrounding double-quoted style attribute
    svg = re.sub(
        r"font-family\s*:\s*[^;\"']+",
        f"font-family:{FONT_FAMILY_SINGLE}",
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
    """Inject a <style> block scoped to the SVG with explicit hex colors.
    These work both in-app and as a standalone downloaded file."""
    style = f"""<style>
      .chart-bg {{ fill: transparent; }}
      .chart-ink, text {{ fill: {INK}; font-family: {FONT_FAMILY}; }}
      .chart-muted {{ fill: {INK_MUTED}; }}
      .chart-grid {{ stroke: {INK_FAINT}; stroke-width: 0.75; }}
      .chart-accent {{ fill: {ACCENT}; stroke: {ACCENT}; }}
    </style>"""
    return re.sub(r"(<svg[^>]*>)", r"\1" + style, svg, count=1, flags=re.IGNORECASE)


def to_standalone_svg(svg: str) -> str:
    """Minimal hardening for downloaded SVG:
    - Ensure XML namespace declarations are present
    - Add XML prolog
    - Do NOT touch existing attributes / dimensions / inner content
      (heavier post-processing was breaking the XML for some Plotly outputs)
    """
    if not svg or "<svg" not in svg:
        return svg

    out = svg.strip()

    # Add xmlns if missing — this is the only requirement for standalone files
    if "xmlns=" not in out[:200]:
        out = re.sub(
            r"<svg(?![^>]*xmlns=)",
            '<svg xmlns="http://www.w3.org/2000/svg"',
            out, count=1, flags=re.IGNORECASE,
        )
    if "xmlns:xlink=" not in out[:300]:
        out = re.sub(
            r"<svg(?![^>]*xmlns:xlink=)",
            '<svg xmlns:xlink="http://www.w3.org/1999/xlink"',
            out, count=1, flags=re.IGNORECASE,
        )

    # XML prolog
    if not out.startswith("<?xml"):
        out = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + out

    return out


def _attr(svg: str, name: str) -> str | None:
    m = re.search(rf'\s{name}\s*=\s*"([^"]+)"', svg, re.IGNORECASE)
    return m.group(1) if m else None

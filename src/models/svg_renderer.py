"""
SVG Renderer model that converts chart configs to SVG visualizations.

Validates SVG output and provides Plotly fallback for rendering.
"""

import logging
from typing import Optional, Any, Dict, List

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class SVGRenderer(BaseModel):
    """Render chart configs to SVG format with validation and fallbacks."""

    def __init__(
        self,
        model_name: str = "svg-renderer",
        model_path: Optional[str] = None,
    ) -> None:
        """
        Initialize SVG Renderer.

        Args:
            model_name: Model identifier
            model_path: Path to model (unused for this renderer)
        """
        super().__init__(model_name, model_path)
        self.plotly_available = self._check_plotly()

    def _check_plotly(self) -> bool:
        """Check if Plotly is available."""
        try:
            import plotly
            return True
        except ImportError:
            logger.warning("Plotly not available, will use fallback SVG generation")
            return False

    def load(self) -> None:
        """Initialize renderer (minimal setup for SVG rendering)."""
        self.is_loaded = True
        logger.info("SVG Renderer loaded")

    def generate(
        self,
        chart_config: Dict[str, Any],
        data: List[Dict[str, Any]],
    ) -> str:
        """
        Generate SVG from chart configuration and data.

        Args:
            chart_config: Chart configuration dict
            data: Data rows for visualization

        Returns:
            SVG string
        """
        self._validate_loaded()

        try:
            if self.plotly_available:
                svg = self._render_plotly(chart_config, data)
            else:
                svg = self._render_fallback(chart_config, data)

            self._validate_svg(svg)
            logger.info(f"Generated SVG ({len(svg)} chars)")
            return svg

        except Exception as e:
            logger.error(f"Error generating SVG: {e}")
            return self._generate_error_svg(str(e))

    def _render_plotly(self, chart_config: Dict[str, Any], data: List[Dict[str, Any]]) -> str:
        """Render using Plotly."""
        import plotly.graph_objects as go
        from plotly.io import to_image

        chart_type = chart_config.get("chart_type", "table")
        title = chart_config.get("title", "Data Visualization")
        x_col = chart_config.get("x_column")
        y_col = chart_config.get("y_column")
        color_col = chart_config.get("color_column")

        try:
            if chart_type == "table":
                fig = self._create_table_figure(data, title)
            elif chart_type == "line":
                fig = self._create_line_figure(data, x_col, y_col, color_col, title)
            elif chart_type == "bar":
                fig = self._create_bar_figure(data, x_col, y_col, color_col, title)
            elif chart_type == "scatter":
                fig = self._create_scatter_figure(data, x_col, y_col, color_col, title)
            elif chart_type == "pie":
                fig = self._create_pie_figure(data, x_col, y_col, title)
            else:
                fig = self._create_table_figure(data, title)

            return fig.to_html(include_plotlyjs=False, div_id="plotly-chart")

        except Exception as e:
            logger.warning(f"Plotly render failed: {e}, falling back")
            return self._render_fallback(chart_config, data)

    def _create_table_figure(self, data: List[Dict[str, Any]], title: str) -> Any:
        """Create Plotly table figure."""
        import plotly.graph_objects as go

        if not data:
            return go.Figure().add_annotation(text="No data available")

        columns = list(data[0].keys())
        rows = [[str(row.get(col, "")) for col in columns] for row in data]

        fig = go.Figure(data=[go.Table(
            header=dict(values=columns),
            cells=dict(values=list(zip(*rows)) if rows else []),
        )])
        fig.update_layout(title_text=title)
        return fig

    def _create_line_figure(
        self,
        data: List[Dict[str, Any]],
        x_col: Optional[str],
        y_col: Optional[str],
        color_col: Optional[str],
        title: str,
    ) -> Any:
        """Create Plotly line figure."""
        import plotly.express as px

        if not x_col or not y_col:
            return self._create_table_figure(data, title)

        return px.line(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
            markers=True,
        )

    def _create_bar_figure(
        self,
        data: List[Dict[str, Any]],
        x_col: Optional[str],
        y_col: Optional[str],
        color_col: Optional[str],
        title: str,
    ) -> Any:
        """Create Plotly bar figure."""
        import plotly.express as px

        if not x_col or not y_col:
            return self._create_table_figure(data, title)

        return px.bar(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
        )

    def _create_scatter_figure(
        self,
        data: List[Dict[str, Any]],
        x_col: Optional[str],
        y_col: Optional[str],
        color_col: Optional[str],
        title: str,
    ) -> Any:
        """Create Plotly scatter figure."""
        import plotly.express as px

        if not x_col or not y_col:
            return self._create_table_figure(data, title)

        return px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
        )

    def _create_pie_figure(
        self,
        data: List[Dict[str, Any]],
        x_col: Optional[str],
        y_col: Optional[str],
        title: str,
    ) -> Any:
        """Create Plotly pie figure."""
        import plotly.express as px

        if not x_col or not y_col:
            return self._create_table_figure(data, title)

        return px.pie(
            data,
            names=x_col,
            values=y_col,
            title=title,
        )

    def _render_fallback(self, chart_config: Dict[str, Any], data: List[Dict[str, Any]]) -> str:
        """Fallback SVG rendering for basic charts."""
        title = chart_config.get("title", "Data Visualization")
        chart_type = chart_config.get("chart_type", "table")

        svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">
  <defs>
    <style>
      .title {{ font-size: 24px; font-weight: bold; fill: #333; }}
      .label {{ font-size: 12px; fill: #666; }}
      .grid {{ stroke: #ddd; stroke-width: 0.5; }}
      .axis {{ stroke: #333; stroke-width: 2; }}
    </style>
  </defs>

  <!-- Background -->
  <rect width="800" height="600" fill="white"/>

  <!-- Title -->
  <text class="title" x="400" y="30" text-anchor="middle">{title}</text>

  <!-- Chart type info -->
  <text class="label" x="400" y="60" text-anchor="middle">
    Chart Type: {chart_type.upper()}
  </text>

  <!-- Data summary -->
  <text class="label" x="50" y="150">Data Points: {len(data)}</text>
  <text class="label" x="50" y="180">Columns: {len(data[0]) if data else 0}</text>

  <!-- Grid -->
  <line class="grid" x1="50" y1="100" x2="750" y2="100"/>
  <line class="grid" x1="50" y1="200" x2="750" y2="200"/>
  <line class="grid" x1="50" y1="300" x2="750" y2="300"/>
  <line class="grid" x1="50" y1="400" x2="750" y2="400"/>
  <line class="grid" x1="50" y1="500" x2="750" y2="500"/>

  <!-- Axes -->
  <line class="axis" x1="50" y1="100" x2="50" y2="550"/>
  <line class="axis" x1="50" y1="550" x2="750" y2="550"/>

  <!-- Data table (fallback) -->
  <g transform="translate(50, 250)">
    <text class="label" x="0" y="20">Raw Data ({len(data)} rows)</text>
  </g>
</svg>"""
        return svg

    def _validate_svg(self, svg: str) -> bool:
        """Validate SVG structure."""
        try:
            from lxml import etree

            etree.fromstring(svg.encode("utf-8"))
            return True
        except ImportError:
            logger.warning("lxml not available, skipping SVG validation")
            return True
        except Exception as e:
            logger.warning(f"SVG validation failed: {e}")
            return False

    def _generate_error_svg(self, error_msg: str) -> str:
        """Generate error SVG."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="600" height="400">
  <rect width="600" height="400" fill="#fff5f5"/>
  <text x="300" y="200" font-size="20" text-anchor="middle" fill="#d32f2f">
    Visualization Error
  </text>
  <text x="300" y="240" font-size="14" text-anchor="middle" fill="#666">
    {error_msg[:80]}
  </text>
</svg>"""

"""
Plotly-based fallback visualization generation.

Creates Plotly figures programmatically from chart configs
and data, with SVG export capabilities.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class PlotlyFallback:
    """Generate visualizations using Plotly as fallback."""

    def __init__(self) -> None:
        """Initialize Plotly fallback renderer."""
        self.has_plotly = self._check_plotly()

    def _check_plotly(self) -> bool:
        """Check if Plotly is available."""
        try:
            import plotly
            import plotly.graph_objects as go
            import plotly.express as px
            return True
        except ImportError:
            logger.warning("Plotly not available for visualization")
            return False

    def create_figure(
        self,
        chart_config: Dict[str, Any],
        data: List[Dict[str, Any]],
    ) -> Optional[Any]:
        """
        Create Plotly figure from configuration.

        Args:
            chart_config: Chart configuration
            data: Data rows

        Returns:
            Plotly Figure object or None
        """
        if not self.has_plotly:
            return None

        try:
            chart_type = chart_config.get("chart_type", "table")
            title = chart_config.get("title", "Data Visualization")
            x_col = chart_config.get("x_column")
            y_col = chart_config.get("y_column")
            color_col = chart_config.get("color_column")

            if chart_type == "table":
                return self._create_table(data, title)
            elif chart_type == "line":
                return self._create_line(data, x_col, y_col, color_col, title)
            elif chart_type == "bar":
                return self._create_bar(data, x_col, y_col, color_col, title)
            elif chart_type == "scatter":
                return self._create_scatter(data, x_col, y_col, color_col, title)
            elif chart_type == "pie":
                return self._create_pie(data, x_col, y_col, title)
            elif chart_type == "histogram":
                return self._create_histogram(data, x_col, title)
            elif chart_type == "box":
                return self._create_box(data, x_col, y_col, title)
            else:
                return self._create_table(data, title)

        except Exception as e:
            logger.error(f"Error creating figure: {e}")
            return None

    def _create_table(self, data: List[Dict[str, Any]], title: str) -> Any:
        """Create table figure."""
        import plotly.graph_objects as go

        if not data:
            fig = go.Figure()
            fig.add_annotation(text="No data to display")
            return fig

        columns = list(data[0].keys())
        rows = [[str(row.get(col, "")) for col in columns] for row in data]

        fig = go.Figure(data=[go.Table(
            header=dict(values=columns, fill_color="lightblue"),
            cells=dict(
                values=list(zip(*rows)) if rows else [],
                fill_color="white",
            ),
        )])

        fig.update_layout(title_text=title, height=500)
        return fig

    def _create_line(
        self,
        data: List[Dict[str, Any]],
        x_col: Optional[str],
        y_col: Optional[str],
        color_col: Optional[str],
        title: str,
    ) -> Any:
        """Create line chart."""
        import plotly.express as px

        if not x_col or not y_col:
            return self._create_table(data, title)

        try:
            return px.line(
                data,
                x=x_col,
                y=y_col,
                color=color_col,
                title=title,
                markers=True,
            )
        except Exception as e:
            logger.warning(f"Line chart creation failed: {e}")
            return self._create_table(data, title)

    def _create_bar(
        self,
        data: List[Dict[str, Any]],
        x_col: Optional[str],
        y_col: Optional[str],
        color_col: Optional[str],
        title: str,
    ) -> Any:
        """Create bar chart."""
        import plotly.express as px

        if not x_col or not y_col:
            return self._create_table(data, title)

        try:
            return px.bar(
                data,
                x=x_col,
                y=y_col,
                color=color_col,
                title=title,
                barmode="group",
            )
        except Exception as e:
            logger.warning(f"Bar chart creation failed: {e}")
            return self._create_table(data, title)

    def _create_scatter(
        self,
        data: List[Dict[str, Any]],
        x_col: Optional[str],
        y_col: Optional[str],
        color_col: Optional[str],
        title: str,
    ) -> Any:
        """Create scatter plot."""
        import plotly.express as px

        if not x_col or not y_col:
            return self._create_table(data, title)

        try:
            return px.scatter(
                data,
                x=x_col,
                y=y_col,
                color=color_col,
                title=title,
                size_max=10,
            )
        except Exception as e:
            logger.warning(f"Scatter plot creation failed: {e}")
            return self._create_table(data, title)

    def _create_pie(
        self,
        data: List[Dict[str, Any]],
        x_col: Optional[str],
        y_col: Optional[str],
        title: str,
    ) -> Any:
        """Create pie chart."""
        import plotly.express as px

        if not x_col or not y_col:
            return self._create_table(data, title)

        try:
            return px.pie(
                data,
                names=x_col,
                values=y_col,
                title=title,
            )
        except Exception as e:
            logger.warning(f"Pie chart creation failed: {e}")
            return self._create_table(data, title)

    def _create_histogram(
        self,
        data: List[Dict[str, Any]],
        x_col: Optional[str],
        title: str,
    ) -> Any:
        """Create histogram."""
        import plotly.express as px

        if not x_col:
            return self._create_table(data, title)

        try:
            return px.histogram(
                data,
                x=x_col,
                title=title,
                nbins=30,
            )
        except Exception as e:
            logger.warning(f"Histogram creation failed: {e}")
            return self._create_table(data, title)

    def _create_box(
        self,
        data: List[Dict[str, Any]],
        x_col: Optional[str],
        y_col: Optional[str],
        title: str,
    ) -> Any:
        """Create box plot."""
        import plotly.express as px

        if not y_col:
            return self._create_table(data, title)

        try:
            return px.box(
                data,
                x=x_col,
                y=y_col,
                title=title,
            )
        except Exception as e:
            logger.warning(f"Box plot creation failed: {e}")
            return self._create_table(data, title)

    def to_svg(self, fig: Any) -> str:
        """
        Convert Plotly figure to SVG.

        Args:
            fig: Plotly Figure object

        Returns:
            SVG string
        """
        try:
            return fig.to_image(format="svg").decode("utf-8")
        except Exception:
            # Fallback to HTML
            logger.warning("SVG export not available, using HTML")
            return fig.to_html()

    def to_html(self, fig: Any) -> str:
        """
        Convert Plotly figure to HTML.

        Args:
            fig: Plotly Figure object

        Returns:
            HTML string
        """
        try:
            return fig.to_html(include_plotlyjs=False)
        except Exception as e:
            logger.error(f"Error converting to HTML: {e}")
            return "<p>Error rendering visualization</p>"

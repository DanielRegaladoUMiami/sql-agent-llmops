"""
Visualization module for generating charts and SVG outputs.

Provides SVG rendering, validation, and Plotly fallback capabilities
for converting data into visual formats.
"""

from src.visualization.svg_validator import SVGValidator
from src.visualization.plotly_fallback import PlotlyFallback

__all__ = ["SVGValidator", "PlotlyFallback"]

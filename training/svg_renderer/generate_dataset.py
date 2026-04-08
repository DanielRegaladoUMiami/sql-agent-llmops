#!/usr/bin/env python3
"""
Script to generate SVG training data programmatically.

Takes chart configurations, generates synthetic data, creates charts with Plotly,
exports to SVG, and cleans SVGs for training.
"""

import json
import random
import logging
from typing import Optional, List, Dict, Any
import re
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datasets import Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generates synthetic data for different chart types."""

    @staticmethod
    def generate_bar_chart_data(
        num_categories: int = 5,
        num_series: int = 1,
    ) -> Dict[str, Any]:
        """
        Generate data for bar chart.

        Args:
            num_categories: Number of categories.
            num_series: Number of data series.

        Returns:
            Dictionary with data and metadata.
        """
        categories = [f"Category_{i}" for i in range(num_categories)]
        data = {"category": categories}

        for series_idx in range(num_series):
            series_name = f"Series_{series_idx}"
            data[series_name] = [random.randint(10, 100) for _ in range(num_categories)]

        return data

    @staticmethod
    def generate_line_chart_data(
        num_points: int = 12,
        num_series: int = 1,
    ) -> Dict[str, Any]:
        """Generate data for line chart."""
        x = list(range(num_points))
        data = {"x": x}

        for series_idx in range(num_series):
            series_name = f"Series_{series_idx}"
            y = [random.randint(20, 200) for _ in x]
            data[series_name] = y

        return data

    @staticmethod
    def generate_scatter_data(
        num_points: int = 50,
    ) -> Dict[str, Any]:
        """Generate data for scatter plot."""
        x = [random.uniform(0, 100) for _ in range(num_points)]
        y = [random.uniform(0, 100) for _ in range(num_points)]

        return {"x": x, "y": y}

    @staticmethod
    def generate_pie_chart_data(
        num_slices: int = 5,
    ) -> Dict[str, Any]:
        """Generate data for pie chart."""
        labels = [f"Slice_{i}" for i in range(num_slices)]
        values = [random.randint(10, 100) for _ in range(num_slices)]

        return {"labels": labels, "values": values}

    @staticmethod
    def generate_histogram_data(
        num_bins: int = 10,
        num_samples: int = 1000,
    ) -> Dict[str, Any]:
        """Generate data for histogram."""
        data = np.random.normal(loc=50, scale=15, size=num_samples)
        return {"data": data.tolist()}

    @staticmethod
    def generate_data(
        chart_type: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate data for given chart type.

        Args:
            chart_type: Type of chart ('bar', 'line', 'scatter', 'pie', 'histogram').
            **kwargs: Additional arguments for specific chart types.

        Returns:
            Generated data dictionary.
        """
        generators = {
            "bar": SyntheticDataGenerator.generate_bar_chart_data,
            "line": SyntheticDataGenerator.generate_line_chart_data,
            "scatter": SyntheticDataGenerator.generate_scatter_data,
            "pie": SyntheticDataGenerator.generate_pie_chart_data,
            "histogram": SyntheticDataGenerator.generate_histogram_data,
        }

        generator = generators.get(chart_type, SyntheticDataGenerator.generate_bar_chart_data)
        return generator(**kwargs)


class PlotlyChartGenerator:
    """Generates Plotly charts from configurations."""

    @staticmethod
    def create_bar_chart(data: Dict[str, Any], config: Dict[str, Any]) -> go.Figure:
        """Create bar chart."""
        fig = go.Figure()

        categories = data.get("category", [])
        for key in data:
            if key != "category":
                fig.add_trace(go.Bar(x=categories, y=data[key], name=key))

        fig.update_layout(
            title=config.get("title", "Bar Chart"),
            xaxis_title=config.get("x_axis", "X Axis"),
            yaxis_title=config.get("y_axis", "Y Axis"),
            hovermode="x unified",
            showlegend=True,
        )

        return fig

    @staticmethod
    def create_line_chart(data: Dict[str, Any], config: Dict[str, Any]) -> go.Figure:
        """Create line chart."""
        fig = go.Figure()

        x = data.get("x", [])
        for key in data:
            if key != "x":
                fig.add_trace(go.Scatter(x=x, y=data[key], mode="lines", name=key))

        fig.update_layout(
            title=config.get("title", "Line Chart"),
            xaxis_title=config.get("x_axis", "X Axis"),
            yaxis_title=config.get("y_axis", "Y Axis"),
            hovermode="x unified",
        )

        return fig

    @staticmethod
    def create_scatter_chart(data: Dict[str, Any], config: Dict[str, Any]) -> go.Figure:
        """Create scatter plot."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data.get("x", []),
            y=data.get("y", []),
            mode="markers",
            marker=dict(size=8),
            name="Data",
        ))

        fig.update_layout(
            title=config.get("title", "Scatter Plot"),
            xaxis_title=config.get("x_axis", "X Axis"),
            yaxis_title=config.get("y_axis", "Y Axis"),
        )

        return fig

    @staticmethod
    def create_pie_chart(data: Dict[str, Any], config: Dict[str, Any]) -> go.Figure:
        """Create pie chart."""
        fig = go.Figure(data=[
            go.Pie(
                labels=data.get("labels", []),
                values=data.get("values", []),
            )
        ])

        fig.update_layout(title=config.get("title", "Pie Chart"))

        return fig

    @staticmethod
    def create_histogram(data: Dict[str, Any], config: Dict[str, Any]) -> go.Figure:
        """Create histogram."""
        fig = go.Figure()

        fig.add_trace(go.Histogram(x=data.get("data", []), nbinsx=30))

        fig.update_layout(
            title=config.get("title", "Histogram"),
            xaxis_title=config.get("x_axis", "X Axis"),
            yaxis_title="Frequency",
        )

        return fig

    @staticmethod
    def create_chart(
        chart_type: str,
        data: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Optional[go.Figure]:
        """
        Create chart from type, data, and configuration.

        Args:
            chart_type: Type of chart.
            data: Chart data.
            config: Chart configuration.

        Returns:
            Plotly Figure or None if creation failed.
        """
        creators = {
            "bar": PlotlyChartGenerator.create_bar_chart,
            "line": PlotlyChartGenerator.create_line_chart,
            "scatter": PlotlyChartGenerator.create_scatter_chart,
            "pie": PlotlyChartGenerator.create_pie_chart,
            "histogram": PlotlyChartGenerator.create_histogram,
        }

        creator = creators.get(chart_type)
        if not creator:
            return None

        try:
            return creator(data, config)
        except Exception as e:
            logger.warning(f"Failed to create {chart_type} chart: {e}")
            return None


class SVGCleaner:
    """Cleans and optimizes SVG code."""

    @staticmethod
    def remove_plotly_artifacts(svg: str) -> str:
        """
        Remove Plotly-specific artifacts from SVG.

        Args:
            svg: Raw SVG string.

        Returns:
            Cleaned SVG string.
        """
        # Remove large data URIs (Plotly mouse tracking)
        svg = re.sub(r'<image[^>]*data:image[^>]*>', '', svg)

        # Remove script tags
        svg = re.sub(r'<script[^>]*>.*?</script>', '', svg, flags=re.DOTALL)

        # Remove xmlns attributes that aren't root
        svg = re.sub(r'xmlns="[^"]*"(?!>)', '', svg)

        # Remove Plotly-specific classes
        svg = re.sub(r'class="[^"]*plotly[^"]*"', '', svg)

        # Minify whitespace
        svg = re.sub(r'\s+', ' ', svg)
        svg = re.sub(r'>\s+<', '><', svg)

        return svg.strip()

    @staticmethod
    def optimize_svg(svg: str) -> str:
        """
        Optimize SVG for smaller file size.

        Args:
            svg: SVG string.

        Returns:
            Optimized SVG string.
        """
        # Round decimal numbers
        svg = re.sub(
            r'(\d+\.\d{4,})',
            lambda m: f"{float(m.group(1)):.2f}",
            svg,
        )

        # Remove empty attributes
        svg = re.sub(r'\s+\w+=""', '', svg)

        return svg


class DatasetGenerator:
    """Generates SVG training dataset."""

    CHART_TYPES = ["bar", "line", "scatter", "pie", "histogram"]

    @staticmethod
    def generate_examples(
        num_examples: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Generate SVG training examples.

        Args:
            num_examples: Number of examples to generate.

        Returns:
            List of training examples.
        """
        logger.info(f"Generating {num_examples} SVG training examples...")

        examples = []
        chart_generator = PlotlyChartGenerator()
        svg_cleaner = SVGCleaner()

        for idx in range(num_examples):
            # Choose random chart type
            chart_type = random.choice(DatasetGenerator.CHART_TYPES)

            # Generate configuration
            config = {
                "type": chart_type,
                "title": f"Chart {idx + 1}",
                "x_axis": "X Axis",
                "y_axis": "Y Axis",
            }

            # Generate data
            data = SyntheticDataGenerator.generate_data(chart_type)

            # Create chart
            fig = chart_generator.create_chart(chart_type, data, config)
            if not fig:
                continue

            # Export to SVG
            try:
                svg = fig.to_image(format="svg").decode("utf-8")
            except Exception:
                # Fallback to HTML SVG
                try:
                    svg = fig.to_html(include_plotlyjs=False, div_id="chart")
                except Exception as e:
                    logger.warning(f"Failed to export SVG: {e}")
                    continue

            # Clean SVG
            svg = svg_cleaner.remove_plotly_artifacts(svg)
            svg = svg_cleaner.optimize_svg(svg)

            # Create training example
            example = {
                "chart_config": json.dumps(config),
                "chart_data": json.dumps(data),
                "svg_code": svg,
            }

            examples.append(example)

        logger.info(f"Generated {len(examples)} examples")
        return examples

    @staticmethod
    def save_dataset(
        examples: List[Dict[str, Any]],
        output_dir: str = "data/svg-rendering",
    ) -> None:
        """
        Save dataset to disk.

        Args:
            examples: List of training examples.
            output_dir: Output directory.
        """
        import os

        logger.info(f"Saving dataset to {output_dir}")

        dataset = Dataset.from_dict({
            "chart_config": [ex["chart_config"] for ex in examples],
            "chart_data": [ex["chart_data"] for ex in examples],
            "svg_code": [ex["svg_code"] for ex in examples],
        })

        os.makedirs(output_dir, exist_ok=True)
        dataset.save_to_disk(output_dir)

        logger.info(f"Saved {len(dataset)} examples")

    @staticmethod
    def push_to_hub(
        examples: List[Dict[str, Any]],
        dataset_name: str = "sql-agent/svg-rendering-training",
        hf_token: Optional[str] = None,
    ) -> None:
        """
        Push dataset to HuggingFace Hub.

        Args:
            examples: List of training examples.
            dataset_name: Name for Hub dataset.
            hf_token: HuggingFace API token.
        """
        logger.info(f"Pushing to Hub: {dataset_name}")

        dataset = Dataset.from_dict({
            "chart_config": [ex["chart_config"] for ex in examples],
            "chart_data": [ex["chart_data"] for ex in examples],
            "svg_code": [ex["svg_code"] for ex in examples],
        })

        try:
            dataset.push_to_hub(dataset_name, token=hf_token, private=True)
            logger.info("Successfully pushed to Hub!")
        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")


def generate_and_save(
    num_examples: int = 500,
    output_dir: str = "data/svg-rendering",
    push_to_hub: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    """
    Main function to generate and save SVG dataset.

    Args:
        num_examples: Number of examples to generate.
        output_dir: Output directory for dataset.
        push_to_hub: Whether to push to HuggingFace Hub.
        hf_token: HuggingFace API token.
    """
    logger.info("=" * 80)
    logger.info("SVG Training Dataset Generation")
    logger.info("=" * 80)

    examples = DatasetGenerator.generate_examples(num_examples)

    DatasetGenerator.save_dataset(examples, output_dir)

    if push_to_hub and hf_token:
        DatasetGenerator.push_to_hub(examples, hf_token=hf_token)

    logger.info("\nDataset generation completed!")


if __name__ == "__main__":
    generate_and_save(num_examples=50)

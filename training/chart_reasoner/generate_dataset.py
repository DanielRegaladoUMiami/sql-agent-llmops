#!/usr/bin/env python3
"""
Knowledge distillation script for chart reasoning training data generation.

Uses free HuggingFace Inference API (Qwen 72B, Llama 70B, Mixtral 8x22B) to generate
chart reasoning training data from SQL examples. Also includes rule-based augmentation.
"""

import time
import json
import random
from typing import Optional, List, Dict, Any
import logging
import requests
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from datasets import load_dataset, Dataset
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ChartReasoningExample:
    """Chart reasoning training example."""

    sql: str
    data_preview: str
    column_info: str
    chart_config: str
    source: str


class RateLimiter:
    """Rate limiter for API calls."""

    def __init__(self, max_calls: int = 10, period_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed per period.
            period_seconds: Period in seconds.
        """
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.call_times: List[float] = []

    def wait_if_needed(self) -> None:
        """Wait if necessary to stay within rate limits."""
        now = time.time()

        # Remove old calls outside the period
        self.call_times = [t for t in self.call_times if now - t < self.period_seconds]

        # If at limit, wait
        if len(self.call_times) >= self.max_calls:
            sleep_time = self.period_seconds - (now - self.call_times[0]) + 0.1
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

        self.call_times.append(time.time())


class TeacherModelAPI:
    """Interface to HuggingFace Inference API."""

    MODELS = {
        "qwen": "Qwen/Qwen2-72B-Instruct",
        "llama": "meta-llama/Llama-2-70b-chat-hf",
        "mixtral": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    }

    def __init__(self, hf_token: Optional[str] = None, model: str = "qwen"):
        """
        Initialize teacher model API.

        Args:
            hf_token: HuggingFace API token.
            model: Model to use ('qwen', 'llama', 'mixtral').
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.model_name = self.MODELS.get(model, self.MODELS["qwen"])
        self.rate_limiter = RateLimiter(max_calls=5, period_seconds=60)

        if not self.hf_token:
            logger.warning("HF_TOKEN not provided, API calls will fail")

    def query(self, prompt: str) -> Optional[str]:
        """
        Query teacher model via HF Inference API.

        Args:
            prompt: Input prompt for the model.

        Returns:
            Model output or None if failed.
        """
        self.rate_limiter.wait_if_needed()

        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        }

        try:
            api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
            else:
                logger.warning(f"API returned status {response.status_code}")

        except requests.Timeout:
            logger.warning("API request timed out")
        except Exception as e:
            logger.warning(f"API error: {e}")

        return None


class ChartConfigValidator:
    """Validates generated chart configurations."""

    VALID_CHART_TYPES = [
        "bar",
        "line",
        "scatter",
        "pie",
        "area",
        "histogram",
        "box",
        "heatmap",
        "bubble",
        "funnel",
    ]

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate chart configuration.

        Args:
            config: Chart configuration dictionary.

        Returns:
            True if valid, False otherwise.
        """
        if not isinstance(config, dict):
            return False

        if "type" not in config:
            return False

        if config.get("type") not in ChartConfigValidator.VALID_CHART_TYPES:
            return False

        # Check required fields
        if "title" not in config:
            return False

        return True

    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text.

        Args:
            text: Text containing JSON.

        Returns:
            Parsed JSON or None.
        """
        try:
            # Try to find JSON block
            if "{" in text and "}" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                json_str = text[start:end]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")

        return None


class RuleBasedChartGenerator:
    """Rule-based chart configuration generation."""

    @staticmethod
    def infer_chart_type(
        columns: List[str],
        column_types: Dict[str, str],
        query: str,
    ) -> str:
        """
        Infer chart type based on query and column types.

        Args:
            columns: List of column names.
            column_types: Dictionary mapping column names to types.
            query: SQL query.

        Returns:
            Recommended chart type.
        """
        query_lower = query.lower()

        # Time series
        if any(x in query_lower for x in ["over time", "temporal", "date", "month", "year"]):
            return "line"

        # Composition/parts of whole
        if "composition" in query_lower or "percentage" in query_lower:
            return "pie"

        # Comparison
        if "compare" in query_lower or "vs" in query_lower:
            return "bar"

        # Correlation
        if "correlation" in query_lower or "relationship" in query_lower:
            return "scatter"

        # Default: bar chart
        return "bar"

    @staticmethod
    def generate_config(
        columns: List[str],
        column_types: Dict[str, str],
        query: str,
        title: str,
    ) -> Dict[str, Any]:
        """
        Generate chart configuration.

        Args:
            columns: Column names.
            column_types: Column type mapping.
            query: SQL query.
            title: Chart title.

        Returns:
            Chart configuration dictionary.
        """
        chart_type = RuleBasedChartGenerator.infer_chart_type(
            columns,
            column_types,
            query,
        )

        # Find numeric and categorical columns
        numeric_cols = [c for c in columns if column_types.get(c) == "numeric"]
        categorical_cols = [c for c in columns if column_types.get(c) == "categorical"]

        config = {
            "type": chart_type,
            "title": title,
        }

        # Add axes based on chart type and columns
        if chart_type in ["bar", "line", "area"]:
            if categorical_cols and numeric_cols:
                config["x_axis"] = categorical_cols[0]
                config["y_axis"] = numeric_cols[0]
            elif len(numeric_cols) >= 2:
                config["x_axis"] = numeric_cols[0]
                config["y_axis"] = numeric_cols[1]

        elif chart_type == "scatter":
            if len(numeric_cols) >= 2:
                config["x_axis"] = numeric_cols[0]
                config["y_axis"] = numeric_cols[1]
                if len(numeric_cols) >= 3:
                    config["size"] = numeric_cols[2]

        elif chart_type == "pie":
            if categorical_cols and numeric_cols:
                config["labels"] = categorical_cols[0]
                config["values"] = numeric_cols[0]

        return config


class DatasetGenerator:
    """Generates chart reasoning training dataset."""

    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize dataset generator.

        Args:
            hf_token: HuggingFace API token.
        """
        self.teacher_model = TeacherModelAPI(hf_token=hf_token)
        self.validator = ChartConfigValidator()

    def generate_from_sql_examples(
        self,
        sql_dataset: Dataset,
        num_examples: int = 100,
        use_rule_based: bool = True,
        use_api: bool = False,
    ) -> List[ChartReasoningExample]:
        """
        Generate chart reasoning examples from SQL dataset.

        Args:
            sql_dataset: SQL examples dataset.
            num_examples: Number of examples to generate.
            use_rule_based: Whether to use rule-based generation.
            use_api: Whether to use teacher API for generation.

        Returns:
            List of generated chart reasoning examples.
        """
        logger.info(f"Generating {num_examples} chart reasoning examples...")

        examples = []
        sample_indices = random.sample(range(len(sql_dataset)), min(num_examples, len(sql_dataset)))

        for idx in sample_indices:
            example = sql_dataset[idx]
            sql = example.get("sql", "")
            context = example.get("context", "")

            if not sql or not context:
                continue

            # Extract basic info from SQL
            title = self._extract_title_from_query(sql)
            columns = self._extract_columns_from_query(sql)
            column_types = self._infer_column_types(columns)

            # Generate sample data
            data_preview = self._generate_sample_data(columns, column_types)

            if use_api and random.random() < 0.5:
                # Try API-based generation
                chart_config = self._generate_via_api(
                    sql,
                    context,
                    data_preview,
                    title,
                )
                source = "teacher-api"
            else:
                # Use rule-based generation
                chart_config_dict = RuleBasedChartGenerator.generate_config(
                    columns,
                    column_types,
                    sql,
                    title,
                )
                chart_config = json.dumps(chart_config_dict)
                source = "rule-based"

            if chart_config:
                column_info = self._format_column_info(columns, column_types)

                examples.append(
                    ChartReasoningExample(
                        sql=sql,
                        data_preview=data_preview,
                        column_info=column_info,
                        chart_config=chart_config,
                        source=source,
                    )
                )

        logger.info(f"Generated {len(examples)} examples")
        return examples

    def _extract_title_from_query(self, sql: str) -> str:
        """Extract a title from SQL query."""
        # Simple heuristic: capitalize key words
        if "SELECT" in sql:
            parts = sql.split("FROM")
            if len(parts) > 0:
                return "Query Results"
        return "Data Visualization"

    def _extract_columns_from_query(self, sql: str) -> List[str]:
        """Extract column names from SQL query."""
        # Very basic parsing
        if "SELECT" in sql:
            select_part = sql.split("SELECT")[1].split("FROM")[0]
            columns = [c.strip() for c in select_part.split(",")]
            return columns[:5]  # Limit to 5 columns
        return ["col1", "col2"]

    def _infer_column_types(self, columns: List[str]) -> Dict[str, str]:
        """Infer column types from names."""
        numeric_keywords = ["count", "sum", "total", "amount", "price", "value", "revenue"]
        types = {}

        for col in columns:
            if any(kw in col.lower() for kw in numeric_keywords):
                types[col] = "numeric"
            else:
                types[col] = "categorical"

        return types

    def _generate_sample_data(
        self,
        columns: List[str],
        column_types: Dict[str, str],
    ) -> str:
        """Generate sample data for preview."""
        rows = []
        for _ in range(3):
            row = {}
            for col in columns:
                if column_types.get(col) == "numeric":
                    row[col] = random.randint(100, 10000)
                else:
                    row[col] = f"Category_{random.randint(1, 5)}"
            rows.append(row)

        df = pd.DataFrame(rows)
        return df.to_string(index=False)

    def _format_column_info(
        self,
        columns: List[str],
        column_types: Dict[str, str],
    ) -> str:
        """Format column information."""
        info = []
        for col in columns:
            col_type = column_types.get(col, "unknown")
            info.append(f"- {col}: {col_type}")
        return "\n".join(info)

    def _generate_via_api(
        self,
        sql: str,
        context: str,
        data_preview: str,
        title: str,
    ) -> Optional[str]:
        """Generate chart config via teacher API."""
        prompt = f"""Given the following SQL query result, recommend a chart configuration in JSON format.

SQL Query:
{sql}

Data Preview:
{data_preview}

Context:
{context}

Respond ONLY with valid JSON configuration for a chart, like:
{{"type": "bar", "x_axis": "...", "y_axis": "...", "title": "{title}"}}"""

        response = self.teacher_model.query(prompt)

        if response:
            config = self.validator.extract_json(response)
            if config and self.validator.validate_config(config):
                return json.dumps(config)

        return None


def generate_and_save(
    output_dataset_name: str = "sql-agent/chart-reasoning-training",
    num_examples: int = 500,
    push_to_hub: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    """
    Generate and optionally push chart reasoning dataset.

    Args:
        output_dataset_name: Name for output dataset.
        num_examples: Number of examples to generate.
        push_to_hub: Whether to push to HuggingFace Hub.
        hf_token: HuggingFace API token.
    """
    logger.info("=" * 80)
    logger.info("Chart Reasoning Dataset Generation (Knowledge Distillation)")
    logger.info("=" * 80)

    # Load SQL dataset
    logger.info("\nLoading SQL examples...")
    try:
        sql_dataset = load_dataset("sql-agent/sql-training-unified", split="train")
    except Exception as e:
        logger.warning(f"Could not load dataset: {e}, using synthetic data")
        sql_dataset = Dataset.from_dict({
            "sql": ["SELECT * FROM users"],
            "context": ["CREATE TABLE users (id INT, name VARCHAR(100))"],
        })

    # Generate examples
    generator = DatasetGenerator(hf_token=hf_token)
    examples = generator.generate_from_sql_examples(
        sql_dataset,
        num_examples=num_examples,
        use_rule_based=True,
        use_api=False,  # Set to True if you have API access
    )

    # Convert to HF dataset
    logger.info("\nConverting to HuggingFace format...")
    data = {
        "sql": [ex.sql for ex in examples],
        "data_preview": [ex.data_preview for ex in examples],
        "column_info": [ex.column_info for ex in examples],
        "chart_config": [ex.chart_config for ex in examples],
        "source": [ex.source for ex in examples],
    }

    dataset = Dataset.from_dict(data)

    # Save locally
    import os
    output_dir = "data/chart-reasoning"
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    logger.info(f"Saved to {output_dir}")

    # Push to hub if requested
    if push_to_hub and hf_token:
        logger.info(f"Pushing to Hub: {output_dataset_name}")
        try:
            dataset.push_to_hub(output_dataset_name, token=hf_token, private=True)
            logger.info("Successfully pushed!")
        except Exception as e:
            logger.error(f"Failed to push: {e}")


if __name__ == "__main__":
    import os

    generate_and_save(num_examples=100, push_to_hub=False)

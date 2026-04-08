"""
Chart Reasoner model that determines optimal visualizations for data.

Analyzes query results and recommends chart type, configuration,
and display parameters. Returns structured chart config JSON.
"""

import json
import logging
from typing import Optional, Any, Dict, List

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class ChartReasoner(BaseModel):
    """Reason about optimal chart visualizations for SQL query results."""

    def __init__(
        self,
        model_name: str = "chart-reasoner",
        model_path: Optional[str] = None,
        hf_model: str = "mistralai/Mistral-7B-Instruct-v0.1",
        temperature: float = 0.2,
        max_tokens: int = 300,
    ) -> None:
        """
        Initialize Chart Reasoner model.

        Args:
            model_name: Model identifier
            model_path: Path to model
            hf_model: Hugging Face model identifier
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__(model_name, model_path)
        self.hf_model = hf_model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def load(self) -> None:
        """Load Chart Reasoner model from Hugging Face."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            logger.info(f"Loading Chart Reasoner model: {self.hf_model}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
            )
            self.is_loaded = True
            logger.info("Chart Reasoner model loaded successfully")

        except ImportError:
            logger.error("transformers not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(
        self,
        question: str,
        sql: str,
        results: List[Dict[str, Any]],
        columns: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Generate chart configuration from query results.

        Args:
            question: Original question
            sql: SQL query executed
            results: Query result rows
            columns: Column metadata

        Returns:
            Chart config dictionary with keys:
            - chart_type: str (line, bar, scatter, pie, etc.)
            - title: str
            - x_column: str or None
            - y_column: str or None
            - color_column: str or None
            - config: dict with chart-specific options
        """
        self._validate_loaded()

        try:
            prompt = self._build_prompt(question, sql, results, columns)
            response = self._generate(prompt)
            config = self._parse_config(response)

            logger.info(f"Generated chart config: {config['chart_type']}")
            return config

        except Exception as e:
            logger.error(f"Error generating chart config: {e}")
            return self._default_config()

    def _generate(self, prompt: str) -> str:
        """Generate using model."""
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.95,
                do_sample=True,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def _build_prompt(
        self,
        question: str,
        sql: str,
        results: List[Dict[str, Any]],
        columns: List[Dict[str, str]],
    ) -> str:
        """Build prompt for chart reasoning."""
        sample_rows = results[:3] if results else []
        column_names = [col["name"] for col in columns]

        prompt = f"""You are an expert data visualization specialist. Analyze the SQL query results and recommend the best chart type.

Question: {question}
SQL Query: {sql}

Columns: {column_names}
Sample Data:
{json.dumps(sample_rows, indent=2)}

Based on the data, recommend a visualization. Respond with ONLY a JSON object (no markdown) with these keys:
- chart_type: one of [line, bar, scatter, pie, histogram, box, heatmap, table]
- title: descriptive title
- x_column: column for X axis (null for pie/table)
- y_column: column for Y axis (null for pie/table)
- color_column: optional column for color encoding
- show_legend: boolean
- show_grid: boolean

JSON Response:"""
        return prompt

    def _parse_config(self, response: str) -> Dict[str, Any]:
        """Parse JSON config from response."""
        try:
            lines = response.strip().split("\n")
            json_str = ""
            in_json = False

            for line in lines:
                if "{" in line:
                    in_json = True
                if in_json:
                    json_str += line

            config = json.loads(json_str)

            # Validate required fields
            if "chart_type" not in config:
                config["chart_type"] = "table"

            config = {
                "chart_type": config.get("chart_type", "table"),
                "title": config.get("title", "Data Visualization"),
                "x_column": config.get("x_column"),
                "y_column": config.get("y_column"),
                "color_column": config.get("color_column"),
                "config": {
                    "show_legend": config.get("show_legend", True),
                    "show_grid": config.get("show_grid", True),
                },
            }

            return config

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse chart config JSON: {e}")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default chart configuration."""
        return {
            "chart_type": "table",
            "title": "Query Results",
            "x_column": None,
            "y_column": None,
            "color_column": None,
            "config": {
                "show_legend": True,
                "show_grid": True,
            },
        }

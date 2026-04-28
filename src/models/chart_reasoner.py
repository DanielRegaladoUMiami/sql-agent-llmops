"""
Chart Reasoner: query results -> chart spec via the trained Phi-3 Mini LoRA.

Uses the adapter-only repo so the LoRA loads on top of the original
Phi-3-mini-4k-instruct base, keeping Hub downloads small.
"""

import json
import logging
import re
from typing import Any, Dict, List

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a data visualization expert. Given a question, the SQL that "
    "answers it, and a sample of the result rows, produce a JSON chart "
    "specification. Choose the chart type that tells the clearest story. "
    "Return only valid JSON, no commentary."
)


class ChartReasoner(BaseModel):
    """Generate chart specs from SQL result sets."""

    DEFAULT_MERGED = "DanielRegaladoCardoso/chart-reasoner-phi3-mini-lora"

    def __init__(
        self,
        hf_model: str = DEFAULT_MERGED,
        temperature: float = 0.0,
        max_new_tokens: int = 300,
    ) -> None:
        super().__init__(model_name="chart-reasoner")
        self.hf_model = hf_model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info(f"Loading chart reasoner: {self.hf_model}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_model,
            torch_dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        self.is_loaded = True
        logger.info(f"Chart reasoner loaded on {device}")

    def generate(  # type: ignore[override]
        self,
        question: str,
        sql: str,
        results: List[Dict[str, Any]],
        columns: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        self._validate_loaded()
        import torch

        sample = results[:5]
        col_names = [c["name"] for c in columns]
        user_content = (
            f"Question: {question}\n"
            f"SQL: {sql}\n"
            f"Columns: {col_names}\n"
            f"Sample rows: {json.dumps(sample, default=str)}\n\n"
            "Return JSON with: chart_type (one of: bar, line, scatter, "
            "pie, area, table), title, x_column, y_column, "
            "color_column (optional), rationale."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(
            out[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        return self._parse_spec(raw, columns)

    def _parse_spec(
        self, text: str, columns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        # Try to extract a JSON object from the response
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if not match:
            logger.warning("No JSON found in chart reasoner output")
            return self._fallback_spec(columns)
        try:
            spec = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            logger.warning(f"Chart spec JSON invalid: {e}")
            return self._fallback_spec(columns)

        # Normalize
        return {
            "chart_type": spec.get("chart_type", "bar").lower(),
            "title": spec.get("title", "Result"),
            "x_column": spec.get("x_column"),
            "y_column": spec.get("y_column"),
            "color_column": spec.get("color_column"),
            "rationale": spec.get("rationale", ""),
        }

    def _fallback_spec(self, columns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Heuristic fallback when the model output can't be parsed."""
        if not columns:
            return {"chart_type": "table", "title": "Result"}
        if len(columns) == 1:
            return {
                "chart_type": "table",
                "title": "Result",
                "x_column": columns[0]["name"],
                "y_column": None,
            }
        return {
            "chart_type": "bar",
            "title": "Result",
            "x_column": columns[0]["name"],
            "y_column": columns[1]["name"],
            "color_column": None,
        }

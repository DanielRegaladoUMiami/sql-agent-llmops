"""
Chart Reasoner: load the trained LoRA on top of Phi-3 Mini base.
"""

import json
import logging
import re
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a data visualization expert. Given a question, the SQL that "
    "answers it, and a sample of the result rows, produce a JSON chart "
    "specification. Return only valid JSON, no commentary."
)

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_REPO = "DanielRegaladoCardoso/chart-reasoner-phi3-mini-adapter-only"


class ChartReasoner:

    def __init__(self, temperature: float = 0.0, max_new_tokens: int = 300) -> None:
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        logger.info(f"Loading chart base: {BASE_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        logger.info(f"Applying LoRA adapter: {ADAPTER_REPO}")
        self.model = PeftModel.from_pretrained(
            base,
            ADAPTER_REPO,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        logger.info("Chart reasoner ready")

    def generate(
        self,
        question: str,
        sql: str,
        results: List[Dict[str, Any]],
        columns: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        sample = results[:5]
        col_names = [c["name"] for c in columns]
        user_content = (
            f"Question: {question}\n"
            f"SQL: {sql}\n"
            f"Columns: {col_names}\n"
            f"Sample rows: {json.dumps(sample, default=str)}\n\n"
            "Return JSON with: chart_type (one of: bar, line, scatter, pie, "
            "area, table), title, x_column, y_column, color_column "
            "(optional), rationale."
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

    def _parse_spec(self, text: str, columns: List[Dict[str, Any]]) -> Dict[str, Any]:
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if not match:
            return self._fallback_spec(columns)
        try:
            spec = json.loads(match.group(0))
        except json.JSONDecodeError:
            return self._fallback_spec(columns)
        return {
            "chart_type": spec.get("chart_type", "bar").lower(),
            "title": spec.get("title", "Result"),
            "x_column": spec.get("x_column"),
            "y_column": spec.get("y_column"),
            "color_column": spec.get("color_column"),
            "rationale": spec.get("rationale", ""),
        }

    def _fallback_spec(self, columns: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not columns:
            return {"chart_type": "table", "title": "Result"}
        if len(columns) == 1:
            return {"chart_type": "table", "title": "Result",
                    "x_column": columns[0]["name"], "y_column": None}
        return {"chart_type": "bar", "title": "Result",
                "x_column": columns[0]["name"],
                "y_column": columns[1]["name"], "color_column": None}

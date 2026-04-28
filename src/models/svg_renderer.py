"""
SVG Renderer: chart spec + data -> inline SVG.

Model loaded at root module level (ZeroGPU best practice). If the model
output isn't a valid SVG, falls back to themed Plotly.
"""

import logging
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.visualization.plotly_fallback import PlotlyRenderer
from src.visualization.svg_theme import apply_theme, is_renderable_svg

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an SVG chart artist. Given a chart spec and a small data "
    "sample, produce a single inline SVG visualization. Use a clean, "
    "minimalist style. Return only the SVG, starting with <svg."
)

DEFAULT_MODEL = "DanielRegaladoCardoso/svg-renderer-deepseek-coder-1.3b-lora"


class SVGRenderer:
    """Render a chart spec to inline SVG."""

    def __init__(
        self,
        hf_model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        max_new_tokens: int = 1500,
    ) -> None:
        self.hf_model = hf_model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._plotly = PlotlyRenderer()

        logger.info(f"Loading SVG renderer at module level: {self.hf_model}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            )
            self.model.eval()
            logger.info("SVG renderer ready")
        except Exception as e:
            logger.warning(f"SVG model load failed ({e}); will use Plotly fallback only")
            self.model = None
            self.tokenizer = None

    def generate(
        self,
        chart_spec: Dict[str, Any],
        data: List[Dict[str, Any]],
    ) -> str:
        if self.model is not None and self.tokenizer is not None:
            try:
                svg = self._generate_model(chart_spec, data)
                if is_renderable_svg(svg):
                    return apply_theme(svg)
                logger.info("Model SVG failed validation; using Plotly fallback")
            except Exception as e:
                logger.warning(f"Model SVG generation error: {e}; falling back")

        svg = self._plotly.render(chart_spec, data)
        return apply_theme(svg)

    def _generate_model(
        self, chart_spec: Dict[str, Any], data: List[Dict[str, Any]]
    ) -> str:
        import json

        sample = data[:50]
        user_content = (
            f"Chart spec: {json.dumps(chart_spec, default=str)}\n"
            f"Data ({len(data)} rows, showing {len(sample)}): "
            f"{json.dumps(sample, default=str)}\n\n"
            "Render an inline SVG. Use viewBox 0 0 600 400."
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
        text = self.tokenizer.decode(
            out[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        return self._extract_svg(text)

    @staticmethod
    def _extract_svg(text: str) -> str:
        import re
        m = re.search(r"<svg[\s\S]*?</svg>", text, re.IGNORECASE)
        return m.group(0) if m else text.strip()

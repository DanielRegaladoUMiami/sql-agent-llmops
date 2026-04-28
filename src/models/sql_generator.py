"""
SQL Generator: text-to-SQL via the trained Qwen2.5-Coder-7B LoRA.

Loads at module import time (root level), as required by ZeroGPU best
practices. Inference happens inside @spaces.GPU in the orchestrator.
"""

import logging
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a SQL expert. Given a SQL schema and a natural-language "
    "question, generate a correct SQL query answering the question. "
    "Return only the SQL."
)

DEFAULT_MODEL = "DanielRegaladoCardoso/sql-generator-qwen25-coder-7b-lora"


class SQLGenerator:
    """Text-to-SQL generator. Model loaded at construction time onto CUDA."""

    def __init__(
        self,
        hf_model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_new_tokens: int = 400,
    ) -> None:
        self.hf_model = hf_model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        logger.info(f"Loading SQL generator at module level: {self.hf_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
        # On ZeroGPU, device_map='cuda' uses emulation mode at module load and
        # real GPU inside @spaces.GPU calls.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_model,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()
        logger.info("SQL generator ready")

    def generate(self, question: str, schema: str) -> str:
        user_content = f"### Schema\n{schema}\n\n### Question\n{question}"
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
        return self._clean_sql(text)

    @staticmethod
    def _clean_sql(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
        if ";" in text:
            stmt, _, _ = text.partition(";")
            text = stmt + ";"
        return text.strip()

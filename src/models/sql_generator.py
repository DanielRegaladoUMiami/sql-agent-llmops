"""
SQL Generator: text-to-SQL via the trained Qwen2.5-Coder-7B LoRA.

Loads the merged 16-bit checkpoint from Hugging Face Hub by default.
The same `DanielRegaladoCardoso/sql-generator-qwen25-coder-7b-lora` repo
contains both the LoRA adapter and the merged model.
"""

import logging
import re
from typing import Optional

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a SQL expert. Given a SQL schema and a natural-language "
    "question, generate a correct SQL query answering the question. "
    "Return only the SQL."
)


class SQLGenerator(BaseModel):
    """Text-to-SQL generator using the trained Qwen2.5-Coder-7B model."""

    DEFAULT_MODEL = "DanielRegaladoCardoso/sql-generator-qwen25-coder-7b-lora"

    def __init__(
        self,
        hf_model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_new_tokens: int = 400,
    ) -> None:
        super().__init__(model_name="sql-generator")
        self.hf_model = hf_model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info(f"Loading SQL generator: {self.hf_model}")
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
        logger.info(f"SQL generator loaded on {device}")

    def generate(self, question: str, schema: str) -> str:  # type: ignore[override]
        self._validate_loaded()
        import torch

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
        sql = self._clean_sql(text)
        logger.info(f"Generated SQL: {sql[:100]}...")
        return sql

    @staticmethod
    def _clean_sql(text: str) -> str:
        """Strip code fences, trailing prose, ensure a single SQL statement."""
        text = text.strip()
        # Strip fences ```sql ... ```
        text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
        # Cut at the first ;  if followed by prose
        if ";" in text:
            stmt, _, _ = text.partition(";")
            text = stmt + ";"
        return text.strip()

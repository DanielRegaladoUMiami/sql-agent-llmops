"""
SQL Generator: load the trained LoRA adapter on top of the standard Qwen
2.5 Coder 7B base. Loaded at module level per ZeroGPU best practice.
"""

import logging
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert SQL analyst working with a DuckDB database.

Your job: convert a natural-language question into ONE correct SQL query.

## Rules
1. **Map user wording to schema columns.** Users rarely use the exact column name. Infer which column they mean from the schema, sample rows, and any distinct-value hints. Examples: "shoe size" → use the "size" column; "where" → likely a "country" or "city" column; "biggest" → ORDER BY DESC LIMIT.
2. **Use ONLY the exact column names that appear in the CREATE TABLE statements.** Wrap them in double quotes when they contain spaces or reserved words.
3. **DuckDB syntax** (PostgreSQL-flavored): supports CTEs, window functions, LIMIT, OFFSET, regex, date arithmetic, JSON functions.
4. **Aggregations**: alias them descriptively, e.g. `AVG(price) AS avg_price`, `COUNT(*) AS total`.
5. **Default to TOP 10** when the user asks for "top", "best", "most" without specifying a number.
6. **Filter explicitly**: if the user mentions a categorical value (e.g. "active customers"), match it against distinct values in the hints and use the exact spelling.
7. Output **only the SQL**. No markdown fences, no explanation. End with a semicolon.

## Examples
Schema: CREATE TABLE sales ("id" INT, "product_name" VARCHAR, "revenue" DOUBLE);
Question: top earners
SQL: SELECT product_name, revenue FROM sales ORDER BY revenue DESC LIMIT 10;

Schema: CREATE TABLE coffee ("age" INT, "wait_time_sec" DOUBLE, "method" VARCHAR);
-- coffee.method distinct values: 'espresso', 'pour_over', 'french_press'
Question: average wait by brewing style
SQL: SELECT method, AVG(wait_time_sec) AS avg_wait FROM coffee GROUP BY method ORDER BY avg_wait DESC;

Schema: CREATE TABLE companies ("name" VARCHAR, "founded" INT, "country" VARCHAR);
Question: how many startups per region
SQL: SELECT country, COUNT(*) AS total FROM companies GROUP BY country ORDER BY total DESC;
"""

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_REPO = "DanielRegaladoCardoso/sql-generator-qwen25-coder-7b-lora"


class SQLGenerator:

    def __init__(self, temperature: float = 0.0, max_new_tokens: int = 400) -> None:
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        logger.info(f"Loading SQL base: {BASE_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        logger.info(f"Applying LoRA adapter: {ADAPTER_REPO}")
        self.model = PeftModel.from_pretrained(
            base,
            ADAPTER_REPO,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        logger.info("SQL generator ready (LoRA applied on Qwen base)")

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

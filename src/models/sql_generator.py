"""
SQL Generator model that converts natural language to SQL queries.

Uses either llama-cpp-python for GGUF models or transformers for Hugging Face models.
Includes prompt engineering for robust SQL generation.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class SQLGenerator(BaseModel):
    """Generate SQL queries from natural language questions using LLMs."""

    def __init__(
        self,
        model_name: str = "sql-generator",
        model_path: Optional[str] = None,
        use_gguf: bool = False,
        hf_model: str = "mistralai/Mistral-7B-Instruct-v0.1",
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> None:
        """
        Initialize SQL Generator model.

        Args:
            model_name: Model identifier
            model_path: Path to GGUF file or HF model directory
            use_gguf: Whether to use GGUF format (via llama-cpp-python)
            hf_model: Hugging Face model identifier if not using GGUF
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
        """
        super().__init__(model_name, model_path)
        self.use_gguf = use_gguf
        self.hf_model = hf_model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def load(self) -> None:
        """Load SQL Generator model from specified source."""
        if self.use_gguf:
            self._load_gguf()
        else:
            self._load_huggingface()

    def _load_gguf(self) -> None:
        """Load GGUF model via llama-cpp-python."""
        try:
            from llama_cpp import Llama

            if not self.model_path or not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"GGUF model not found: {self.model_path}. "
                    "Download a GGUF model or use Hugging Face models instead."
                )

            logger.info(f"Loading GGUF model from {self.model_path}")
            self.model = Llama(
                model_path=self.model_path,
                n_gpu_layers=-1,
                verbose=False,
            )
            self.is_loaded = True
            logger.info("GGUF model loaded successfully")

        except ImportError:
            logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            raise
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            raise

    def _load_huggingface(self) -> None:
        """Load model from Hugging Face."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            logger.info(f"Loading Hugging Face model: {self.hf_model}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
            )
            self.is_loaded = True
            logger.info("Hugging Face model loaded successfully")

        except ImportError:
            logger.error("transformers not installed. Install with: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model: {e}")
            raise

    def generate(
        self,
        question: str,
        schema: str,
        additional_context: str = "",
    ) -> str:
        """
        Generate SQL from natural language question.

        Args:
            question: Natural language question
            schema: Database schema description
            additional_context: Optional additional context

        Returns:
            Generated SQL query string
        """
        self._validate_loaded()

        prompt = self._build_prompt(question, schema, additional_context)

        try:
            if self.use_gguf:
                response = self._generate_gguf(prompt)
            else:
                response = self._generate_huggingface(prompt)

            sql = self._extract_sql(response)
            logger.info(f"Generated SQL: {sql[:100]}...")
            return sql

        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise

    def _generate_gguf(self, prompt: str) -> str:
        """Generate using GGUF model."""
        response = self.model(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=0.95,
            stop=["SELECT", ";"],
        )
        return response["choices"][0]["text"]

    def _generate_huggingface(self, prompt: str) -> str:
        """Generate using Hugging Face model."""
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

    def _build_prompt(self, question: str, schema: str, context: str = "") -> str:
        """Build prompt for SQL generation."""
        prompt = f"""You are an expert SQL query generator. Generate a single, valid SQL query that answers the following question.

Database Schema:
{schema}

{f"Additional Context: {context}" if context else ""}

Question: {question}

Generate only the SQL query without any explanation. Output the SQL query directly:
SQL:"""
        return prompt

    def _extract_sql(self, response: str) -> str:
        """Extract SQL from model response."""
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.upper().startswith(("SELECT", "WITH")):
                return line
        return response.strip()

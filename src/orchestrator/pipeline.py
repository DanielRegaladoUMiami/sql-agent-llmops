"""
SQL Agent orchestrator.

Holds an in-memory DuckDB connection and the three specialist models, and
walks a question through the pipeline:

    schema (DuckDB) -> SQL (Qwen) -> execute (DuckDB)
                    -> chart spec (Phi-3) -> SVG (DeepSeek + theme)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.models.chart_reasoner import ChartReasoner
from src.models.sql_generator import SQLGenerator
from src.models.svg_renderer import SVGRenderer
from src.rag.engine import RAGEngine
from src.utils.sql_executor import SQLExecutor

logger = logging.getLogger(__name__)


class SQLAgentOrchestrator:
    """End-to-end NL -> SQL -> chart pipeline backed by DuckDB."""

    def __init__(self) -> None:
        self.executor = SQLExecutor()
        self.rag = RAGEngine(self.executor.con)

        # Models are constructed eagerly but loaded lazily (HF Spaces ZeroGPU
        # gives us a GPU only inside @spaces.GPU calls, so model.load() must
        # happen there, not at import time).
        self.sql_generator = SQLGenerator()
        self.chart_reasoner = ChartReasoner()
        self.svg_renderer = SVGRenderer()

    # --------------------------------------------------------------- data
    def load_data(
        self,
        source: Union[str, Path, pd.DataFrame],
        table_name: Optional[str] = None,
    ) -> str:
        """Register a DataFrame or file as a queryable table. Returns the table name."""
        if isinstance(source, pd.DataFrame):
            name = table_name or "data"
            self.executor.register_dataframe(name, source)
            return name
        return self.executor.register_file(source, table_name)

    def schema_text(self) -> str:
        return self.rag.retrieve("", top_k=5)

    def list_tables(self) -> List[str]:
        return self.executor.get_table_names()

    def sample(self, table: str, n: int = 5) -> pd.DataFrame:
        return self.executor.get_sample(table, n)

    # ----------------------------------------------------------- pipeline
    def process(self, question: str) -> Dict[str, Any]:
        """
        Run the full pipeline for one question.

        Models are loaded and unloaded sequentially to keep peak VRAM low
        (only one of the 3 models lives in GPU at a time).
        """
        result: Dict[str, Any] = {
            "question": question,
            "sql": None,
            "results": [],
            "columns": [],
            "chart_spec": None,
            "svg": None,
            "error": None,
        }

        try:
            schema = self.schema_text()
            if not schema:
                result["error"] = "No data loaded. Upload a CSV/JSON first."
                return result

            # 1) SQL — load Qwen, generate, unload
            logger.info("Step 1/4: SQL generation")
            self.sql_generator.load()
            sql = self.sql_generator.generate(question=question, schema=schema)
            self.sql_generator.unload()
            result["sql"] = sql

            if not self.executor.validate_query(sql):
                result["error"] = f"Generated SQL is invalid:\n{sql}"
                return result

            # 2) Execute (CPU-only, no model needed)
            logger.info("Step 2/4: SQL execution")
            rows, cols = self.executor.execute(sql)
            result["results"] = rows
            result["columns"] = cols

            # 3) Chart spec — load Phi-3, generate, unload
            logger.info("Step 3/4: chart reasoning")
            self.chart_reasoner.load()
            spec = self.chart_reasoner.generate(
                question=question, sql=sql, results=rows, columns=cols,
            )
            self.chart_reasoner.unload()
            result["chart_spec"] = spec

            # 4) Render — load DeepSeek (or Plotly fallback), render, unload
            logger.info("Step 4/4: SVG rendering")
            self.svg_renderer.load()
            svg = self.svg_renderer.generate(spec, rows)
            self.svg_renderer.unload()
            result["svg"] = svg

            return result

        except Exception as e:
            logger.exception("Pipeline failed")
            result["error"] = str(e)
            # Best-effort cleanup so a failure doesn't leak a model in VRAM
            for m in (self.sql_generator, self.chart_reasoner, self.svg_renderer):
                try:
                    if m.is_loaded:
                        m.unload()
                except Exception:
                    pass
            return result

    def reset(self) -> None:
        """Drop all data tables (keeps the connection alive)."""
        self.executor.close()
        self.executor = SQLExecutor()
        self.rag.bind(self.executor.con)

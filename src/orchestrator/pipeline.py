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
    def ensure_models_loaded(self) -> None:
        if not self.sql_generator.is_loaded:
            self.sql_generator.load()
        if not self.chart_reasoner.is_loaded:
            self.chart_reasoner.load()
        if not self.svg_renderer.is_loaded:
            self.svg_renderer.load()

    def process(self, question: str) -> Dict[str, Any]:
        """Run the full pipeline for one question."""
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

            self.ensure_models_loaded()

            # 1) SQL
            sql = self.sql_generator.generate(question=question, schema=schema)
            result["sql"] = sql
            if not self.executor.validate_query(sql):
                result["error"] = f"Generated SQL is invalid:\n{sql}"
                return result

            # 2) Execute
            rows, cols = self.executor.execute(sql)
            result["results"] = rows
            result["columns"] = cols

            # 3) Chart spec
            spec = self.chart_reasoner.generate(
                question=question, sql=sql, results=rows, columns=cols,
            )
            result["chart_spec"] = spec

            # 4) Render
            svg = self.svg_renderer.generate(spec, rows)
            result["svg"] = svg

            return result

        except Exception as e:
            logger.exception("Pipeline failed")
            result["error"] = str(e)
            return result

    def reset(self) -> None:
        """Drop all data tables (keeps the connection alive)."""
        self.executor.close()
        self.executor = SQLExecutor()
        self.rag.bind(self.executor.con)

"""
SQL Agent orchestrator. Models are constructed (loaded onto cuda) at
import time per ZeroGPU best practices. The pipeline runs inference only.
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

    def __init__(
        self,
        sql_generator: SQLGenerator,
        chart_reasoner: ChartReasoner,
        svg_renderer: SVGRenderer,
    ) -> None:
        self.executor = SQLExecutor()
        self.rag = RAGEngine(self.executor.con)
        self.sql_generator = sql_generator
        self.chart_reasoner = chart_reasoner
        self.svg_renderer = svg_renderer

    def load_data(
        self,
        source: Union[str, Path, pd.DataFrame],
        table_name: Optional[str] = None,
    ) -> str:
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

    def process(self, question: str) -> Dict[str, Any]:
        """Inference-only pipeline; models already loaded at module level."""
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

            sql = self.sql_generator.generate(question=question, schema=schema)
            result["sql"] = sql

            if not self.executor.validate_query(sql):
                result["error"] = f"Generated SQL is invalid:\n{sql}"
                return result

            rows, cols = self.executor.execute(sql)
            result["results"] = rows
            result["columns"] = cols

            spec = self.chart_reasoner.generate(
                question=question, sql=sql, results=rows, columns=cols,
            )
            result["chart_spec"] = spec

            svg = self.svg_renderer.generate(spec, rows)
            result["svg"] = svg

            return result

        except Exception as e:
            logger.exception("Pipeline failed")
            result["error"] = str(e)
            return result

    def reset(self) -> None:
        self.executor.close()
        self.executor = SQLExecutor()
        self.rag.bind(self.executor.con)

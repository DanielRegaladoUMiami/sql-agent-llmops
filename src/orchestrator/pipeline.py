"""
Main orchestrator pipeline that coordinates all models and components.

Manages the flow from natural language question to SQL execution
to visualization generation.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.models.sql_generator import SQLGenerator
from src.models.chart_reasoner import ChartReasoner
from src.models.svg_renderer import SVGRenderer
from src.rag.engine import RAGEngine
from src.utils.sql_executor import SQLExecutor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SQLAgentOrchestrator:
    """
    Main orchestrator for the SQL Agent system.

    Coordinates SQL generation, execution, chart reasoning, and visualization.
    """

    def __init__(
        self,
        db_path: str | Path,
        sql_model_path: Optional[str] = None,
        use_gguf: bool = False,
        sql_model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
    ) -> None:
        """
        Initialize orchestrator.

        Args:
            db_path: Path to SQLite database
            sql_model_path: Path to SQL model (GGUF or local)
            use_gguf: Whether to use GGUF format
            sql_model_name: HF model name if not using GGUF
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        logger.info(f"Initializing orchestrator with database: {db_path}")

        # Initialize components
        self.executor = SQLExecutor(self.db_path)
        self.rag_engine = RAGEngine()

        # Initialize models
        self.sql_generator = SQLGenerator(
            model_path=sql_model_path,
            use_gguf=use_gguf,
            hf_model=sql_model_name,
        )
        self.chart_reasoner = ChartReasoner()
        self.svg_renderer = SVGRenderer()

        # Index database
        try:
            self.rag_engine.index_database(self.db_path)
            logger.info("RAG engine indexed database")
        except Exception as e:
            logger.warning(f"RAG indexing failed: {e}")

    def load_models(self) -> None:
        """Load all models into memory."""
        logger.info("Loading models...")

        try:
            self.sql_generator.load()
            logger.info("SQL Generator loaded")
        except Exception as e:
            logger.error(f"Failed to load SQL Generator: {e}")
            raise

        try:
            self.chart_reasoner.load()
            logger.info("Chart Reasoner loaded")
        except Exception as e:
            logger.warning(f"Chart Reasoner load optional: {e}")

        try:
            self.svg_renderer.load()
            logger.info("SVG Renderer loaded")
        except Exception as e:
            logger.warning(f"SVG Renderer load optional: {e}")

    def unload_models(self) -> None:
        """Unload all models to free memory."""
        logger.info("Unloading models...")
        self.sql_generator.unload()
        self.chart_reasoner.unload()
        self.svg_renderer.unload()

    async def process(self, question: str) -> Dict[str, Any]:
        """
        Process a natural language question end-to-end.

        Args:
            question: Natural language question about the data

        Returns:
            Dictionary with:
            - question: Original question
            - sql: Generated SQL query
            - results: Query results (list of dicts)
            - columns: Column metadata
            - chart_config: Recommended chart configuration
            - visualization: SVG visualization
            - errors: Any errors encountered
        """
        logger.info(f"Processing question: {question[:100]}...")

        result = {
            "question": question,
            "sql": None,
            "results": [],
            "columns": [],
            "chart_config": None,
            "visualization": None,
            "errors": [],
        }

        try:
            # Step 1: Retrieve relevant schema
            logger.info("Step 1: Retrieving schema via RAG")
            schema = self.rag_engine.retrieve(question, top_k=5)
            result["schema"] = schema

            # Step 2: Generate SQL
            logger.info("Step 2: Generating SQL")
            sql = self._generate_sql(question, schema)
            if not sql:
                result["errors"].append("Failed to generate SQL")
                return result

            result["sql"] = sql

            # Step 3: Execute SQL
            logger.info("Step 3: Executing SQL")
            results, columns = self.executor.execute(sql)
            result["results"] = results
            result["columns"] = columns

            logger.info(f"Query returned {len(results)} rows")

            # Step 4: Reason about chart
            logger.info("Step 4: Reasoning about visualization")
            chart_config = self._reason_chart(question, sql, results, columns)
            result["chart_config"] = chart_config

            # Step 5: Render visualization
            logger.info("Step 5: Rendering visualization")
            visualization = self._render_visualization(chart_config, results)
            result["visualization"] = visualization

            logger.info("Processing complete")
            return result

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            result["errors"].append(str(e))
            return result

    def _generate_sql(self, question: str, schema: str) -> Optional[str]:
        """Generate SQL from question and schema."""
        try:
            if not self.sql_generator.is_loaded:
                self.sql_generator.load()

            sql = self.sql_generator.generate(
                question=question,
                schema=schema,
            )

            # Validate SQL
            if not self.executor.validate_query(sql):
                logger.warning(f"Invalid SQL generated: {sql}")
                return None

            return sql

        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            return None

    def _reason_chart(
        self,
        question: str,
        sql: str,
        results: List[Dict[str, Any]],
        columns: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Reason about best visualization."""
        try:
            if not self.chart_reasoner.is_loaded:
                self.chart_reasoner.load()

            config = self.chart_reasoner.generate(
                question=question,
                sql=sql,
                results=results,
                columns=columns,
            )

            return config

        except Exception as e:
            logger.error(f"Chart reasoning error: {e}")
            return {"chart_type": "table", "title": "Query Results"}

    def _render_visualization(
        self,
        chart_config: Dict[str, Any],
        data: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Render SVG visualization."""
        try:
            if not self.svg_renderer.is_loaded:
                self.svg_renderer.load()

            svg = self.svg_renderer.generate(chart_config, data)
            return svg

        except Exception as e:
            logger.error(f"Visualization rendering error: {e}")
            return None

    def __enter__(self):
        """Context manager entry."""
        self.load_models()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_models()
        self.rag_engine.clear()


async def main():
    """Example usage."""
    # This would require a real database file
    db_path = "example.db"

    try:
        orchestrator = SQLAgentOrchestrator(db_path)
        orchestrator.load_models()

        result = await orchestrator.process("What are the top 10 items by revenue?")

        print(f"SQL: {result['sql']}")
        print(f"Results: {len(result['results'])} rows")
        print(f"Chart type: {result['chart_config']['chart_type']}")

        orchestrator.unload_models()

    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())

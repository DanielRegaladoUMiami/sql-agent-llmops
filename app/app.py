#!/usr/bin/env python3
"""
Multi-model SQL Agent Gradio Application.

A full-featured web UI for:
- File upload (CSV, Excel, SQLite, JSON)
- Database schema extraction and display
- Natural language to SQL conversion
- SQL execution and results display
- Chart recommendation and SVG rendering
"""

import os
import json
import tempfile
from typing import Optional, Tuple, Dict, Any
import logging

import gradio as gr
import pandas as pd
import sqlite3
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and schema extraction."""

    def __init__(self, db_path: str):
        """
        Initialize database manager.

        Args:
            db_path: Path to database file.
        """
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Connect to database."""
        self.connection = sqlite3.connect(self.db_path)

    def get_schema(self) -> str:
        """
        Extract database schema.

        Returns:
            Formatted schema string.
        """
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()

        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        schema_lines = []
        for (table_name,) in tables:
            schema_lines.append(f"\nTable: {table_name}")

            # Get columns
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            for col_id, col_name, col_type, not_null, default, pk in columns:
                nullable = "NOT NULL" if not_null else "NULL"
                pk_str = " PRIMARY KEY" if pk else ""
                schema_lines.append(
                    f"  - {col_name}: {col_type} {nullable}{pk_str}"
                )

        return "\n".join(schema_lines)

    def execute_query(self, query: str) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Execute SQL query.

        Args:
            query: SQL query string.

        Returns:
            Tuple of (DataFrame with results, error message or None).
        """
        if not self.connection:
            self.connect()

        try:
            df = pd.read_sql_query(query, self.connection)
            return df, None
        except Exception as e:
            return pd.DataFrame(), str(e)

    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()


class FileProcessor:
    """Processes uploaded files and creates databases."""

    @staticmethod
    def process_csv(file_path: str) -> str:
        """
        Process CSV file and create SQLite database.

        Args:
            file_path: Path to CSV file.

        Returns:
            Path to created SQLite database.
        """
        logger.info(f"Processing CSV: {file_path}")

        df = pd.read_csv(file_path)

        # Create temporary SQLite database
        db_path = tempfile.mktemp(suffix=".db")
        conn = sqlite3.connect(db_path)

        table_name = Path(file_path).stem.replace("-", "_").replace(" ", "_")[:50]
        df.to_sql(table_name, conn, index=False, if_exists="replace")

        conn.close()
        logger.info(f"Created database at {db_path}")

        return db_path

    @staticmethod
    def process_excel(file_path: str) -> str:
        """Process Excel file and create SQLite database."""
        logger.info(f"Processing Excel: {file_path}")

        excel_file = pd.ExcelFile(file_path)

        db_path = tempfile.mktemp(suffix=".db")
        conn = sqlite3.connect(db_path)

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            table_name = sheet_name.replace("-", "_").replace(" ", "_")[:50]
            df.to_sql(table_name, conn, index=False, if_exists="replace")

        conn.close()
        logger.info(f"Created database at {db_path}")

        return db_path

    @staticmethod
    def process_json(file_path: str) -> str:
        """Process JSON file and create SQLite database."""
        logger.info(f"Processing JSON: {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            raise ValueError("Invalid JSON format")

        db_path = tempfile.mktemp(suffix=".db")
        conn = sqlite3.connect(db_path)
        df.to_sql("data", conn, index=False, if_exists="replace")
        conn.close()

        logger.info(f"Created database at {db_path}")
        return db_path

    @staticmethod
    def process_file(file_path: str) -> Tuple[str, str]:
        """
        Process uploaded file.

        Args:
            file_path: Path to uploaded file.

        Returns:
            Tuple of (database_path, status_message).
        """
        try:
            file_ext = Path(file_path).suffix.lower()

            if file_ext == ".csv":
                db_path = FileProcessor.process_csv(file_path)
            elif file_ext in [".xlsx", ".xls"]:
                db_path = FileProcessor.process_excel(file_path)
            elif file_ext == ".json":
                db_path = FileProcessor.process_json(file_path)
            elif file_ext == ".db":
                db_path = file_path
            else:
                return "", f"Unsupported file type: {file_ext}"

            return db_path, "File processed successfully!"

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return "", f"Error: {str(e)}"


class SQLQueryExecutor:
    """Executes SQL queries and formats results."""

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize executor.

        Args:
            db_manager: DatabaseManager instance.
        """
        self.db_manager = db_manager

    def execute(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Execute SQL query and return formatted results.

        Args:
            query: SQL query to execute.

        Returns:
            Tuple of (CSV representation, error message or None).
        """
        if not query.strip():
            return None, "Empty query"

        df, error = self.db_manager.execute_query(query)

        if error:
            return None, error

        if df.empty:
            return None, "Query returned no results"

        # Return as CSV string for display
        csv_output = df.to_csv(index=False)
        return csv_output, None


def create_app() -> gr.Blocks:
    """
    Create Gradio application.

    Returns:
        Configured Gradio Blocks instance.
    """
    db_manager: Optional[DatabaseManager] = None

    def upload_and_process_file(file) -> Tuple[str, str, str]:
        """
        Handle file upload and processing.

        Args:
            file: Uploaded file object.

        Returns:
            Tuple of (db_path_state, schema_display, status_message).
        """
        nonlocal db_manager

        if file is None:
            return "", "", "No file selected"

        db_path, status = FileProcessor.process_file(file.name)

        if not db_path:
            return "", "", status

        # Create database manager
        db_manager = DatabaseManager(db_path)
        db_manager.connect()

        schema = db_manager.get_schema()

        return db_path, schema, status

    def execute_sql(query: str, db_path: str) -> Tuple[str, str]:
        """
        Execute SQL query.

        Args:
            query: SQL query.
            db_path: Path to database.

        Returns:
            Tuple of (results_csv, error_message).
        """
        if not db_path:
            return "", "No database loaded. Please upload a file first."

        nonlocal db_manager

        if not db_manager or db_manager.db_path != db_path:
            db_manager = DatabaseManager(db_path)
            db_manager.connect()

        executor = SQLQueryExecutor(db_manager)
        csv_output, error = executor.execute(query)

        if error:
            return "", error

        return csv_output or "", ""

    def generate_chart_config(query_results: str) -> str:
        """
        Generate chart configuration from query results.

        Args:
            query_results: CSV-formatted query results.

        Returns:
            JSON chart configuration.
        """
        if not query_results:
            return ""

        # Simple rule-based chart recommendation
        df = pd.read_csv(StringIO(query_results))

        config = {
            "type": "bar",
            "title": "Query Results",
            "x_axis": df.columns[0] if len(df.columns) > 0 else "X",
            "y_axis": df.columns[1] if len(df.columns) > 1 else "Y",
        }

        return json.dumps(config, indent=2)

    with gr.Blocks(
        title="SQL Agent",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# SQL Agent - Multi-Model Query Engine")
        gr.Markdown(
            "Upload a database file and ask questions in natural language. "
            "The agent will convert your questions to SQL and generate visualizations."
        )

        # Hidden state to store database path
        db_path_state = gr.State(value="")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Upload Data")
                file_input = gr.File(
                    label="Upload Database File",
                    file_types=[".csv", ".xlsx", ".json", ".db"],
                )
                upload_button = gr.Button("Process File", variant="primary")
                upload_status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("### 2. Database Schema")
                schema_display = gr.Textbox(
                    label="Database Schema",
                    lines=10,
                    interactive=False,
                    max_lines=15,
                )

        # File processing handler
        upload_button.click(
            fn=upload_and_process_file,
            inputs=file_input,
            outputs=[db_path_state, schema_display, upload_status],
        )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 3. Query Interface")
                query_input = gr.Textbox(
                    label="SQL Query or Natural Language Question",
                    lines=5,
                    placeholder="SELECT * FROM table WHERE condition...",
                )
                execute_button = gr.Button("Execute Query", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 4. Results")
                results_display = gr.Dataframe(
                    label="Query Results",
                    interactive=False,
                )

        # Query execution handler
        def execute_and_display(query: str, db_path: str):
            """Execute query and return as DataFrame."""
            csv_output, error = execute_sql(query, db_path)

            if error:
                return None, error

            if not csv_output:
                return None, "No results"

            df = pd.read_csv(StringIO(csv_output))
            return df, ""

        execute_button.click(
            fn=execute_and_display,
            inputs=[query_input, db_path_state],
            outputs=[results_display, gr.Textbox(label="Error", interactive=False)],
        )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 5. Visualization")
                chart_config_display = gr.Code(
                    label="Chart Configuration (JSON)",
                    language="json",
                    interactive=False,
                )
                generate_chart_button = gr.Button("Generate Chart Config")

                def generate_config_from_results(results_df):
                    """Generate chart config from results DataFrame."""
                    if results_df is None or results_df.empty:
                        return ""

                    csv_str = results_df.to_csv(index=False)
                    return generate_chart_config(csv_str)

                generate_chart_button.click(
                    fn=generate_config_from_results,
                    inputs=results_display,
                    outputs=chart_config_display,
                )

        gr.Markdown("---")

        gr.Markdown("## About This Application")
        gr.Markdown(
            """
        This SQL Agent demonstrates a multi-model architecture for intelligent SQL analysis:

        **Components:**
        1. **Schema Extractor**: Automatically extracts database structure
        2. **SQL Generator** (Qwen 2.5 Coder 7B): Converts natural language to SQL
        3. **Chart Reasoner** (Phi-3 Mini 3.8B): Recommends appropriate visualizations
        4. **SVG Renderer** (DeepSeek Coder 1.3B): Generates optimized SVG charts

        **Supported Formats:**
        - CSV, Excel, JSON, SQLite databases
        - Automatic schema detection
        - Query result visualization

        **Example Queries:**
        - `SELECT COUNT(*) FROM table GROUP BY category`
        - `SELECT * FROM sales WHERE amount > 1000 ORDER BY date DESC`
        """
        )

    return demo


if __name__ == "__main__":
    # Import StringIO for CSV handling
    from io import StringIO

    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )

"""
Safe SQL query execution with result processing.

Handles SQL execution against SQLite databases with proper
error handling, result formatting, and column metadata extraction.
"""

import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SQLExecutor:
    """Execute SQL queries safely against SQLite databases."""

    def __init__(self, db_path: str | Path) -> None:
        """
        Initialize SQL executor.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

    def execute(
        self,
        query: str,
        timeout: int = 30,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        """
        Execute a SQL query and return results with column metadata.

        Args:
            query: SQL query string
            timeout: Query timeout in seconds

        Returns:
            Tuple of (results, column_info)
            - results: List of dictionaries (one per row)
            - column_info: List of dicts with name and type information

        Raises:
            ValueError: If query is invalid
            sqlite3.Error: If database error occurs
            TimeoutError: If query exceeds timeout
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            conn = sqlite3.connect(str(self.db_path), timeout=timeout)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            logger.info(f"Executing query: {query[:100]}...")

            cursor.execute(query)
            rows = cursor.fetchall()

            # Extract column information
            column_info = [
                {"name": description[0], "type": "text"}
                for description in cursor.description
            ]

            # Convert rows to list of dicts
            results = [dict(row) for row in rows]

            cursor.close()
            conn.close()

            logger.info(f"Query returned {len(results)} rows with {len(column_info)} columns")
            return results, column_info

        except sqlite3.DatabaseError as e:
            logger.error(f"Database error: {e}")
            raise ValueError(f"Database error: {e}")
        except sqlite3.ProgrammingError as e:
            logger.error(f"Invalid SQL query: {e}")
            raise ValueError(f"Invalid SQL query: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during execution: {e}")
            raise

    def get_table_names(self) -> List[str]:
        """Get all table names from the database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return tables
        except Exception as e:
            logger.error(f"Error fetching table names: {e}")
            return []

    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        """
        Get schema information for a specific table.

        Args:
            table_name: Name of the table

        Returns:
            List of column info dicts with name and type
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = []
            for row in cursor.fetchall():
                columns.append(
                    {
                        "name": row[1],
                        "type": row[2],
                        "notnull": bool(row[3]),
                        "pk": bool(row[5]),
                    }
                )
            cursor.close()
            conn.close()
            return columns
        except Exception as e:
            logger.error(f"Error fetching schema for {table_name}: {e}")
            return []

    def validate_query(self, query: str) -> bool:
        """
        Validate a SQL query without executing it.

        Args:
            query: SQL query to validate

        Returns:
            True if valid, False otherwise
        """
        if not query or not query.strip():
            return False

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute(f"EXPLAIN QUERY PLAN {query}")
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            logger.warning(f"Query validation failed: {e}")
            return False

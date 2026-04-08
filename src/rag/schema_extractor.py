"""
Extract database schema information from SQLite databases.

Retrieves table names, columns, types, constraints, foreign keys,
and sample data for RAG indexing.
"""

import sqlite3
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SchemaExtractor:
    """Extract comprehensive schema information from SQLite databases."""

    def __init__(self, db_path: str | Path) -> None:
        """
        Initialize schema extractor.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

    def extract_full_schema(self) -> Dict[str, Any]:
        """
        Extract complete schema from database.

        Returns:
            Dict with keys:
            - tables: List of table schemas
            - relationships: Foreign key relationships
            - summary: Overall database summary
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            tables = self._extract_tables(cursor)
            relationships = self._extract_relationships(cursor)

            conn.close()

            schema = {
                "tables": tables,
                "relationships": relationships,
                "summary": self._generate_summary(tables),
            }

            logger.info(f"Extracted schema with {len(tables)} tables")
            return schema

        except Exception as e:
            logger.error(f"Error extracting schema: {e}")
            raise

    def _extract_tables(self, cursor: sqlite3.Cursor) -> List[Dict[str, Any]]:
        """Extract all user tables with column details."""
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        table_names = [row[0] for row in cursor.fetchall()]

        tables = []
        for table_name in table_names:
            table_schema = {
                "name": table_name,
                "columns": self._extract_columns(cursor, table_name),
                "primary_key": self._extract_primary_key(cursor, table_name),
                "row_count": self._get_row_count(cursor, table_name),
            }
            tables.append(table_schema)

        return tables

    def _extract_columns(self, cursor: sqlite3.Cursor, table_name: str) -> List[Dict[str, Any]]:
        """Extract column information."""
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = []

        for row in cursor.fetchall():
            col_idx, col_name, col_type, notnull, default_val, pk = row
            columns.append(
                {
                    "name": col_name,
                    "type": col_type,
                    "nullable": not bool(notnull),
                    "default": default_val,
                    "primary_key": bool(pk),
                }
            )

        return columns

    def _extract_primary_key(self, cursor: sqlite3.Cursor, table_name: str) -> Optional[List[str]]:
        """Extract primary key columns."""
        cursor.execute(f"PRAGMA table_info({table_name})")
        pk_cols = [row[1] for row in cursor.fetchall() if row[5]]
        return pk_cols if pk_cols else None

    def _get_row_count(self, cursor: sqlite3.Cursor, table_name: str) -> int:
        """Get row count for table."""
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cursor.fetchone()[0]
        except Exception:
            return 0

    def _extract_relationships(self, cursor: sqlite3.Cursor) -> List[Dict[str, str]]:
        """Extract foreign key relationships."""
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        table_names = [row[0] for row in cursor.fetchall()]

        relationships = []
        for table_name in table_names:
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            for row in cursor.fetchall():
                rel = {
                    "from_table": table_name,
                    "from_column": row[3],
                    "to_table": row[2],
                    "to_column": row[4],
                }
                relationships.append(rel)

        return relationships

    def _generate_summary(self, tables: List[Dict[str, Any]]) -> str:
        """Generate human-readable schema summary."""
        lines = ["Database Schema Summary:", "=" * 50]

        for table in tables:
            lines.append(f"\nTable: {table['name']} ({table['row_count']} rows)")
            lines.append("  Columns:")
            for col in table["columns"]:
                nullable = "NULL" if col["nullable"] else "NOT NULL"
                pk = " [PK]" if col["primary_key"] else ""
                lines.append(f"    - {col['name']}: {col['type']} {nullable}{pk}")

        return "\n".join(lines)

    def get_table_schema_text(self, table_name: str) -> str:
        """
        Get human-readable schema for a specific table.

        Args:
            table_name: Name of table

        Returns:
            Formatted schema string
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            columns = self._extract_columns(cursor, table_name)
            row_count = self._get_row_count(cursor, table_name)

            conn.close()

            lines = [f"Table: {table_name} ({row_count} rows)"]
            for col in columns:
                nullable = "NULL" if col["nullable"] else "NOT NULL"
                pk = " [PRIMARY KEY]" if col["primary_key"] else ""
                lines.append(f"  {col['name']}: {col['type']} {nullable}{pk}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error getting table schema: {e}")
            return ""

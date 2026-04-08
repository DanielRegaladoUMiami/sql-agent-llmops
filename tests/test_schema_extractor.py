"""
Tests for database schema extraction functionality.
"""

import pytest
import sqlite3
from pathlib import Path


class TestSchemaExtractor:
    """Test cases for schema extraction."""

    def test_extract_table_names(self, sample_db: str) -> None:
        """Test extracting table names from database."""
        conn = sqlite3.connect(sample_db)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        assert len(tables) == 2
        table_names = [t[0] for t in tables]
        assert 'users' in table_names
        assert 'orders' in table_names

        conn.close()

    def test_extract_column_info(self, sample_db: str) -> None:
        """Test extracting column information from table."""
        conn = sqlite3.connect(sample_db)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()

        assert len(columns) == 4
        column_names = [c[1] for c in columns]
        assert 'id' in column_names
        assert 'name' in column_names
        assert 'email' in column_names
        assert 'created_at' in column_names

        conn.close()

    def test_extract_primary_keys(self, sample_db: str) -> None:
        """Test identifying primary keys."""
        conn = sqlite3.connect(sample_db)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()

        # Find primary key column (pk field is non-zero)
        pk_columns = [c[1] for c in columns if c[5]]
        assert 'id' in pk_columns

        conn.close()

    def test_extract_foreign_keys(self, sample_db: str) -> None:
        """Test identifying foreign key relationships."""
        conn = sqlite3.connect(sample_db)
        cursor = conn.cursor()

        cursor.execute("PRAGMA foreign_key_list(orders)")
        fks = cursor.fetchall()

        assert len(fks) > 0
        # Foreign key points to users table
        assert any(fk[2] == 'users' for fk in fks)

        conn.close()

    def test_schema_format_output(self, sample_db: str) -> None:
        """Test formatted schema output."""
        conn = sqlite3.connect(sample_db)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        schema_lines = []
        for (table_name,) in tables:
            schema_lines.append(f"Table: {table_name}")

            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            for col_id, col_name, col_type, not_null, default, pk in columns:
                nullable = "NOT NULL" if not_null else "NULL"
                pk_str = " PRIMARY KEY" if pk else ""
                schema_lines.append(
                    f"  - {col_name}: {col_type} {nullable}{pk_str}"
                )

        schema_text = "\n".join(schema_lines)

        assert "Table: users" in schema_text
        assert "Table: orders" in schema_text
        assert "id: INTEGER" in schema_text
        assert "PRIMARY KEY" in schema_text

        conn.close()

    def test_empty_database_schema(self, tmp_path) -> None:
        """Test schema extraction from empty database."""
        db_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_path)
        conn.close()

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        assert len(tables) == 0

        conn.close()

    def test_schema_consistency(self, sample_db: str) -> None:
        """Test schema remains consistent after queries."""
        conn = sqlite3.connect(sample_db)

        # Get initial schema
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        initial_tables = cursor.fetchall()

        # Run a query
        cursor.execute("SELECT COUNT(*) FROM users")
        result = cursor.fetchone()

        # Verify schema unchanged
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        final_tables = cursor.fetchall()

        assert initial_tables == final_tables
        assert result[0] == 5

        conn.close()

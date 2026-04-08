"""
Tests for SQL query execution functionality.
"""

import pytest
import sqlite3
import pandas as pd
from typing import Tuple, Optional


class SQLExecutor:
    """Simple SQL executor for testing."""

    def __init__(self, db_path: str):
        """Initialize executor with database path."""
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)

    def execute(self, query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Execute SQL query and return DataFrame."""
        try:
            df = pd.read_sql_query(query, self.connection)
            return df, None
        except Exception as e:
            return None, str(e)

    def close(self):
        """Close database connection."""
        self.connection.close()


class TestSQLExecutor:
    """Test cases for SQL execution."""

    def test_simple_select(self, sample_db: str) -> None:
        """Test simple SELECT query."""
        executor = SQLExecutor(sample_db)

        df, error = executor.execute("SELECT * FROM users LIMIT 3")

        assert error is None
        assert df is not None
        assert len(df) == 3
        assert 'name' in df.columns
        assert 'email' in df.columns

        executor.close()

    def test_where_clause(self, sample_db: str) -> None:
        """Test SELECT with WHERE clause."""
        executor = SQLExecutor(sample_db)

        df, error = executor.execute(
            'SELECT * FROM orders WHERE status = "completed"'
        )

        assert error is None
        assert df is not None
        assert len(df) == 4
        assert all(df['status'] == 'completed')

        executor.close()

    def test_aggregation(self, sample_db: str) -> None:
        """Test aggregation functions."""
        executor = SQLExecutor(sample_db)

        df, error = executor.execute(
            'SELECT COUNT(*) as count, SUM(amount) as total FROM orders'
        )

        assert error is None
        assert df is not None
        assert len(df) == 1
        assert df['count'].iloc[0] == 7
        assert df['total'].iloc[0] == pytest.approx(1801.25, rel=0.01)

        executor.close()

    def test_group_by(self, sample_db: str) -> None:
        """Test GROUP BY clause."""
        executor = SQLExecutor(sample_db)

        df, error = executor.execute(
            'SELECT user_id, COUNT(*) as order_count FROM orders GROUP BY user_id'
        )

        assert error is None
        assert df is not None
        assert len(df) == 5
        assert 'order_count' in df.columns

        executor.close()

    def test_join_query(self, sample_db: str) -> None:
        """Test JOIN queries."""
        executor = SQLExecutor(sample_db)

        df, error = executor.execute(
            '''SELECT u.name, COUNT(o.id) as orders
               FROM users u
               LEFT JOIN orders o ON u.id = o.user_id
               GROUP BY u.id'''
        )

        assert error is None
        assert df is not None
        assert 'name' in df.columns
        assert 'orders' in df.columns

        executor.close()

    def test_order_by(self, sample_db: str) -> None:
        """Test ORDER BY clause."""
        executor = SQLExecutor(sample_db)

        df, error = executor.execute(
            'SELECT * FROM orders ORDER BY amount DESC LIMIT 3'
        )

        assert error is None
        assert df is not None
        assert len(df) == 3
        assert df['amount'].iloc[0] >= df['amount'].iloc[1]

        executor.close()

    def test_invalid_query(self, sample_db: str) -> None:
        """Test handling of invalid SQL."""
        executor = SQLExecutor(sample_db)

        df, error = executor.execute("SELECT * FROM nonexistent_table")

        assert error is not None
        assert df is None

        executor.close()

    def test_syntax_error(self, sample_db: str) -> None:
        """Test handling of syntax errors."""
        executor = SQLExecutor(sample_db)

        df, error = executor.execute("SELECT * FORM users")  # FORM instead of FROM

        assert error is not None

        executor.close()

    def test_empty_result_set(self, sample_db: str) -> None:
        """Test query returning no results."""
        executor = SQLExecutor(sample_db)

        df, error = executor.execute(
            "SELECT * FROM orders WHERE status = 'nonexistent'"
        )

        assert error is None
        assert df is not None
        assert len(df) == 0

        executor.close()

    def test_distinct(self, sample_db: str) -> None:
        """Test DISTINCT keyword."""
        executor = SQLExecutor(sample_db)

        df, error = executor.execute(
            'SELECT DISTINCT status FROM orders'
        )

        assert error is None
        assert df is not None
        assert len(df) == 2  # 'completed' and 'pending'

        executor.close()

    def test_limit_offset(self, sample_db: str) -> None:
        """Test LIMIT and OFFSET."""
        executor = SQLExecutor(sample_db)

        df, error = executor.execute(
            'SELECT * FROM users LIMIT 2 OFFSET 1'
        )

        assert error is None
        assert df is not None
        assert len(df) == 2

        executor.close()

    def test_date_filtering(self, sample_db: str) -> None:
        """Test date filtering."""
        executor = SQLExecutor(sample_db)

        df, error = executor.execute(
            'SELECT * FROM users WHERE created_at > "2023-02-01"'
        )

        assert error is None
        assert df is not None
        assert len(df) == 4

        executor.close()

    def test_numeric_operations(self, sample_db: str) -> None:
        """Test numeric calculations."""
        executor = SQLExecutor(sample_db)

        df, error = executor.execute(
            'SELECT user_id, amount * 1.1 as increased_amount FROM orders LIMIT 1'
        )

        assert error is None
        assert df is not None
        assert 'increased_amount' in df.columns

        executor.close()

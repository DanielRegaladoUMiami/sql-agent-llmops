"""
Profile database data for semantic understanding and visualization hints.

Analyzes column statistics, detects temporal columns, identifies
categorical vs numeric data, and other data characteristics.
"""

import sqlite3
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataProfiler:
    """Profile data in SQLite database for analysis."""

    def __init__(self, db_path: str | Path) -> None:
        """
        Initialize data profiler.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

    def profile_table(self, table_name: str, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Profile a single table.

        Args:
            table_name: Table to profile
            sample_size: Number of rows to sample

        Returns:
            Dict with column profiles and statistics
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(f"SELECT * FROM {table_name} LIMIT {sample_size}")
            rows = cursor.fetchall()

            if not rows:
                cursor.close()
                conn.close()
                return {}

            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]

            profile = {
                "table_name": table_name,
                "total_rows": self._get_row_count(cursor, table_name),
                "columns": {},
            }

            for i, col_name in enumerate(columns):
                col_values = [row[i] for row in rows if row[i] is not None]
                profile["columns"][col_name] = self._profile_column(col_name, col_values)

            cursor.close()
            conn.close()

            return profile

        except Exception as e:
            logger.error(f"Error profiling table {table_name}: {e}")
            return {}

    def _profile_column(self, col_name: str, values: List[Any]) -> Dict[str, Any]:
        """Profile a single column."""
        if not values:
            return {"type": "unknown", "sample_values": []}

        profile = {
            "non_null_count": len(values),
            "unique_count": len(set(values)),
            "sample_values": values[:5],
        }

        # Detect data type and characteristics
        is_numeric = self._is_numeric(values)
        is_temporal = self._is_temporal(values)
        is_categorical = self._is_categorical(values, len(set(values)))

        if is_temporal:
            profile["type"] = "temporal"
            profile["characteristics"] = ["temporal", "date-like"]
        elif is_numeric:
            profile["type"] = "numeric"
            profile["min"] = min(float(v) for v in values)
            profile["max"] = max(float(v) for v in values)
            profile["mean"] = sum(float(v) for v in values) / len(values)
            profile["characteristics"] = ["numeric", "quantitative"]
        elif is_categorical:
            profile["type"] = "categorical"
            profile["characteristics"] = ["categorical", "discrete"]
        else:
            profile["type"] = "text"
            profile["characteristics"] = ["text", "string"]

        return profile

    def _is_numeric(self, values: List[Any]) -> bool:
        """Check if values are numeric."""
        if not values:
            return False

        numeric_count = 0
        for v in values[:min(10, len(values))]:
            try:
                float(v)
                numeric_count += 1
            except (ValueError, TypeError):
                pass

        return numeric_count / len(values[:10]) > 0.8

    def _is_temporal(self, values: List[Any]) -> bool:
        """Check if values are temporal/date."""
        if not values:
            return False

        temporal_patterns = [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%Y/%m/%d",
        ]

        temporal_count = 0
        for v in values[:min(10, len(values))]:
            if not v:
                continue
            for pattern in temporal_patterns:
                try:
                    datetime.strptime(str(v), pattern)
                    temporal_count += 1
                    break
                except ValueError:
                    pass

        return temporal_count > 0

    def _is_categorical(self, values: List[Any], unique_count: int) -> bool:
        """Check if column is categorical."""
        if not values:
            return False

        cardinality = unique_count / len(values)
        # Categorical if low cardinality (less than 10% unique values)
        return cardinality < 0.1

    def _get_row_count(self, cursor: sqlite3.Cursor, table_name: str) -> int:
        """Get total row count."""
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cursor.fetchone()[0]
        except Exception:
            return 0

    def get_column_recommendations(
        self,
        table_name: str,
    ) -> Dict[str, List[str]]:
        """
        Get visualization recommendations based on column characteristics.

        Args:
            table_name: Table to analyze

        Returns:
            Dict mapping column names to recommended chart types
        """
        profile = self.profile_table(table_name)
        recommendations = {}

        for col_name, col_profile in profile.get("columns", {}).items():
            col_type = col_profile.get("type", "text")
            characteristics = col_profile.get("characteristics", [])

            if col_type == "temporal":
                recommendations[col_name] = ["line", "bar", "area"]
            elif col_type == "numeric":
                recommendations[col_name] = ["histogram", "box", "scatter"]
            elif col_type == "categorical":
                recommendations[col_name] = ["bar", "pie"]
            else:
                recommendations[col_name] = ["bar", "table"]

        return recommendations

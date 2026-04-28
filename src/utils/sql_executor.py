"""
DuckDB-based SQL executor for in-memory analytical queries.

Accepts pandas DataFrames or CSV/JSON paths and exposes them as
queryable tables in a single in-memory connection.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class SQLExecutor:
    """Execute SQL queries against an in-memory DuckDB connection."""

    def __init__(self) -> None:
        self.con = duckdb.connect(database=":memory:")
        self._tables: Dict[str, int] = {}

    def register_dataframe(self, name: str, df: pd.DataFrame) -> None:
        """Register a DataFrame as a queryable table."""
        safe = self._sanitize_name(name)
        self.con.register(f"_tmp_{safe}", df)
        self.con.execute(f'CREATE OR REPLACE TABLE "{safe}" AS SELECT * FROM _tmp_{safe}')
        self.con.unregister(f"_tmp_{safe}")
        self._tables[safe] = len(df)
        logger.info(f"Registered table '{safe}' ({len(df):,} rows, {len(df.columns)} cols)")

    def register_file(self, path: Union[str, Path], name: Optional[str] = None) -> str:
        """Load a CSV/JSON/Parquet file into a table. Returns the table name used."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        safe = self._sanitize_name(name or path.stem)
        ext = path.suffix.lower()

        if ext == ".csv":
            self.con.execute(
                f"CREATE OR REPLACE TABLE \"{safe}\" AS SELECT * FROM read_csv_auto('{path}')"
            )
        elif ext == ".json":
            self.con.execute(
                f"CREATE OR REPLACE TABLE \"{safe}\" AS SELECT * FROM read_json_auto('{path}')"
            )
        elif ext in (".parquet", ".pq"):
            self.con.execute(
                f"CREATE OR REPLACE TABLE \"{safe}\" AS SELECT * FROM read_parquet('{path}')"
            )
        elif ext in (".xls", ".xlsx"):
            df = pd.read_excel(path)
            self.register_dataframe(safe, df)
            return safe
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        rows = self.con.execute(f'SELECT COUNT(*) FROM "{safe}"').fetchone()[0]
        self._tables[safe] = rows
        logger.info(f"Loaded '{path.name}' as table '{safe}' ({rows:,} rows)")
        return safe

    def execute(self, query: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        """Execute a query and return (rows, column_info)."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query = query.strip().rstrip(";")
        logger.info(f"Executing: {query[:120]}...")

        try:
            cur = self.con.execute(query)
            rows = cur.fetchall()
            descriptions = cur.description or []
            columns = [
                {"name": d[0], "type": str(d[1]) if d[1] else "VARCHAR"}
                for d in descriptions
            ]
            results = [dict(zip([c["name"] for c in columns], row)) for row in rows]
            logger.info(f"Returned {len(results):,} rows × {len(columns)} cols")
            return results, columns
        except duckdb.Error as e:
            logger.error(f"DuckDB error: {e}")
            raise ValueError(f"SQL error: {e}")

    def validate_query(self, query: str) -> bool:
        """Check that a query parses and references valid tables, without executing."""
        if not query or not query.strip():
            return False
        try:
            self.con.execute(f"EXPLAIN {query.strip().rstrip(';')}")
            return True
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return False

    def get_table_names(self) -> List[str]:
        rows = self.con.execute("SHOW TABLES").fetchall()
        return [r[0] for r in rows]

    def get_table_schema(self, table: str) -> List[Dict[str, Any]]:
        safe = self._sanitize_name(table)
        rows = self.con.execute(f'DESCRIBE "{safe}"').fetchall()
        return [
            {"name": r[0], "type": r[1], "nullable": r[2] != "NO" if r[2] else True}
            for r in rows
        ]

    def get_sample(self, table: str, n: int = 5) -> pd.DataFrame:
        safe = self._sanitize_name(table)
        return self.con.execute(f'SELECT * FROM "{safe}" LIMIT {n}').df()

    def close(self) -> None:
        self.con.close()

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Make a string safe to use as an unquoted table identifier fallback."""
        s = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        if s and s[0].isdigit():
            s = "t_" + s
        return s or "table"

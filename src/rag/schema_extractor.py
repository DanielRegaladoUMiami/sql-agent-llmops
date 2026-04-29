"""
Schema extractor for an in-memory DuckDB connection.

Reads tables/columns/types via DuckDB introspection and produces both a
structured dict and a CREATE-TABLE-style text rendering used as context
for the SQL generator model.
"""

import logging
from typing import Any, Dict, List

import duckdb

logger = logging.getLogger(__name__)


class SchemaExtractor:
    """Extract schema information from a DuckDB connection."""

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self.con = con

    def extract_full_schema(self) -> Dict[str, Any]:
        tables = self._extract_tables()
        return {
            "tables": tables,
            "summary": self._generate_summary(tables),
            "create_statements": self._build_create_statements(tables),
        }

    def _extract_tables(self) -> List[Dict[str, Any]]:
        names = [r[0] for r in self.con.execute("SHOW TABLES").fetchall()]
        out: List[Dict[str, Any]] = []
        for name in names:
            cols = self._extract_columns(name)
            row_count = self._get_row_count(name)
            sample = self._get_sample(name)
            out.append({
                "name": name,
                "columns": cols,
                "row_count": row_count,
                "sample": sample,
            })
        return out

    def _extract_columns(self, table: str) -> List[Dict[str, Any]]:
        rows = self.con.execute(f'DESCRIBE "{table}"').fetchall()
        cols: List[Dict[str, Any]] = []
        for r in rows:
            name, dtype = r[0], r[1]
            nullable = (r[2] != "NO") if len(r) > 2 and r[2] is not None else True
            cols.append({
                "name": name,
                "type": dtype,
                "nullable": nullable,
            })
        return cols

    def _get_row_count(self, table: str) -> int:
        try:
            return self.con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        except Exception:
            return 0

    def _get_sample(self, table: str, n: int = 3) -> List[Dict[str, Any]]:
        try:
            cur = self.con.execute(f'SELECT * FROM "{table}" LIMIT {n}')
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description or []]
            return [dict(zip(cols, row)) for row in rows]
        except Exception:
            return []

    def _generate_summary(self, tables: List[Dict[str, Any]]) -> str:
        lines = []
        for t in tables:
            lines.append(f"Table {t['name']} ({t['row_count']:,} rows):")
            for c in t["columns"]:
                lines.append(f"  - {c['name']}: {c['type']}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _build_create_statements(self, tables: List[Dict[str, Any]]) -> str:
        """Render schema as CREATE TABLE statements (best format for LLM context)."""
        chunks = []
        for t in tables:
            cols = ",\n  ".join(
                f'"{c["name"]}" {c["type"]}' for c in t["columns"]
            )
            chunks.append(f'CREATE TABLE "{t["name"]}" (\n  {cols}\n);')
        return "\n\n".join(chunks)

    def get_schema_text(self) -> str:
        """Schema description optimized for LLM SQL generation:
        - CREATE TABLE statements (canonical column names + types)
        - Sample rows (so model sees realistic values)
        - For low-cardinality categorical columns: list of distinct values
          (gives the model a vocabulary to match user wording against)
        """
        info = self.extract_full_schema()
        out = [info["create_statements"]]

        for t in info["tables"]:
            if t["sample"]:
                out.append(f"\n-- Sample rows from {t['name']}:")
                for row in t["sample"]:
                    out.append(f"-- {row}")

            # Distinct-values hints for categorical columns
            for col in t["columns"]:
                hint = self._distinct_values_hint(t["name"], col)
                if hint:
                    out.append(hint)

        return "\n".join(out)

    def _distinct_values_hint(
        self, table: str, col: Dict[str, Any], threshold: int = 25
    ) -> str | None:
        """If a column is categorical with <=threshold distinct values,
        list them so the LLM can map user wording to actual values."""
        dtype = (col.get("type") or "").upper()
        # Only worth doing for string-ish columns
        if not any(t in dtype for t in ("VARCHAR", "STRING", "TEXT", "CHAR")):
            return None
        try:
            rows = self.con.execute(
                f'SELECT DISTINCT "{col["name"]}" FROM "{table}" '
                f'WHERE "{col["name"]}" IS NOT NULL '
                f'LIMIT {threshold + 1}'
            ).fetchall()
        except Exception:
            return None
        if len(rows) > threshold:
            return None  # too many distinct values, not categorical
        if len(rows) <= 1:
            return None  # not informative
        values = ", ".join(repr(r[0]) for r in rows)
        return f"-- {table}.{col['name']} distinct values: {values}"

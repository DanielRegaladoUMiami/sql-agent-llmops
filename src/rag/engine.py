"""
Schema retrieval engine for the SQL Agent.

For typical user uploads (1-3 tables, a few dozen columns) we can pass the
full schema to the model. ChromaDB-based semantic retrieval is overkill at
this scale; we keep the API simple and let SchemaExtractor produce
CREATE-TABLE-style context.

If the schema ever exceeds the model's context budget, this is the place
to add embedding-based filtering.
"""

import logging

import duckdb

from src.rag.schema_extractor import SchemaExtractor

logger = logging.getLogger(__name__)


class RAGEngine:
    """Light wrapper that returns the full schema as model context."""

    def __init__(self, con: duckdb.DuckDBPyConnection | None = None) -> None:
        self.con = con
        self.extractor: SchemaExtractor | None = (
            SchemaExtractor(con) if con is not None else None
        )

    def bind(self, con: duckdb.DuckDBPyConnection) -> None:
        self.con = con
        self.extractor = SchemaExtractor(con)

    def retrieve(self, question: str, top_k: int = 5) -> str:
        if not self.extractor:
            logger.warning("RAG engine called before bind()")
            return ""
        return self.extractor.get_schema_text()

    def clear(self) -> None:
        self.extractor = None
        self.con = None

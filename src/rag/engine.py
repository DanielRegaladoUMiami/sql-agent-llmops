"""
RAG (Retrieval-Augmented Generation) engine for semantic schema retrieval.

Uses ChromaDB with sentence-transformers embeddings to index and retrieve
relevant database schema information based on natural language queries.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.rag.schema_extractor import SchemaExtractor
from src.rag.data_profiler import DataProfiler

logger = logging.getLogger(__name__)


class RAGEngine:
    """Retrieve relevant schema using semantic search."""

    def __init__(self) -> None:
        """Initialize RAG engine."""
        self.collection = None
        self.chroma_client = None
        self.is_initialized = False
        self._init_chroma()

    def _init_chroma(self) -> None:
        """Initialize ChromaDB client."""
        try:
            import chromadb

            self.chroma_client = chromadb.EphemeralClient()
            self.is_initialized = True
            logger.info("ChromaDB initialized (in-memory)")
        except ImportError:
            logger.warning(
                "chromadb not available. Install with: pip install chromadb. "
                "Using fallback schema matching."
            )
            self.is_initialized = False

    def index_database(self, db_path: str | Path) -> None:
        """
        Index database schema and create embeddings.

        Args:
            db_path: Path to SQLite database
        """
        try:
            db_path = Path(db_path)
            if not db_path.exists():
                raise FileNotFoundError(f"Database not found: {db_path}")

            logger.info(f"Indexing database: {db_path}")

            extractor = SchemaExtractor(db_path)
            profiler = DataProfiler(db_path)
            schema = extractor.extract_full_schema()

            # Create documents for indexing
            documents = []
            metadatas = []
            ids = []

            for table in schema.get("tables", []):
                table_name = table["name"]

                # Table-level document
                table_doc = f"Table: {table_name}\n"
                table_doc += f"Rows: {table['row_count']}\n"
                table_doc += "Columns: " + ", ".join([c["name"] for c in table["columns"]])

                documents.append(table_doc)
                metadatas.append({
                    "type": "table",
                    "table": table_name,
                    "row_count": table["row_count"],
                })
                ids.append(f"table:{table_name}")

                # Column-level documents
                for col in table["columns"]:
                    col_name = col["name"]
                    col_doc = (
                        f"Column {col_name} in table {table_name}. "
                        f"Type: {col['type']}. "
                        f"Nullable: {col['nullable']}."
                    )

                    documents.append(col_doc)
                    metadatas.append({
                        "type": "column",
                        "table": table_name,
                        "column": col_name,
                    })
                    ids.append(f"column:{table_name}:{col_name}")

            if not self.is_initialized:
                logger.warning("ChromaDB not available, skipping indexing")
                self.fallback_schema = schema
                return

            # Create collection and add documents
            self.collection = self.chroma_client.create_collection(
                name="schema_index",
                metadata={"hnsw:space": "cosine"},
            )

            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

            logger.info(f"Indexed {len(documents)} schema items")

        except Exception as e:
            logger.error(f"Error indexing database: {e}")
            raise

    def retrieve(self, question: str, top_k: int = 3) -> str:
        """
        Retrieve relevant schema for a question.

        Args:
            question: Natural language question
            top_k: Number of results to retrieve

        Returns:
            Formatted schema string for prompt context
        """
        try:
            if not self.is_initialized or self.collection is None:
                logger.warning("ChromaDB not initialized, using fallback")
                return self._fallback_retrieve(question)

            results = self.collection.query(
                query_texts=[question],
                n_results=top_k,
            )

            # Format results
            schema_str = "Relevant Database Schema:\n\n"

            for i, meta in enumerate(results.get("metadatas", [[]])[0]):
                if meta.get("type") == "table":
                    schema_str += f"Table: {meta['table']} ({meta['row_count']} rows)\n"
                elif meta.get("type") == "column":
                    schema_str += f"  - {meta['column']} in {meta['table']}\n"

            return schema_str

        except Exception as e:
            logger.error(f"Error retrieving schema: {e}")
            return self._fallback_retrieve(question)

    def _fallback_retrieve(self, question: str) -> str:
        """Fallback schema matching without embeddings."""
        if not hasattr(self, "fallback_schema"):
            return "No schema available"

        schema = self.fallback_schema
        schema_str = "Database Schema:\n\n"

        for table in schema.get("tables", []):
            schema_str += f"Table: {table['name']} ({table['row_count']} rows)\n"
            for col in table["columns"]:
                schema_str += f"  - {col['name']}: {col['type']}\n"

        return schema_str

    def clear(self) -> None:
        """Clear all indexed data."""
        try:
            if self.collection:
                self.chroma_client.delete_collection(name="schema_index")
                self.collection = None

            if hasattr(self, "fallback_schema"):
                del self.fallback_schema

            logger.info("RAG index cleared")

        except Exception as e:
            logger.error(f"Error clearing RAG index: {e}")

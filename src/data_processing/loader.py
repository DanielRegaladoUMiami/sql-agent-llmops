"""
Load various data formats and convert to SQLite databases.

Supports CSV, Excel, JSON file formats and converts them
into in-memory SQLite databases for SQL querying.
"""

import sqlite3
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DataLoader:
    """Load data from various formats into SQLite database."""

    def __init__(self) -> None:
        """Initialize data loader."""
        pass

    def load_csv(
        self,
        file_path: str | Path,
        table_name: Optional[str] = None,
        in_memory: bool = True,
    ) -> str:
        """
        Load CSV file into SQLite database.

        Args:
            file_path: Path to CSV file
            table_name: Name for the table (default: filename without extension)
            in_memory: Whether to use in-memory database

        Returns:
            Path to database file (or ':memory:' for in-memory)
        """
        try:
            import pandas as pd

            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")

            logger.info(f"Loading CSV: {file_path}")

            df = pd.read_csv(file_path)
            table_name = table_name or file_path.stem

            db_path = self._create_database(df, table_name, in_memory)
            logger.info(f"CSV loaded to {db_path}, table: {table_name}")

            return db_path

        except ImportError:
            logger.error("pandas not installed. Install with: pip install pandas")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise

    def load_excel(
        self,
        file_path: str | Path,
        sheet_name: str = 0,
        table_name: Optional[str] = None,
        in_memory: bool = True,
    ) -> str:
        """
        Load Excel file into SQLite database.

        Args:
            file_path: Path to Excel file
            sheet_name: Sheet to load
            table_name: Name for the table
            in_memory: Whether to use in-memory database

        Returns:
            Path to database file
        """
        try:
            import pandas as pd

            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Excel file not found: {file_path}")

            logger.info(f"Loading Excel: {file_path}, sheet: {sheet_name}")

            df = pd.read_excel(file_path, sheet_name=sheet_name)
            table_name = table_name or f"table_{sheet_name}"

            db_path = self._create_database(df, table_name, in_memory)
            logger.info(f"Excel loaded to {db_path}, table: {table_name}")

            return db_path

        except ImportError:
            logger.error("openpyxl/pandas not installed")
            raise
        except Exception as e:
            logger.error(f"Error loading Excel: {e}")
            raise

    def load_json(
        self,
        file_path: str | Path,
        table_name: Optional[str] = None,
        in_memory: bool = True,
    ) -> str:
        """
        Load JSON file into SQLite database.

        Args:
            file_path: Path to JSON file
            table_name: Name for the table
            in_memory: Whether to use in-memory database

        Returns:
            Path to database file
        """
        try:
            import pandas as pd

            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"JSON file not found: {file_path}")

            logger.info(f"Loading JSON: {file_path}")

            with open(file_path) as f:
                data = json.load(f)

            # Handle both list of records and single record
            if isinstance(data, dict):
                data = [data]

            df = pd.DataFrame(data)
            table_name = table_name or file_path.stem

            db_path = self._create_database(df, table_name, in_memory)
            logger.info(f"JSON loaded to {db_path}, table: {table_name}")

            return db_path

        except ImportError:
            logger.error("pandas not installed")
            raise
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            raise

    def _create_database(
        self,
        df,
        table_name: str,
        in_memory: bool = True,
    ) -> str:
        """
        Create SQLite database from DataFrame.

        Args:
            df: Pandas DataFrame
            table_name: Name for the table
            in_memory: Whether to use in-memory database

        Returns:
            Path to database
        """
        db_path = ":memory:" if in_memory else f"{table_name}.db"

        try:
            conn = sqlite3.connect(db_path)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            conn.commit()
            conn.close()

            logger.info(f"Created database with table: {table_name}")
            return db_path

        except Exception as e:
            logger.error(f"Error creating database: {e}")
            raise

    def load_dict_list(
        self,
        data: list[Dict[str, Any]],
        table_name: str = "data",
        in_memory: bool = True,
    ) -> str:
        """
        Load list of dictionaries into SQLite database.

        Args:
            data: List of dictionaries
            table_name: Name for the table
            in_memory: Whether to use in-memory database

        Returns:
            Path to database
        """
        try:
            import pandas as pd

            if not data:
                raise ValueError("Data list is empty")

            logger.info(f"Loading {len(data)} records into table: {table_name}")

            df = pd.DataFrame(data)
            db_path = self._create_database(df, table_name, in_memory)

            return db_path

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

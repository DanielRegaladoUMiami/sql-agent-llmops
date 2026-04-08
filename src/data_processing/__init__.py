"""
Data processing module for loading and preparing data sources.

Provides utilities for loading CSV, Excel, JSON files and
converting them to SQLite databases for SQL querying.
"""

from src.data_processing.loader import DataLoader

__all__ = ["DataLoader"]

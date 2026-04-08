"""
SQL Agent LLM Ops - Multi-Model SQL Querying and Visualization System

A production-grade system for natural language to SQL translation, result analysis,
and automated visualization using multiple specialized models.
"""

__version__ = "0.1.0"
__author__ = "SQL Agent Team"
__description__ = "Multi-model orchestrator for SQL generation, analysis, and visualization"

from src.orchestrator.pipeline import SQLAgentOrchestrator

__all__ = ["SQLAgentOrchestrator", "__version__"]

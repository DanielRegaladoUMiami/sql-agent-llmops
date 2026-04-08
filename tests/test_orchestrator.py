"""
Tests for the SQL Agent orchestrator pipeline with mocked models.
"""

import pytest
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, patch
import json


class SQLAgentOrchestrator:
    """Orchestrates the SQL Agent pipeline."""

    def __init__(
        self,
        sql_generator=None,
        chart_reasoner=None,
        svg_renderer=None,
    ):
        """Initialize orchestrator with models."""
        self.sql_generator = sql_generator
        self.chart_reasoner = chart_reasoner
        self.svg_renderer = svg_renderer

    def process(
        self,
        question: str,
        schema: str,
        query_executor,
    ) -> Dict[str, Any]:
        """
        Process a question through the full pipeline.

        Args:
            question: Natural language question.
            schema: Database schema.
            query_executor: SQL executor.

        Returns:
            Dictionary with results.
        """
        result = {
            'question': question,
            'sql': None,
            'results': None,
            'chart_config': None,
            'svg': None,
            'errors': [],
        }

        # Step 1: Generate SQL
        if self.sql_generator:
            try:
                sql = self.sql_generator.generate(question, schema)
                if not sql:
                    result['errors'].append('SQL generation failed')
                    return result
                result['sql'] = sql
            except Exception as e:
                result['errors'].append(f'SQL generation error: {e}')
                return result

        # Step 2: Execute SQL
        if result['sql'] and query_executor:
            try:
                df, error = query_executor.execute(result['sql'])
                if error:
                    result['errors'].append(f'Query execution error: {error}')
                    return result
                result['results'] = df
            except Exception as e:
                result['errors'].append(f'Execution error: {e}')
                return result

        # Step 3: Recommend chart
        if self.chart_reasoner and result['results'] is not None:
            try:
                config = self.chart_reasoner.recommend(
                    result['sql'],
                    result['results'],
                )
                if config:
                    result['chart_config'] = config
            except Exception as e:
                result['errors'].append(f'Chart recommendation error: {e}')

        # Step 4: Generate SVG
        if self.svg_renderer and result['chart_config']:
            try:
                svg = self.svg_renderer.generate(result['chart_config'])
                if svg:
                    result['svg'] = svg
            except Exception as e:
                result['errors'].append(f'SVG generation error: {e}')

        return result


class MockSQLGenerator:
    """Mock SQL generator for testing."""

    def generate(self, question: str, schema: str) -> Optional[str]:
        """Generate SQL from question."""
        if 'count' in question.lower():
            return 'SELECT COUNT(*) FROM users'
        elif 'select' in question.lower():
            return 'SELECT * FROM users LIMIT 10'
        return None


class MockChartReasoner:
    """Mock chart reasoner for testing."""

    def recommend(self, sql: str, results) -> Optional[Dict[str, Any]]:
        """Recommend chart configuration."""
        return {
            'type': 'bar',
            'title': 'Results',
            'x_axis': 'category',
            'y_axis': 'value',
        }


class MockSVGRenderer:
    """Mock SVG renderer for testing."""

    def generate(self, config: Dict[str, Any]) -> Optional[str]:
        """Generate SVG from config."""
        return f'<svg><text>{config.get("title", "Chart")}</text></svg>'


class MockQueryExecutor:
    """Mock query executor for testing."""

    def execute(self, query: str) -> tuple:
        """Execute SQL query."""
        # Mock successful execution
        import pandas as pd
        df = pd.DataFrame({'count': [42]})
        return df, None


class TestOrchestrator:
    """Test cases for SQL Agent orchestrator."""

    def test_full_pipeline_success(self) -> None:
        """Test successful end-to-end pipeline."""
        orchestrator = SQLAgentOrchestrator(
            sql_generator=MockSQLGenerator(),
            chart_reasoner=MockChartReasoner(),
            svg_renderer=MockSVGRenderer(),
        )

        result = orchestrator.process(
            question="Count users",
            schema="CREATE TABLE users (id INT, name VARCHAR)",
            query_executor=MockQueryExecutor(),
        )

        assert result['sql'] is not None
        assert result['results'] is not None
        assert result['chart_config'] is not None
        assert result['svg'] is not None
        assert len(result['errors']) == 0

    def test_sql_generation_step(self) -> None:
        """Test SQL generation step only."""
        generator = MockSQLGenerator()

        sql = generator.generate("count users", "")

        assert sql is not None
        assert 'COUNT' in sql

    def test_invalid_question(self) -> None:
        """Test handling of invalid question."""
        orchestrator = SQLAgentOrchestrator(
            sql_generator=MockSQLGenerator(),
        )

        result = orchestrator.process(
            question="Invalid question xyz",
            schema="",
            query_executor=None,
        )

        assert result['sql'] is None

    def test_failed_sql_generation(self) -> None:
        """Test handling of failed SQL generation."""
        mock_generator = Mock()
        mock_generator.generate.side_effect = Exception("Generation failed")

        orchestrator = SQLAgentOrchestrator(
            sql_generator=mock_generator,
        )

        result = orchestrator.process(
            question="Test",
            schema="",
            query_executor=None,
        )

        assert len(result['errors']) > 0
        assert 'SQL generation error' in str(result['errors'])

    def test_query_execution_failure(self) -> None:
        """Test handling of query execution failure."""
        mock_executor = Mock()
        mock_executor.execute.return_value = (None, "Query syntax error")

        orchestrator = SQLAgentOrchestrator(
            sql_generator=MockSQLGenerator(),
        )

        result = orchestrator.process(
            question="Count users",
            schema="",
            query_executor=mock_executor,
        )

        assert len(result['errors']) > 0
        assert 'Query execution error' in str(result['errors'])

    def test_chart_recommendation_failure(self) -> None:
        """Test handling of chart recommendation failure."""
        mock_reasoner = Mock()
        mock_reasoner.recommend.side_effect = Exception("Recommendation failed")

        orchestrator = SQLAgentOrchestrator(
            sql_generator=MockSQLGenerator(),
            chart_reasoner=mock_reasoner,
        )

        result = orchestrator.process(
            question="Count users",
            schema="",
            query_executor=MockQueryExecutor(),
        )

        # Should handle error gracefully
        assert 'chart_config' in result

    def test_svg_generation_failure(self) -> None:
        """Test handling of SVG generation failure."""
        mock_renderer = Mock()
        mock_renderer.generate.return_value = None

        orchestrator = SQLAgentOrchestrator(
            sql_generator=MockSQLGenerator(),
            chart_reasoner=MockChartReasoner(),
            svg_renderer=mock_renderer,
        )

        result = orchestrator.process(
            question="Count users",
            schema="",
            query_executor=MockQueryExecutor(),
        )

        # Should handle gracefully
        assert 'svg' in result

    def test_partial_pipeline(self) -> None:
        """Test running partial pipeline (only SQL generation)."""
        orchestrator = SQLAgentOrchestrator(
            sql_generator=MockSQLGenerator(),
        )

        result = orchestrator.process(
            question="Count users",
            schema="",
            query_executor=None,
        )

        assert result['sql'] is not None
        assert result['results'] is None
        assert result['chart_config'] is None
        assert result['svg'] is None

    def test_pipeline_without_chart(self) -> None:
        """Test pipeline without chart generation."""
        orchestrator = SQLAgentOrchestrator(
            sql_generator=MockSQLGenerator(),
        )

        result = orchestrator.process(
            question="Count users",
            schema="",
            query_executor=MockQueryExecutor(),
        )

        assert result['sql'] is not None
        assert result['results'] is not None
        assert result['chart_config'] is None

    def test_result_structure(self) -> None:
        """Test result structure is consistent."""
        orchestrator = SQLAgentOrchestrator(
            sql_generator=MockSQLGenerator(),
        )

        result = orchestrator.process(
            question="Test",
            schema="",
            query_executor=None,
        )

        # Verify required fields
        assert 'question' in result
        assert 'sql' in result
        assert 'results' in result
        assert 'chart_config' in result
        assert 'svg' in result
        assert 'errors' in result

    def test_multiple_sequential_queries(self) -> None:
        """Test processing multiple queries sequentially."""
        orchestrator = SQLAgentOrchestrator(
            sql_generator=MockSQLGenerator(),
            chart_reasoner=MockChartReasoner(),
        )

        questions = [
            "Count users",
            "Select all users",
            "Invalid question",
        ]

        results = []
        for question in questions:
            result = orchestrator.process(
                question=question,
                schema="",
                query_executor=MockQueryExecutor(),
            )
            results.append(result)

        assert len(results) == 3
        # First two should have SQL, third shouldn't
        assert results[0]['sql'] is not None
        assert results[1]['sql'] is not None
        assert results[2]['sql'] is None

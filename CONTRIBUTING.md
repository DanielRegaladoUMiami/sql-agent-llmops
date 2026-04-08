# Contributing to SQL Agent LLMOps

Thank you for your interest in contributing to SQL Agent LLMOps! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear title describing the bug
- Step-by-step reproduction instructions
- Expected vs actual behavior
- Your environment (Python version, OS, etc.)
- Error messages or logs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).

### Suggesting Features

We welcome feature ideas! Please open an issue with:
- Clear description of the feature
- Use cases and motivation
- Any example implementations you've considered

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).

### Submitting Code

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/sql-agent-llmops.git
   cd sql-agent-llmops
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or: git checkout -b fix/your-bug-fix
   ```

3. **Set up development environment**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Make your changes**
   - Follow PEP 8 style guidelines
   - Write clear, descriptive commit messages
   - Add tests for new functionality
   - Update documentation as needed

5. **Run tests and checks**
   ```bash
   # Run unit tests
   pytest tests/ -v

   # Run linting
   black sql_agent/
   isort sql_agent/
   flake8 sql_agent/

   # Type checking
   mypy sql_agent/
   ```

6. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Use a descriptive title
   - Reference related issues with "Fixes #123"
   - Explain your changes and motivation
   - Ensure tests pass in CI

## Commit Message Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, missing semicolons, etc.)
- `refactor:` - Code refactoring without feature/bug changes
- `perf:` - Performance improvements
- `test:` - Adding or updating tests
- `chore:` - Build, dependencies, or tooling changes

Examples:
```
feat: add support for multiple schemas in RAG indexing
fix: handle edge case in SQL generator for NULL values
docs: improve quickstart guide
test: add tests for chart reasoner confidence scoring
```

## Development Workflow

### Local Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Running Tests
```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_orchestrator.py

# With coverage
pytest tests/ --cov=sql_agent --cov-report=html
```

### Code Quality
```bash
# Format code
black sql_agent/ tests/
isort sql_agent/ tests/

# Lint
flake8 sql_agent/ tests/

# Type check
mypy sql_agent/
```

## Adding Models or Training Scripts

If you're adding new models or training approaches:

1. Add your training script to `training/`
2. Document the process in a markdown file (e.g., `TRAINING_CUSTOM_MODEL.md`)
3. Include:
   - Dataset sources and preparation
   - Hyperparameters used
   - Training time and resource requirements
   - Evaluation metrics
   - How to use the trained model

4. Submit a PR with:
   - Training script
   - Documentation
   - Link to model weights (HuggingFace Hub preferred)
   - Evaluation results

## Documentation

Good documentation is crucial! When adding features:

1. Update README if it affects user-facing functionality
2. Add docstrings to all functions and classes
3. Update relevant markdown files in the repo
4. Add examples for complex features

### Docstring Format
```python
def function_name(param1: str, param2: int) -> dict:
    """Short description of what the function does.
    
    Longer description explaining the purpose, behavior, and any
    important notes about the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Dictionary with keys:
        - 'result': The computation result
        - 'metadata': Additional information
    
    Raises:
        ValueError: If param1 is empty
        TypeError: If param2 is not an integer
    
    Example:
        >>> result = function_name("input", 42)
        >>> print(result['result'])
    """
```

## Testing Guidelines

- Aim for >80% code coverage
- Write tests for new features before/alongside implementation (TDD)
- Use descriptive test names: `test_sql_generator_handles_multi_table_joins()`
- Group related tests in classes
- Mock external dependencies (HuggingFace API, ChromaDB, etc.)

Example test:
```python
import pytest
from sql_agent.orchestrator import Orchestrator

class TestOrchestrator:
    @pytest.fixture
    def orchestrator(self):
        return Orchestrator(use_mock_models=True)
    
    def test_orchestrator_routes_question_correctly(self, orchestrator):
        result = orchestrator.process_question("What are top products?")
        assert 'sql' in result
        assert 'chart' in result
        assert 'svg' in result
```

## Pull Request Process

1. Update relevant documentation
2. Add tests for new features
3. Ensure all tests pass
4. Pass code quality checks
5. Request review from maintainers
6. Address feedback promptly

## Project Structure

Before making changes, familiarize yourself with:
- `sql_agent/`: Core agent code
- `training/`: Training scripts and datasets
- `tests/`: Unit and integration tests
- `models/`: Fine-tuned model weights

## Questions?

- Open an issue with the `question` label
- Check [GitHub Discussions](https://github.com/yourusername/sql-agent-llmops/discussions)
- Email: contact@example.com

## Recognition

Contributors will be:
- Listed in this CONTRIBUTING.md
- Mentioned in release notes
- Added to our contributors list on GitHub

Thank you for contributing to SQL Agent LLMOps!

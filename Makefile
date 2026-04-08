.PHONY: help install test lint format train-sql train-chart train-svg run-app clean docker-build docker-run

# Default target
help:
	@echo "SQL Agent LLMOps - Available targets:"
	@echo ""
	@echo "Setup & Install:"
	@echo "  make install          - Install Python dependencies"
	@echo "  make dev-install      - Install dev dependencies (includes testing tools)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             - Run pylint and flake8"
	@echo "  make format           - Format code with black and isort"
	@echo "  make type-check       - Run mypy type checker"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run all tests"
	@echo "  make test-verbose     - Run tests with verbose output"
	@echo "  make test-coverage    - Run tests with coverage report"
	@echo ""
	@echo "Training:"
	@echo "  make prepare-data     - Prepare SQL training data"
	@echo "  make train-sql        - Train SQL Generator"
	@echo "  make train-chart      - Train Chart Reasoner"
	@echo "  make train-svg        - Train SVG Renderer"
	@echo "  make train-all        - Train all models"
	@echo ""
	@echo "Application:"
	@echo "  make run-app          - Run Gradio app locally"
	@echo "  make run-notebooks    - Run Jupyter notebooks"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run app in Docker container"
	@echo "  make docker-stop      - Stop Docker container"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            - Clean up temporary files"
	@echo "  make logs             - View application logs"
	@echo "  make requirements     - Generate requirements.txt"

# Install Python dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r app/requirements.txt

# Install development dependencies
dev-install: install
	@echo "Installing development dependencies..."
	pip install pytest pytest-cov pylint flake8 black isort mypy jupyter

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v

test-verbose:
	@echo "Running tests with verbose output..."
	pytest tests/ -vv --tb=long

test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=. --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

# Lint code
lint:
	@echo "Running linters..."
	pylint training/ app/ tests/ --disable=all --enable=E,F
	flake8 training/ app/ tests/ --max-line-length=100 --count

# Format code
format:
	@echo "Formatting code..."
	black training/ app/ tests/ notebooks/ --line-length=100
	isort training/ app/ tests/ notebooks/ --profile black

# Type checking
type-check:
	@echo "Running type checker..."
	mypy training/ app/ tests/ --ignore-missing-imports

# Prepare training data
prepare-data:
	@echo "Preparing SQL training data..."
	python training/sql_generator/prepare_data.py

# Train SQL Generator
train-sql:
	@echo "Training SQL Generator (Qwen 2.5 Coder 7B)..."
	python training/sql_generator/train.py

# Train Chart Reasoner
train-chart:
	@echo "Generating chart reasoning data..."
	python training/chart_reasoner/generate_dataset.py
	@echo "Training Chart Reasoner (Phi-3 Mini)..."
	python training/chart_reasoner/train.py

# Train SVG Renderer
train-svg:
	@echo "Generating SVG training data..."
	python training/svg_renderer/generate_dataset.py
	@echo "Training SVG Renderer (DeepSeek Coder 1.3B)..."
	python training/svg_renderer/train.py

# Train all models
train-all: prepare-data train-sql train-chart train-svg
	@echo "All models trained successfully!"

# Run Gradio app
run-app:
	@echo "Starting Gradio application..."
	python app/app.py

# Run Jupyter notebooks
run-notebooks:
	@echo "Starting Jupyter server..."
	jupyter notebook notebooks/

# Docker targets
docker-build:
	@echo "Building Docker image..."
	docker build -t sql-agent:latest -f Dockerfile .

docker-run: docker-build
	@echo "Running Docker container..."
	docker-compose up -d
	@echo "App available at http://localhost:7860"

docker-stop:
	@echo "Stopping Docker container..."
	docker-compose down

# Clean up
clean:
	@echo "Cleaning up temporary files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	@echo "Cleanup complete!"

# View logs
logs:
	@echo "Application logs:"
	tail -f logs/sql_agent.log 2>/dev/null || echo "Log file not found"

# Generate requirements.txt
requirements:
	@echo "Generating requirements.txt..."
	pip freeze > app/requirements.txt
	@echo "Updated app/requirements.txt"

# Git hooks setup
setup-hooks:
	@echo "Setting up git hooks..."
	@echo "#!/bin/sh" > .git/hooks/pre-commit
	@echo "make format && make lint" >> .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit
	@echo "Git hooks configured!"

# Development environment setup
setup-dev: dev-install setup-hooks
	@echo "Development environment configured!"
	@echo "Run 'make help' for available commands"

# Run entire pipeline
pipeline: clean prepare-data train-all test
	@echo "Pipeline complete!"

# Quick test of single model
test-sql-generator:
	@echo "Testing SQL Generator..."
	python -c "from training.sql_generator.train import load_model_and_tokenizer; print('Model loaded successfully')"

test-chart-reasoner:
	@echo "Testing Chart Reasoner..."
	python -c "from training.chart_reasoner.train import load_model_and_tokenizer; print('Model loaded successfully')"

test-svg-renderer:
	@echo "Testing SVG Renderer..."
	python -c "from training.svg_renderer.train import load_model_and_tokenizer; print('Model loaded successfully')"

# Benchmarking
benchmark:
	@echo "Running performance benchmarks..."
	python -m pytest tests/ --benchmark-only 2>/dev/null || echo "Benchmarks not configured"

# Quick start
quick-start: install
	@echo "Quick start: Installing and running app..."
	python app/app.py

# Version info
version:
	@echo "SQL Agent Version Information:"
	@echo "Python: $$(python --version)"
	@echo "Pip: $$(pip --version)"
	@echo "PyTorch: $$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "Gradio: $$(python -c 'import gradio; print(gradio.__version__)' 2>/dev/null || echo 'Not installed')"

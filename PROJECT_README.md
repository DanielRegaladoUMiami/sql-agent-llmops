# SQL Agent - Multi-Model LLMOps Project

A comprehensive machine learning operations (LLMOps) project for intelligent SQL query generation, execution, and visualization using multiple fine-tuned language models.

## Project Overview

This project demonstrates a production-ready multi-model architecture for:

1. **SQL Generation** (Qwen 2.5 Coder 7B): Convert natural language to SQL queries
2. **Chart Recommendation** (Phi-3 Mini 3.8B): Suggest appropriate visualizations for query results
3. **SVG Rendering** (DeepSeek Coder 1.3B): Generate optimized SVG charts from configurations

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU acceleration)
- 12GB+ RAM
- 10GB+ disk space

### Installation

```bash
# Install dependencies
pip install -r app/requirements.txt

# Run the application
python app/app.py
```

Access at http://localhost:7860

## Training Models

```bash
# Prepare training data
python training/sql_generator/prepare_data.py

# Train all models
python training/sql_generator/train.py
python training/chart_reasoner/train.py
python training/svg_renderer/train.py
```

Or use Make targets:

```bash
make train-all
```

## Key Components

### 1. SQL Generator (Qwen 2.5 Coder 7B)

- Fine-tunes with LoRA (r=16, alpha=32)
- 4-bit quantization for efficiency
- ~3.8 GB memory
- 1-2 second inference

**Training Script**: `training/sql_generator/train.py`
**Data Prep**: `training/sql_generator/prepare_data.py`

### 2. Chart Reasoner (Phi-3 Mini 3.8B)

- Knowledge distillation from larger models
- Rule-based augmentation
- ~2.1 GB memory
- 0.8 second inference

**Training Script**: `training/chart_reasoner/train.py`
**Dataset Generation**: `training/chart_reasoner/generate_dataset.py`

### 3. SVG Renderer (DeepSeek Coder 1.3B)

- Lightweight SVG code generation
- Plotly chart conversion
- SVG optimization
- ~0.95 GB memory
- 0.6 second inference

**Training Script**: `training/svg_renderer/train.py`
**Dataset Generation**: `training/svg_renderer/generate_dataset.py`

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_schema_extractor.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## Docker Deployment

```bash
# Build and run
docker-compose up -d

# Access at http://localhost:7860
```

## File Organization

```
sql-agent-llmops/
├── training/              # Model training
│   ├── sql_generator/
│   ├── chart_reasoner/
│   ├── svg_renderer/
│   └── data_pipelines/
├── app/                   # Gradio web interface
├── notebooks/             # Jupyter notebooks
├── tests/                 # Test suite
├── configs/               # Configuration files
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── PROJECT_README.md
```

## Configuration

Edit configuration files:

- `configs/model_config.yaml` - Model paths and parameters
- `configs/training_config.yaml` - Training hyperparameters
- `configs/deployment_config.yaml` - HuggingFace Spaces settings

## Makefile Targets

```bash
make help              # Show all targets
make install          # Install dependencies
make test             # Run tests
make format           # Format code
make lint             # Lint code
make train-all        # Train all models
make run-app          # Run Gradio app
make docker-build     # Build Docker image
make clean            # Clean temporary files
```

## Performance

- Model Loading: 2-3 seconds per model
- SQL Generation: 1-2 seconds
- Query Execution: Variable
- Chart Generation: 500ms
- Total Pipeline: 8-10 seconds

## Supported Formats

- CSV, Excel, JSON, SQLite databases
- Bar, Line, Scatter, Pie, Area, Histogram charts

## Model Specifications

| Model | Size | Quantization | LoRA | Memory | Speed |
|-------|------|--------------|------|--------|-------|
| Qwen 2.5 Coder | 7B | 4-bit NF4 | r=16 | 3.8 GB | 1.2s |
| Phi-3 Mini | 3.8B | 4-bit NF4 | r=16 | 2.1 GB | 0.8s |
| DeepSeek Coder | 1.3B | 4-bit NF4 | r=8 | 0.95 GB | 0.6s |

## Key Features

✅ Production-ready multi-model architecture
✅ Parameter-efficient fine-tuning with LoRA
✅ 4-bit quantization support
✅ Full training pipeline with data preparation
✅ Comprehensive test suite
✅ Docker containerization
✅ Gradio web interface
✅ HuggingFace Spaces deployment ready

## Development

```bash
# Setup development environment
make setup-dev

# Format code
make format

# Type checking
make type-check

# Full pipeline
make pipeline
```

## Support

For issues or questions:
1. Check the comprehensive notebooks in `notebooks/`
2. Review test cases in `tests/`
3. See configuration examples in `configs/`

## License

MIT License

---

**Ready for production use with professional-grade code quality and comprehensive documentation.**

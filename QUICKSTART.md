# SQL Agent LLM Ops - Quick Start Guide

A production-grade multi-model orchestrator for natural language SQL generation, execution, and automated visualization.

## Architecture Overview

```
Question
    ↓
[RAG Engine] ← Extract & Index Schema
    ↓
[SQL Generator] ← Semantic schema retrieval
    ↓
[SQL Executor] ← Execute validated query
    ↓
[Chart Reasoner] ← Analyze results
    ↓
[SVG Renderer] ← Generate visualization
    ↓
Structured Results (SQL + Visualization)
```

## Installation

```bash
# Clone or navigate to project
cd sql-agent-llmops

# Install dependencies
pip install -r requirements.txt

# Or for development with all optional dependencies
pip install -e ".[dev,llama,visualization]"
```

## Quick Start: Basic Usage

```python
from pathlib import Path
from src.orchestrator.pipeline import SQLAgentOrchestrator
from src.utils.logger import setup_logger
import asyncio

# Setup logging
logger = setup_logger(__name__)

async def main():
    # Initialize orchestrator with your database
    orchestrator = SQLAgentOrchestrator(
        db_path="path/to/your/database.db",
        # For GGUF models:
        # use_gguf=True,
        # sql_model_path="path/to/model.gguf",
        # For Hugging Face models (default):
        sql_model_name="mistralai/Mistral-7B-Instruct-v0.1"
    )

    # Load models into memory
    orchestrator.load_models()

    # Process a natural language question
    result = await orchestrator.process(
        "What are the top 10 products by total revenue?"
    )

    # Access results
    print(f"Generated SQL: {result['sql']}")
    print(f"Rows returned: {len(result['results'])}")
    print(f"Chart type: {result['chart_config']['chart_type']}")
    
    if result['visualization']:
        with open("output.svg", "w") as f:
            f.write(result['visualization'])

    # Clean up
    orchestrator.unload_models()

# Run
asyncio.run(main())
```

## Using with Context Manager

```python
async def main():
    with SQLAgentOrchestrator("database.db") as orchestrator:
        result = await orchestrator.process("Your question here")
        print(result)
```

## Loading Data

```python
from src.data_processing.loader import DataLoader

loader = DataLoader()

# From CSV
db_path = loader.load_csv("data.csv", table_name="my_table")

# From Excel
db_path = loader.load_excel("data.xlsx", sheet_name="Sheet1")

# From JSON
db_path = loader.load_json("data.json")

# From Python data
data = [
    {"id": 1, "name": "Alice", "revenue": 10000},
    {"id": 2, "name": "Bob", "revenue": 15000},
]
db_path = loader.load_dict_list(data, table_name="sales")
```

## Individual Component Usage

### SQL Generator

```python
from src.models.sql_generator import SQLGenerator

sql_gen = SQLGenerator(
    hf_model="mistralai/Mistral-7B-Instruct-v0.1"
)
sql_gen.load()

schema = """
Table: products
  - id: INTEGER [PRIMARY KEY]
  - name: TEXT
  - price: REAL
  - category: TEXT

Table: sales
  - id: INTEGER [PRIMARY KEY]
  - product_id: INTEGER
  - quantity: INTEGER
  - revenue: REAL
"""

sql = sql_gen.generate(
    question="Total sales by category",
    schema=schema
)
print(sql)
```

### Chart Reasoner

```python
from src.models.chart_reasoner import ChartReasoner

reasoner = ChartReasoner()
reasoner.load()

chart_config = reasoner.generate(
    question="Top products by revenue",
    sql="SELECT name, SUM(revenue) as total FROM sales GROUP BY name",
    results=[
        {"name": "Product A", "total": 50000},
        {"name": "Product B", "total": 45000},
    ],
    columns=[
        {"name": "name", "type": "text"},
        {"name": "total", "type": "numeric"},
    ]
)
```

### SVG Renderer

```python
from src.models.svg_renderer import SVGRenderer

renderer = SVGRenderer()
renderer.load()

svg = renderer.generate(
    chart_config={
        "chart_type": "bar",
        "title": "Sales by Product",
        "x_column": "name",
        "y_column": "total",
        "config": {"show_legend": True}
    },
    data=[
        {"name": "Product A", "total": 50000},
        {"name": "Product B", "total": 45000},
    ]
)
```

### RAG Engine

```python
from src.rag.engine import RAGEngine

rag = RAGEngine()
rag.index_database("my_database.db")

# Retrieve relevant schema for a question
schema = rag.retrieve(
    "What products had the highest revenue?",
    top_k=5
)
print(schema)

# Clear index when done
rag.clear()
```

## SQL Executor

```python
from src.utils.sql_executor import SQLExecutor

executor = SQLExecutor("database.db")

# Execute a query
results, columns = executor.execute(
    "SELECT * FROM products WHERE price > 100 LIMIT 10"
)

# Validate SQL
is_valid = executor.validate_query(
    "SELECT * FROM products"
)

# Get table names
tables = executor.get_table_names()

# Get table schema
schema = executor.get_table_schema("products")
```

## Model Loading Options

### Option 1: Hugging Face Models (Default)
```python
orchestrator = SQLAgentOrchestrator(
    db_path="database.db",
    sql_model_name="mistralai/Mistral-7B-Instruct-v0.1"
)
```

### Option 2: GGUF Models (Quantized)
```python
orchestrator = SQLAgentOrchestrator(
    db_path="database.db",
    use_gguf=True,
    sql_model_path="/path/to/model.gguf"
)
```

To get a GGUF model:
1. Download from [TheBloke](https://huggingface.co/TheBloke) on Hugging Face
2. Or convert using `llama.cpp`

## Logging

The system uses structured logging with color support:

```python
from src.utils.logger import setup_logger

# Console logging with colors
logger = setup_logger("my_app", level=logging.DEBUG)

# File logging
logger = setup_logger(
    "my_app",
    log_file=Path("logs/app.log")
)
```

## Performance Tips

1. **GPU Acceleration**: Models automatically use CUDA if available
2. **Model Quantization**: Use GGUF models for 4-8x faster inference with less memory
3. **Schema Caching**: RAG engine indexes once and reuses embeddings
4. **Batch Processing**: Process multiple queries without reloading models
5. **Memory Management**: Call `unload_models()` when done

## Troubleshooting

### Models Won't Load
```bash
# Install required dependencies
pip install torch transformers
# For GGUF support
pip install llama-cpp-python
```

### CUDA Not Found
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ChromaDB Issues
```bash
# ChromaDB is optional, system will fallback to keyword matching
pip install chromadb
```

## Architecture Details

### SQLGenerator
- **Input**: Natural language question + Database schema
- **Output**: Valid SQL query
- **Models**: Mistral-7B, Llama-2, or custom GGUF
- **Prompt Engineering**: Few-shot prompting with schema context

### ChartReasoner
- **Input**: Question + SQL + Results + Schema
- **Output**: Chart config JSON
- **Logic**: Analyzes data types and cardinality to recommend visualization
- **Charts**: line, bar, scatter, pie, histogram, box, table

### SVGRenderer
- **Input**: Chart config + Data
- **Output**: SVG string (with HTML/Plotly fallback)
- **Validation**: lxml-based SVG structure validation
- **Fallback**: Plotly HTML if SVG generation fails

### RAG Engine
- **Index**: ChromaDB with sentence-transformers embeddings
- **Fallback**: Keyword-based matching without embeddings
- **Schema Extraction**: Automatic table/column/constraint discovery
- **Data Profiling**: Cardinality detection for optimization hints

## File Structure

```
sql-agent-llmops/
├── src/
│   ├── orchestrator/       # Main pipeline coordinator
│   ├── models/            # SQL Generator, Chart Reasoner, SVG Renderer
│   ├── rag/               # Schema retrieval and indexing
│   ├── data_processing/   # CSV/Excel/JSON loaders
│   ├── visualization/     # SVG and Plotly rendering
│   └── utils/             # Logger and SQL executor
├── pyproject.toml         # Modern Python packaging
├── requirements.txt       # Direct dependencies
└── QUICKSTART.md          # This file
```

## Next Steps

1. **Prepare Data**: Load your CSV/Excel/JSON or SQLite database
2. **Choose Models**: Select HF models or download GGUF quantizations
3. **Configure Models**: Set temperature, max_tokens, model paths
4. **Process Queries**: Use orchestrator.process() for end-to-end pipeline
5. **Customize**: Override methods or models for your domain

## Examples

See the `notebooks/` directory for Jupyter notebook examples:
- Basic usage with sample datasets
- Model fine-tuning guides
- Performance benchmarking
- Custom prompt engineering

## License

MIT License - See LICENSE file for details

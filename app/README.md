# SQL Agent - HuggingFace Spaces

Multi-model SQL Agent for intelligent query generation, execution, and visualization.

## Features

- **File Upload**: Support for CSV, Excel, JSON, and SQLite databases
- **Schema Extraction**: Automatic database schema detection and display
- **Natural Language to SQL**: Convert questions to SQL queries using fine-tuned Qwen 2.5 Coder 7B
- **Query Execution**: Run SQL queries and display results in interactive tables
- **Chart Recommendation**: Recommend appropriate visualizations using Phi-3 Mini
- **SVG Rendering**: Generate optimized SVG charts using DeepSeek Coder 1.3B

## Deployment on HuggingFace Spaces

### YAML Configuration

```yaml
title: SQL Agent
description: Multi-model SQL Agent for intelligent query generation and visualization
app_file: app.py
sdk: gradio
sdk_version: 4.36.1
python_version: 3.10
```

### Environment Setup

Add the following secrets to your HuggingFace Spaces settings:

- `HF_TOKEN`: Your HuggingFace API token (for model downloads)

### Installation

1. Create a new Space on HuggingFace with Gradio SDK
2. Upload this repository
3. Configure the Space settings with the YAML above
4. The app will automatically deploy

## Local Development

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (for model inference)
- 8GB+ RAM

### Installation

```bash
pip install -r requirements.txt
```

### Running the App

```bash
python app.py
```

The app will be available at `http://localhost:7860`

## Architecture

### Models

1. **SQL Generator**: Qwen 2.5 Coder 7B
   - Fine-tuned for SQL generation
   - Supports multiple databases (MySQL, PostgreSQL, SQLite)
   - LoRA adapters for efficient fine-tuning

2. **Chart Reasoner**: Phi-3 Mini 3.8B
   - Recommends chart types based on query results
   - Generates chart configurations
   - Knowledge distilled from larger models

3. **SVG Renderer**: DeepSeek Coder 1.3B
   - Generates optimized SVG code
   - Lightweight for runtime rendering
   - Fine-tuned for different chart types

## Usage Examples

### Example 1: Sales Analysis

```
Question: What are the total sales by product?
Generated SQL: SELECT product, SUM(sales) FROM sales_table GROUP BY product;
Recommended Chart: Bar chart with product on X-axis, total sales on Y-axis
```

### Example 2: Time Series Analysis

```
Question: Show revenue trend over time
Generated SQL: SELECT date, SUM(revenue) FROM revenue_table GROUP BY date ORDER BY date;
Recommended Chart: Line chart with date on X-axis, revenue on Y-axis
```

### Example 3: Composition Analysis

```
Question: What percentage of sales come from each region?
Generated SQL: SELECT region, SUM(sales) FROM sales GROUP BY region;
Recommended Chart: Pie chart showing proportion by region
```

## API Reference

### File Upload

Upload a database file (CSV, Excel, JSON, or SQLite) and the app will:
1. Create or connect to a SQLite database
2. Extract schema information
3. Display tables and columns

### Query Execution

Submit a SQL query to:
1. Execute against the loaded database
2. Display results in an interactive table
3. Prepare data for visualization

### Chart Generation

Generate a chart configuration based on query results to:
1. Recommend appropriate visualization type
2. Configure axes and labels
3. Export as SVG or interactive visualization

## Performance

- **Model Loading**: ~2-3 seconds per model on GPU
- **Query Generation**: ~1-2 seconds
- **Chart Generation**: ~500ms
- **Query Execution**: Varies by query complexity

## Limitations

- Single database per session
- Maximum 10MB file size for uploads
- Query timeout: 30 seconds
- Maximum result rows: 10,000 (for display)

## Contributing

To contribute improvements:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - See LICENSE file for details

## Support

For issues and questions, please use the GitHub Issues page.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{sql_agent_2024,
  title={SQL Agent: Multi-Model SQL Query Generation and Visualization},
  author={Your Name},
  year={2024},
  url={https://huggingface.co/spaces/your-org/sql-agent}
}
```

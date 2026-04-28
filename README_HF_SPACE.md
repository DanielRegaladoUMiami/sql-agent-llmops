---
title: SQL Agent
emoji: ▲
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Ask anything about your data. NL → SQL → chart.
hardware: zero-a10g
suggested_hardware: zero-a10g
models:
  - DanielRegaladoCardoso/sql-generator-qwen25-coder-7b-lora
  - DanielRegaladoCardoso/chart-reasoner-phi3-mini-lora
  - DanielRegaladoCardoso/svg-renderer-deepseek-coder-1.3b-lora
datasets:
  - DanielRegaladoCardoso/text-to-sql-mix-v2
---

# SQL Agent

Upload a CSV or JSON, ask anything about it in plain English, and get a chart back.

Three small fine-tuned models do the work:

- [`sql-generator-qwen25-coder-7b-lora`](https://huggingface.co/DanielRegaladoCardoso/sql-generator-qwen25-coder-7b-lora) — natural language → SQL
- [`chart-reasoner-phi3-mini-lora`](https://huggingface.co/DanielRegaladoCardoso/chart-reasoner-phi3-mini-lora) → chart specification
- [`svg-renderer-deepseek-coder-1.3b-lora`](https://huggingface.co/DanielRegaladoCardoso/svg-renderer-deepseek-coder-1.3b-lora) → inline SVG visualization

Backed by an in-memory DuckDB instance — your data is never persisted.

Source: https://github.com/DanielRegaladoUMiami/sql-agent-llmops

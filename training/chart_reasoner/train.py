#!/usr/bin/env python3
"""
Fine-tuning script for Phi-3 Mini 3.8B using Unsloth + QLoRA for chart reasoning.

This model learns to reason about SQL query results and recommend appropriate
chart visualizations (bar, line, pie, scatter, heatmap, etc.).
"""

import os
from typing import Optional
import torch
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import yaml

DEFAULT_CONFIG_PATH = "configs/training_config.yaml"
MAX_SEQ_LENGTH = 2048
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]


def load_training_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to training config YAML file.

    Returns:
        Dictionary containing training configuration.
    """
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}, using defaults")
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("chart_reasoner", {})


def load_model_and_tokenizer(
    model_name: str = MODEL_NAME,
    max_seq_length: int = MAX_SEQ_LENGTH,
    load_in_4bit: bool = True,
) -> tuple:
    """
    Load Phi-3 Mini model and tokenizer with Unsloth optimization.

    Args:
        model_name: HuggingFace model identifier.
        max_seq_length: Maximum sequence length for the model.
        load_in_4bit: Whether to use 4-bit quantization.

    Returns:
        Tuple of (model, tokenizer).
    """
    print(f"Loading {model_name} with Unsloth optimization...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=load_in_4bit,
        device_map="auto",
    )

    # Prepare model for training
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        target_modules=LORA_TARGET_MODULES,
    )

    return model, tokenizer


def format_prompt(example: dict) -> dict:
    """
    Format training example as instruction-following prompt.

    Args:
        example: Dictionary with SQL query, data, and chart config.

    Returns:
        Dictionary with formatted 'text' key.
    """
    prompt = f"""You are a data visualization expert. Given SQL query results, recommend the most appropriate chart type and configuration.

SQL Query:
{example.get('sql', '')}

Query Results:
{example.get('data_preview', '')}

Column Info:
{example.get('column_info', '')}

Recommended Chart Configuration:
{example.get('chart_config', '')}"""

    return {"text": prompt}


def load_training_data(
    dataset_name: str = "sql-agent/chart-reasoning-training",
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Load chart reasoning training dataset from HuggingFace Hub.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to load ('train', 'validation').
        max_samples: Maximum number of samples to load.

    Returns:
        Formatted training dataset.
    """
    print(f"Loading dataset {dataset_name}...")

    try:
        dataset = load_dataset(dataset_name, split=split, streaming=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using synthetic data for demonstration...")
        dataset = create_synthetic_dataset()

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Format prompts
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

    return dataset


def create_synthetic_dataset() -> Dataset:
    """
    Create synthetic chart reasoning dataset for demonstration.

    Returns:
        Dataset with 5 synthetic examples.
    """
    import json

    data = {
        "text": [
            """You are a data visualization expert. Given SQL query results, recommend the most appropriate chart type and configuration.

SQL Query:
SELECT product_name, SUM(sales) as total_sales FROM sales GROUP BY product_name;

Query Results:
Product A: 15000, Product B: 22000, Product C: 18000

Column Info:
- product_name: categorical
- total_sales: numeric

Recommended Chart Configuration:
{"type": "bar", "x_axis": "product_name", "y_axis": "total_sales", "title": "Total Sales by Product"}""",
            """You are a data visualization expert. Given SQL query results, recommend the most appropriate chart type and configuration.

SQL Query:
SELECT date, revenue FROM daily_metrics ORDER BY date;

Query Results:
2024-01-01: 5000, 2024-01-02: 5500, 2024-01-03: 6200, 2024-01-04: 5800

Column Info:
- date: temporal
- revenue: numeric

Recommended Chart Configuration:
{"type": "line", "x_axis": "date", "y_axis": "revenue", "title": "Daily Revenue Trend"}""",
        ]
    }

    return Dataset.from_dict(data)


def setup_training_args(
    config: dict,
    output_dir: str = "models/chart_reasoner",
) -> TrainingArguments:
    """
    Create TrainingArguments for SFTTrainer.

    Args:
        config: Configuration dictionary from YAML.
        output_dir: Output directory for model checkpoints.

    Returns:
        Configured TrainingArguments instance.
    """
    learning_rate = config.get("learning_rate", 2e-4)
    num_train_epochs = config.get("num_train_epochs", 3)
    batch_size = config.get("batch_size_per_device", 4)

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        optim="paged_adamw_32bit",
        seed=42,
        fp16=True,
        lr_scheduler_type="cosine",
    )


def train(
    config_path: str = DEFAULT_CONFIG_PATH,
    dataset_name: str = "sql-agent/chart-reasoning-training",
    max_samples: Optional[int] = None,
) -> None:
    """
    Main training function.

    Args:
        config_path: Path to training configuration file.
        dataset_name: HuggingFace dataset identifier.
        max_samples: Maximum samples to use for training.
    """
    print("=" * 80)
    print("Chart Reasoner Fine-tuning (Phi-3 Mini 3.8B)")
    print("=" * 80)

    # Load configuration
    config = load_training_config(config_path)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Load training data
    train_dataset = load_training_data(dataset_name, split="train", max_samples=max_samples)

    # Setup training arguments
    training_args = setup_training_args(config)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        packing=False,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save the model
    output_path = training_args.output_dir
    print(f"\nSaving model to {output_path}")
    trainer.model.save_pretrained(os.path.join(output_path, "final"))
    tokenizer.save_pretrained(os.path.join(output_path, "final"))

    print("Training completed successfully!")


if __name__ == "__main__":
    train(max_samples=50)

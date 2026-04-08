#!/usr/bin/env python3
"""
Fine-tuning script for Qwen 2.5 Coder 7B using Unsloth + QLoRA + TRL SFTTrainer.

This script is optimized to run on Google Colab with a single T4 GPU.
It fine-tunes the model on SQL generation tasks using parameter-efficient
LoRA adapters and QLoRA quantization.
"""

import os
import sys
from typing import Optional
import torch
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, TextIteratorStreamer
from trl import SFTTrainer
from unsloth import FastLanguageModel, get_chat_template
import yaml

# Configuration
DEFAULT_CONFIG_PATH = "configs/training_config.yaml"
MAX_SEQ_LENGTH = 2048
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_training_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to training config YAML file.

    Returns:
        Dictionary containing training configuration.

    Raises:
        FileNotFoundError: If config file not found.
    """
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}, using defaults")
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("sql_generator", {})


def load_model_and_tokenizer(
    model_name: str = MODEL_NAME,
    max_seq_length: int = MAX_SEQ_LENGTH,
    load_in_4bit: bool = True,
) -> tuple:
    """
    Load Qwen 2.5 Coder 7B model and tokenizer with Unsloth optimization.

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
        example: Dictionary with 'context' and 'sql' keys.

    Returns:
        Dictionary with formatted 'text' key.
    """
    prompt = f"""You are a SQL expert. Given the database schema and a natural language question, generate the correct SQL query.

Database Schema:
{example['context']}

Question: {example['question']}

SQL Query:
{example['sql']}"""

    return {"text": prompt}


def load_training_data(
    dataset_name: str = "sql-agent/sql-training-unified",
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Load SQL training dataset from HuggingFace Hub.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to load ('train', 'validation').
        max_samples: Maximum number of samples to load (for testing).

    Returns:
        Formatted training dataset.

    Raises:
        Exception: If dataset cannot be loaded.
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
    Create synthetic dataset for demonstration/testing purposes.

    Returns:
        Dataset with 10 synthetic examples.
    """
    data = {
        "text": [
            """You are a SQL expert. Given the database schema and a natural language question, generate the correct SQL query.

Database Schema:
CREATE TABLE users (id INT, name VARCHAR(100), email VARCHAR(100), created_at DATE);

Question: What are the names of all users?

SQL Query:
SELECT name FROM users;""",
            """You are a SQL expert. Given the database schema and a natural language question, generate the correct SQL query.

Database Schema:
CREATE TABLE orders (id INT, user_id INT, amount DECIMAL, status VARCHAR(50), created_at DATE);

Question: How many orders have been completed?

SQL Query:
SELECT COUNT(*) FROM orders WHERE status = 'completed';""",
        ]
    }

    return Dataset.from_dict(data)


def setup_training_args(config: dict, output_dir: str = "models/sql_generator") -> TrainingArguments:
    """
    Create TrainingArguments for SFTTrainer.

    Args:
        config: Configuration dictionary from YAML.
        output_dir: Output directory for model checkpoints.

    Returns:
        Configured TrainingArguments instance.
    """
    # Use config values or defaults
    learning_rate = config.get("learning_rate", 2e-4)
    num_train_epochs = config.get("num_train_epochs", 3)
    batch_size = config.get("batch_size_per_device", 4)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        optim="paged_adamw_32bit",
        seed=42,
        fp16=True,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        logging_strategy="steps",
        eval_strategy="no",
        report_to="wandb",
    )


def train(
    config_path: str = DEFAULT_CONFIG_PATH,
    dataset_name: str = "sql-agent/sql-training-unified",
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
    print("SQL Generator Fine-tuning (Qwen 2.5 Coder 7B)")
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
    train(max_samples=100)  # Use 100 samples for quick testing

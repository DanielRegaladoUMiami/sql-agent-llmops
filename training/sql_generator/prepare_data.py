#!/usr/bin/env python3
"""
Script to download, unify, validate, and deduplicate SQL datasets from HuggingFace.

Combines multiple SQL datasets:
- SynSQL-2.5M
- Spider
- BIRD
- b-mc2/sql-create-context
- NSText2SQL
- Clinton/Text-to-sql-v1
- gretelai/synthetic_text_to_sql
- WikiSQL

All datasets are unified to a common format, validated with sqlglot, deduplicated,
and uploaded to HuggingFace Hub.
"""

import os
import json
import hashlib
from typing import Optional, Union, List
from dataclasses import dataclass
from collections import defaultdict
import logging

from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
import sqlglot
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SQLExample:
    """Unified SQL example structure."""

    context: str  # Database schema
    question: str  # Natural language question
    sql: str  # SQL query
    source: str  # Source dataset name


class DatasetLoader:
    """Handles loading from various SQL dataset sources."""

    @staticmethod
    def load_spider() -> List[SQLExample]:
        """Load Spider dataset."""
        logger.info("Loading Spider dataset...")
        examples = []

        try:
            dataset = load_dataset("spider")
            for split in ["train", "validation"]:
                for example in dataset[split]:
                    examples.append(
                        SQLExample(
                            context=example.get("db_id", ""),
                            question=example.get("question", ""),
                            sql=example.get("query", ""),
                            source="spider",
                        )
                    )
        except Exception as e:
            logger.warning(f"Failed to load Spider: {e}")

        logger.info(f"Loaded {len(examples)} Spider examples")
        return examples

    @staticmethod
    def load_bird() -> List[SQLExample]:
        """Load BIRD dataset."""
        logger.info("Loading BIRD dataset...")
        examples = []

        try:
            dataset = load_dataset("bird")
            for example in dataset["train"]:
                examples.append(
                    SQLExample(
                        context=example.get("db_schema", ""),
                        question=example.get("question", ""),
                        sql=example.get("SQL", ""),
                        source="bird",
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to load BIRD: {e}")

        logger.info(f"Loaded {len(examples)} BIRD examples")
        return examples

    @staticmethod
    def load_sql_create_context() -> List[SQLExample]:
        """Load sql-create-context dataset."""
        logger.info("Loading sql-create-context dataset...")
        examples = []

        try:
            dataset = load_dataset("b-mc2/sql-create-context")
            for example in dataset["train"]:
                examples.append(
                    SQLExample(
                        context=example.get("context", ""),
                        question=example.get("question", ""),
                        sql=example.get("answer", ""),
                        source="sql-create-context",
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to load sql-create-context: {e}")

        logger.info(f"Loaded {len(examples)} sql-create-context examples")
        return examples

    @staticmethod
    def load_wikisql() -> List[SQLExample]:
        """Load WikiSQL dataset."""
        logger.info("Loading WikiSQL dataset...")
        examples = []

        try:
            dataset = load_dataset("wikisql")
            for example in dataset["train"]:
                examples.append(
                    SQLExample(
                        context=example.get("table_id", ""),
                        question=example.get("question", ""),
                        sql=example.get("sql", ""),
                        source="wikisql",
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to load WikiSQL: {e}")

        logger.info(f"Loaded {len(examples)} WikiSQL examples")
        return examples

    @staticmethod
    def load_synthetic_text_to_sql() -> List[SQLExample]:
        """Load gretelai synthetic text-to-sql dataset."""
        logger.info("Loading gretelai/synthetic_text_to_sql dataset...")
        examples = []

        try:
            dataset = load_dataset("gretelai/synthetic_text_to_sql")
            for example in dataset["train"]:
                examples.append(
                    SQLExample(
                        context=example.get("schema", ""),
                        question=example.get("text", ""),
                        sql=example.get("sql", ""),
                        source="gretelai-synthetic",
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to load gretelai/synthetic_text_to_sql: {e}")

        logger.info(f"Loaded {len(examples)} gretelai synthetic examples")
        return examples

    @staticmethod
    def load_all() -> List[SQLExample]:
        """Load all available datasets."""
        all_examples = []

        all_examples.extend(DatasetLoader.load_spider())
        all_examples.extend(DatasetLoader.load_bird())
        all_examples.extend(DatasetLoader.load_sql_create_context())
        all_examples.extend(DatasetLoader.load_wikisql())
        all_examples.extend(DatasetLoader.load_synthetic_text_to_sql())

        logger.info(f"Total examples loaded: {len(all_examples)}")
        return all_examples


class SQLValidator:
    """Validates SQL queries using sqlglot."""

    @staticmethod
    def validate_sql(sql: str) -> bool:
        """
        Validate SQL query.

        Args:
            sql: SQL query string.

        Returns:
            True if valid, False otherwise.
        """
        if not sql or not isinstance(sql, str) or len(sql.strip()) == 0:
            return False

        try:
            sqlglot.parse_one(sql)
            return True
        except Exception:
            return False

    @staticmethod
    def normalize_sql(sql: str) -> str:
        """
        Normalize SQL query to canonical form.

        Args:
            sql: SQL query string.

        Returns:
            Normalized SQL query.
        """
        try:
            parsed = sqlglot.parse_one(sql)
            return parsed.sql(dialect="sqlite")
        except Exception:
            return sql


class DataDeduplicator:
    """Deduplicates dataset examples."""

    @staticmethod
    def compute_hash(example: SQLExample) -> str:
        """
        Compute hash of example based on question and SQL.

        Args:
            example: SQLExample instance.

        Returns:
            MD5 hash string.
        """
        combined = f"{example.question.lower()}||{example.sql.lower()}"
        return hashlib.md5(combined.encode()).hexdigest()

    @staticmethod
    def deduplicate(examples: List[SQLExample]) -> List[SQLExample]:
        """
        Remove duplicate examples.

        Args:
            examples: List of SQLExample instances.

        Returns:
            Deduplicated list of examples.
        """
        logger.info("Deduplicating examples...")
        seen_hashes = set()
        unique_examples = []

        for example in examples:
            hash_val = DataDeduplicator.compute_hash(example)
            if hash_val not in seen_hashes:
                seen_hashes.add(hash_val)
                unique_examples.append(example)

        logger.info(
            f"Deduplication: {len(examples)} -> {len(unique_examples)} examples"
        )
        return unique_examples


class DataProcessor:
    """Processes and prepares datasets for training."""

    @staticmethod
    def process_examples(
        examples: List[SQLExample],
        validate: bool = True,
    ) -> List[SQLExample]:
        """
        Process examples: validate, normalize, and filter.

        Args:
            examples: List of SQLExample instances.
            validate: Whether to validate SQL queries.

        Returns:
            Processed list of examples.
        """
        logger.info("Processing examples...")
        processed = []
        validator = SQLValidator()

        for example in tqdm(examples, desc="Processing"):
            # Skip if context or question is empty
            if not example.context or not example.question:
                continue

            # Validate SQL
            if validate and not validator.validate_sql(example.sql):
                continue

            # Normalize SQL
            example.sql = validator.normalize_sql(example.sql)

            # Clean whitespace
            example.context = example.context.strip()
            example.question = example.question.strip()
            example.sql = example.sql.strip()

            processed.append(example)

        logger.info(f"Processing: {len(examples)} -> {len(processed)} examples")
        return processed

    @staticmethod
    def split_dataset(
        examples: List[SQLExample],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> dict:
        """
        Split examples into train/val/test sets.

        Args:
            examples: List of SQLExample instances.
            train_ratio: Proportion for training (default 0.8).
            val_ratio: Proportion for validation (default 0.1).

        Returns:
            Dictionary with 'train', 'validation', 'test' splits.
        """
        import random

        random.shuffle(examples)

        n = len(examples)
        train_split = int(n * train_ratio)
        val_split = int(n * (train_ratio + val_ratio))

        return {
            "train": examples[:train_split],
            "validation": examples[train_split:val_split],
            "test": examples[val_split:],
        }

    @staticmethod
    def to_huggingface_dataset(examples: List[SQLExample]) -> Dataset:
        """
        Convert examples to HuggingFace Dataset.

        Args:
            examples: List of SQLExample instances.

        Returns:
            HuggingFace Dataset instance.
        """
        data = {
            "context": [ex.context for ex in examples],
            "question": [ex.question for ex in examples],
            "sql": [ex.sql for ex in examples],
            "source": [ex.source for ex in examples],
        }
        return Dataset.from_dict(data)


def prepare_and_push(
    output_dataset_name: str = "sql-agent/sql-training-unified",
    push_to_hub: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    """
    Main function to prepare and optionally push dataset to Hub.

    Args:
        output_dataset_name: Name of output dataset on HF Hub.
        push_to_hub: Whether to push to HuggingFace Hub.
        hf_token: HuggingFace API token for pushing.
    """
    logger.info("=" * 80)
    logger.info("SQL Dataset Preparation Pipeline")
    logger.info("=" * 80)

    # Load all datasets
    logger.info("\nStep 1: Loading datasets...")
    all_examples = DatasetLoader.load_all()
    logger.info(f"Total examples: {len(all_examples)}")

    # Process examples
    logger.info("\nStep 2: Processing examples...")
    processed_examples = DataProcessor.process_examples(all_examples, validate=True)

    # Deduplicate
    logger.info("\nStep 3: Deduplicating...")
    unique_examples = DataDeduplicator.deduplicate(processed_examples)

    # Split dataset
    logger.info("\nStep 4: Splitting dataset...")
    splits = DataProcessor.split_dataset(
        unique_examples,
        train_ratio=0.8,
        val_ratio=0.1,
    )

    logger.info(f"Train: {len(splits['train'])} examples")
    logger.info(f"Validation: {len(splits['validation'])} examples")
    logger.info(f"Test: {len(splits['test'])} examples")

    # Convert to HuggingFace datasets
    logger.info("\nStep 5: Converting to HuggingFace format...")
    dataset_dict = DatasetDict({
        "train": DataProcessor.to_huggingface_dataset(splits["train"]),
        "validation": DataProcessor.to_huggingface_dataset(splits["validation"]),
        "test": DataProcessor.to_huggingface_dataset(splits["test"]),
    })

    # Push to Hub if requested
    if push_to_hub:
        logger.info(f"\nStep 6: Pushing to HuggingFace Hub ({output_dataset_name})...")
        try:
            dataset_dict.push_to_hub(
                output_dataset_name,
                token=hf_token,
                private=True,
            )
            logger.info("Successfully pushed to Hub!")
        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")
    else:
        logger.info("\nStep 6: Skipping Hub push (use --push-to-hub flag to upload)")

    # Save locally
    logger.info("\nStep 7: Saving locally...")
    output_dir = "data/sql-unified"
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    logger.info(f"Dataset saved to {output_dir}")

    # Print statistics
    logger.info("\n" + "=" * 80)
    logger.info("Dataset Statistics")
    logger.info("=" * 80)
    for split_name, dataset in dataset_dict.items():
        sources = defaultdict(int)
        for source in dataset["source"]:
            sources[source] += 1
        logger.info(f"\n{split_name.upper()}:")
        logger.info(f"  Total examples: {len(dataset)}")
        logger.info("  By source:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    - {source}: {count}")


if __name__ == "__main__":
    prepare_and_push(push_to_hub=False)

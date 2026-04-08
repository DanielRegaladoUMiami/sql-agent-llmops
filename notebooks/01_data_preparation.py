"""
Notebook: Data Preparation for SQL Agent Training

This notebook walks through the process of preparing SQL training data from
multiple sources, including validation, deduplication, and format unification.
"""

# %% [markdown]
# # Data Preparation for SQL Agent Training
#
# This notebook covers:
# 1. Loading datasets from multiple sources
# 2. Validating SQL queries with sqlglot
# 3. Deduplicating examples
# 4. Unifying format across datasets
# 5. Splitting into train/val/test sets

# %%
import sys
sys.path.insert(0, '..')

from training.sql_generator.prepare_data import (
    DatasetLoader,
    SQLValidator,
    DataDeduplicator,
    DataProcessor,
)

# %% [markdown]
# ## Step 1: Load Datasets from Multiple Sources

# %%
print("Loading datasets from multiple sources...")
all_examples = DatasetLoader.load_all()
print(f"Total examples loaded: {len(all_examples)}")

# Show distribution by source
from collections import Counter
source_counts = Counter([ex.source for ex in all_examples])
print("\nExamples by source:")
for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {source}: {count}")

# %% [markdown]
# ## Step 2: Validate SQL Queries

# %%
validator = SQLValidator()
print("Validating SQL queries...")

valid_count = 0
invalid_count = 0
for example in all_examples[:100]:  # Check first 100
    if validator.validate_sql(example.sql):
        valid_count += 1
    else:
        invalid_count += 1

print(f"Valid SQL: {valid_count}")
print(f"Invalid SQL: {invalid_count}")

# %% [markdown]
# ## Step 3: Process and Filter Examples

# %%
print("Processing examples...")
processed = DataProcessor.process_examples(all_examples, validate=True)
print(f"After processing: {len(processed)} examples")

# %% [markdown]
# ## Step 4: Deduplicate Examples

# %%
print("Deduplicating examples...")
unique = DataDeduplicator.deduplicate(processed)
print(f"After deduplication: {len(unique)} examples")

# %% [markdown]
# ## Step 5: Split into Train/Val/Test

# %%
splits = DataProcessor.split_dataset(unique)
print(f"Train: {len(splits['train'])} examples")
print(f"Validation: {len(splits['validation'])} examples")
print(f"Test: {len(splits['test'])} examples")

# %% [markdown]
# ## Step 6: Convert to HuggingFace Format and Save

# %%
from datasets import DatasetDict

# Convert to HF datasets
dataset_dict = DatasetDict({
    'train': DataProcessor.to_huggingface_dataset(splits['train']),
    'validation': DataProcessor.to_huggingface_dataset(splits['validation']),
    'test': DataProcessor.to_huggingface_dataset(splits['test']),
})

# Save locally
output_dir = "../data/sql-unified"
dataset_dict.save_to_disk(output_dir)
print(f"Saved dataset to {output_dir}")

# %% [markdown]
# ## Step 7: Dataset Statistics and Analysis

# %%
print("Dataset Statistics:")
print(f"Train: {len(dataset_dict['train'])} examples")
print(f"Validation: {len(dataset_dict['validation'])} examples")
print(f"Test: {len(dataset_dict['test'])} examples")

# Show sample examples
print("\nSample training examples:")
for i in range(2):
    example = dataset_dict['train'][i]
    print(f"\nExample {i+1}:")
    print(f"Context: {example['context'][:100]}...")
    print(f"Question: {example['question']}")
    print(f"SQL: {example['sql']}")
    print(f"Source: {example['source']}")

print("\n✓ Data preparation completed successfully!")

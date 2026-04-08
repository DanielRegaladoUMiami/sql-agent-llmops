"""
Notebook: Training Chart Reasoner (Phi-3 Mini 3.8B)

Fine-tunes Phi-3 Mini 3.8B for chart configuration recommendation
using Unsloth + QLoRA.
"""

# %% [markdown]
# # Training Chart Reasoner
#
# This notebook demonstrates fine-tuning Phi-3 Mini 3.8B for chart recommendation
# using knowledge distillation from larger models.

# %%
import sys
sys.path.insert(0, '..')

import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## Step 1: Load Model and Tokenizer

# %%
from training.chart_reasoner.train import load_model_and_tokenizer

print("Loading Phi-3 Mini model...")
model, tokenizer = load_model_and_tokenizer()
print(f"Model loaded: {model.__class__.__name__}")

# %% [markdown]
# ## Step 2: Generate Training Data (Knowledge Distillation)

# %%
from training.chart_reasoner.generate_dataset import DatasetGenerator

print("Generating chart reasoning training data...")
generator = DatasetGenerator(hf_token=None)

# Load SQL examples
from datasets import load_dataset
try:
    sql_dataset = load_dataset('sql-agent/sql-training-unified', split='train')
except:
    print("Using synthetic SQL examples...")
    sql_dataset = load_dataset({
        'sql': ['SELECT * FROM sales GROUP BY product'],
        'context': ['CREATE TABLE sales (product VARCHAR, amount INT)']
    })

# Generate chart reasoning examples
examples = generator.generate_from_sql_examples(
    sql_dataset,
    num_examples=500,
    use_rule_based=True,
    use_api=False,
)

print(f"Generated {len(examples)} training examples")

# %% [markdown]
# ## Step 3: Load and Prepare Dataset

# %%
from training.chart_reasoner.train import load_training_data, format_prompt
from datasets import Dataset

# Create dataset
data = {
    'sql': [ex.sql for ex in examples],
    'data_preview': [ex.data_preview for ex in examples],
    'column_info': [ex.column_info for ex in examples],
    'chart_config': [ex.chart_config for ex in examples],
}

dataset = Dataset.from_dict(data)
dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

print(f"Training dataset size: {len(dataset)}")
print("\nSample training example:")
print(dataset[0]['text'][:400])

# %% [markdown]
# ## Step 4: Configure and Run Training

# %%
import yaml
from transformers import TrainingArguments
from trl import SFTTrainer

with open('../configs/training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    chart_config = config.get('chart_reasoner', {})

training_args = TrainingArguments(
    output_dir='../models/chart_reasoner',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=50,
    logging_steps=10,
    save_steps=50,
    save_total_limit=3,
    optim='paged_adamw_32bit',
    seed=42,
    fp16=True,
    lr_scheduler_type='cosine',
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    packing=False,
    max_seq_length=2048,
)

print("Starting training...")
train_result = trainer.train()
print(f"Training completed! Final loss: {train_result.training_loss:.4f}")

# %% [markdown]
# ## Step 5: Save Model

# %%
import os

output_path = training_args.output_dir
os.makedirs(output_path, exist_ok=True)

print(f"Saving model to {output_path}")
trainer.model.save_pretrained(os.path.join(output_path, 'final'))
tokenizer.save_pretrained(os.path.join(output_path, 'final'))

print("✓ Model saved!")

# %% [markdown]
# ## Step 6: Test Inference

# %%
from transformers import pipeline
import json

pipe = pipeline(
    'text-generation',
    model=os.path.join(output_path, 'final'),
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

test_prompt = """You are a data visualization expert. Given SQL query results, recommend the most appropriate chart type and configuration.

SQL Query:
SELECT product, SUM(sales) FROM sales GROUP BY product;

Query Results:
Product A: 15000, Product B: 22000, Product C: 18000

Column Info:
- product: categorical
- total_sales: numeric

Recommended Chart Configuration:"""

print("Testing inference...")
outputs = pipe(test_prompt, max_length=200, do_sample=False)
generated = outputs[0]['generated_text'][len(test_prompt):]
print("\nGenerated Chart Config:")
print(generated)

# %% [markdown]
# ## Training Complete!

# %%
print("\n" + "="*50)
print("Chart Reasoner Training Summary")
print("="*50)
print(f"Model: Phi-3 Mini 3.8B")
print(f"Training Examples: {len(dataset)}")
print(f"Final Loss: {train_result.training_loss:.4f}")
print(f"Saved to: {output_path}")
print("="*50)

"""
Notebook: Training SVG Renderer (DeepSeek Coder 1.3B)

Fine-tunes DeepSeek Coder 1.3B for SVG code generation
using programmatically generated training data.
"""

# %% [markdown]
# # Training SVG Renderer
#
# This notebook demonstrates fine-tuning DeepSeek Coder 1.3B for SVG generation
# from chart configurations.

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
from training.svg_renderer.train import load_model_and_tokenizer

print("Loading DeepSeek Coder 1.3B model...")
model, tokenizer = load_model_and_tokenizer()
print(f"Model loaded: {model.__class__.__name__}")

# %% [markdown]
# ## Step 2: Generate SVG Training Data

# %%
from training.svg_renderer.generate_dataset import DatasetGenerator

print("Generating SVG training data...")
examples = DatasetGenerator.generate_examples(num_examples=200)

print(f"Generated {len(examples)} SVG training examples")

if examples:
    print("\nSample example:")
    ex = examples[0]
    print(f"Chart config: {ex['chart_config']}")
    print(f"SVG code length: {len(ex['svg_code'])} characters")

# %% [markdown]
# ## Step 3: Prepare Training Dataset

# %%
from training.svg_renderer.train import format_prompt
from datasets import Dataset

data = {
    'chart_config': [ex['chart_config'] for ex in examples],
    'chart_data': [ex['chart_data'] for ex in examples],
    'svg_code': [ex['svg_code'] for ex in examples],
}

dataset = Dataset.from_dict(data)
dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

print(f"Training dataset size: {len(dataset)}")
print("\nSample training prompt:")
print(dataset[0]['text'][:500])

# %% [markdown]
# ## Step 4: Configure Training

# %%
import yaml
from transformers import TrainingArguments
from trl import SFTTrainer

with open('../configs/training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    svg_config = config.get('svg_renderer', {})

training_args = TrainingArguments(
    output_dir='../models/svg_renderer',
    num_train_epochs=3,
    per_device_train_batch_size=8,
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

# %% [markdown]
# ## Step 5: Train Model

# %%
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
# ## Step 6: Save Model

# %%
import os

output_path = training_args.output_dir
os.makedirs(output_path, exist_ok=True)

print(f"Saving model to {output_path}")
trainer.model.save_pretrained(os.path.join(output_path, 'final'))
tokenizer.save_pretrained(os.path.join(output_path, 'final'))

print("✓ Model saved!")

# %% [markdown]
# ## Step 7: Test SVG Generation

# %%
from transformers import pipeline
import json

pipe = pipeline(
    'text-generation',
    model=os.path.join(output_path, 'final'),
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

test_config = {
    'type': 'bar',
    'title': 'Sales by Product',
    'x_axis': 'product',
    'y_axis': 'sales'
}

test_prompt = f"""You are an SVG expert. Generate optimized SVG code for the given chart configuration.

Chart Configuration:
{json.dumps(test_config)}

Chart Data:
[{{"product": "A", "sales": 100}}, {{"product": "B", "sales": 150}}]

Generated SVG:
<svg"""

print("Testing SVG generation...")
outputs = pipe(test_prompt, max_length=300, do_sample=False)
generated = outputs[0]['generated_text'][len(test_prompt):]
print("\nGenerated SVG:")
print(f"<svg{generated[:200]}...")

# %% [markdown]
# ## Summary

# %%
print("\n" + "="*50)
print("SVG Renderer Training Summary")
print("="*50)
print(f"Model: DeepSeek Coder 1.3B")
print(f"Training Examples: {len(dataset)}")
print(f"Chart Types: bar, line, scatter, pie, histogram")
print(f"Final Loss: {train_result.training_loss:.4f}")
print(f"Saved to: {output_path}")
print("="*50)
print("\n✓ SVG Renderer training completed!")

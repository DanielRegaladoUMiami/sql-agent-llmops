"""
Notebook: Training SQL Generator (Qwen 2.5 Coder 7B)

Fine-tunes Qwen 2.5 Coder 7B on unified SQL dataset using Unsloth + QLoRA.
Optimized for Google Colab T4 GPU.
"""

# %% [markdown]
# # Training SQL Generator
#
# This notebook demonstrates fine-tuning Qwen 2.5 Coder 7B for SQL generation
# using parameter-efficient LoRA adapters and 4-bit quantization.

# %%
import sys
sys.path.insert(0, '..')

import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# %% [markdown]
# ## Step 1: Load Model and Tokenizer

# %%
print("Loading model and tokenizer...")
from training.sql_generator.train import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer()
print(f"Model loaded: {model.__class__.__name__}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# %% [markdown]
# ## Step 2: Load and Prepare Training Data

# %%
print("Loading training data...")
from training.sql_generator.train import load_training_data

train_dataset = load_training_data(split='train', max_samples=1000)
print(f"Training examples: {len(train_dataset)}")

# Show sample
print("\nSample training example:")
print(train_dataset[0]['text'][:500])

# %% [markdown]
# ## Step 3: Configure Training Arguments

# %%
import yaml
from transformers import TrainingArguments

# Load config
with open('../configs/training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    sql_config = config.get('sql_generator', {})

training_args = TrainingArguments(
    output_dir='../models/sql_generator',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=10,
    save_steps=50,
    save_total_limit=3,
    optim='paged_adamw_32bit',
    seed=42,
    fp16=True,
    lr_scheduler_type='cosine',
)

print("Training Configuration:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch Size: {training_args.per_device_train_batch_size}")
print(f"  Learning Rate: {training_args.learning_rate}")
print(f"  Output Directory: {training_args.output_dir}")

# %% [markdown]
# ## Step 4: Create Trainer and Train

# %%
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=training_args,
    packing=False,
    max_seq_length=2048,
)

print("Starting training...")
train_result = trainer.train()

print(f"\nTraining completed!")
print(f"Final loss: {train_result.training_loss:.4f}")

# %% [markdown]
# ## Step 5: Save Model

# %%
import os

output_path = training_args.output_dir
os.makedirs(output_path, exist_ok=True)

print(f"Saving model to {output_path}")
trainer.model.save_pretrained(os.path.join(output_path, 'final'))
tokenizer.save_pretrained(os.path.join(output_path, 'final'))

print("✓ Model saved successfully!")

# %% [markdown]
# ## Step 6: Test Inference

# %%
from transformers import pipeline

# Create text generation pipeline
pipe = pipeline(
    'text-generation',
    model=os.path.join(output_path, 'final'),
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

# Test prompt
test_prompt = """You are a SQL expert. Given the database schema and a natural language question, generate the correct SQL query.

Database Schema:
CREATE TABLE users (id INT, name VARCHAR(100), email VARCHAR(100), created_at DATE);

Question: What are all users created in 2024?

SQL Query:"""

print("Testing inference...")
outputs = pipe(test_prompt, max_length=256, do_sample=False)
print("\nGenerated SQL:")
print(outputs[0]['generated_text'][len(test_prompt):])

# %% [markdown]
# ## Step 7: Metrics Summary

# %%
print("\n" + "="*50)
print("Training Summary")
print("="*50)
print(f"Model: Qwen 2.5 Coder 7B")
print(f"Training Samples: {len(train_dataset)}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch Size: {training_args.per_device_train_batch_size}")
print(f"Final Loss: {train_result.training_loss:.4f}")
print(f"Saved to: {output_path}")
print("="*50)

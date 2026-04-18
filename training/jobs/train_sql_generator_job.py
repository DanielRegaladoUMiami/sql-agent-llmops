# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "unsloth_zoo",
#     "trl>=0.12",
#     "peft",
#     "accelerate",
#     "bitsandbytes",
#     "datasets",
#     "xformers",
#     "huggingface_hub",
# ]
# ///
"""
HF Jobs training script for the SQL Generator (Qwen 2.5 Coder 7B).

Launches via:
    hf jobs uv run --flavor a100-large --timeout 15h \
        --secrets HF_TOKEN \
        training/jobs/train_sql_generator_job.py

    # Test run:
    hf jobs uv run --flavor t4-small --timeout 30m \
        --secrets HF_TOKEN \
        training/jobs/train_sql_generator_job.py --sample 100

Environment:
    HF_TOKEN  — HuggingFace token with write scope (passed via --secrets)

Output:
    LoRA adapter pushed to:
    https://huggingface.co/DanielRegaladoCardoso/sql-generator-qwen25-coder-7b-lora
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train-sql")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_USER = "DanielRegaladoCardoso"
OUTPUT_REPO = f"{HF_USER}/sql-generator-qwen25-coder-7b-lora"
DATASET_REPO = f"{HF_USER}/text-to-sql-mix-v2"

PRIMARY_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
FALLBACK_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a SQL schema and a natural-language "
    "question, generate a correct SQL query answering the question. "
    "Return only the SQL."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None,
                   help="Subsample N rows from train (default: use all)")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Per-device batch size (auto-detected if omitted)")
    p.add_argument("--grad-accum", type=int, default=None,
                   help="Gradient accumulation steps (auto-detected if omitted)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--output-repo", default=OUTPUT_REPO)
    p.add_argument("--no-push", action="store_true")
    return p.parse_args()


def detect_gpu():
    """Detect GPU and return optimal batch_size, grad_accum."""
    import torch
    if not torch.cuda.is_available():
        log.error("No GPU detected! Available devices: CPU only")
        log.error("Make sure you launched with a GPU flavor (e.g. --flavor a100-large)")
        sys.exit(1)

    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    log.info(f"GPU: {name} ({vram_gb:.1f} GB VRAM)")
    log.info(f"CUDA version: {torch.version.cuda}")
    log.info(f"PyTorch version: {torch.__version__}")

    if vram_gb >= 120:      # H200 (141GB)
        return 16, 4, name
    elif vram_gb >= 70:     # A100 80GB
        return 8, 4, name
    elif vram_gb >= 40:     # A100 40GB
        return 4, 4, name
    elif vram_gb >= 20:     # L4 24GB / A10G
        return 2, 8, name
    else:                   # T4 (16GB)
        return 2, 8, name


def main():
    args = parse_args()
    start_time = time.time()

    # ----- HF Auth -----
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        log.info("HF_TOKEN found in environment, logging in...")
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
    else:
        log.warning("HF_TOKEN not set! Push to hub will fail. "
                     "Use --secrets HF_TOKEN when launching.")

    import torch
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    # ----- GPU detection -----
    auto_bs, auto_ga, gpu_name = detect_gpu()
    batch_size = args.batch_size or auto_bs
    grad_accum = args.grad_accum or auto_ga
    effective_batch = batch_size * grad_accum
    log.info(f"Training config: batch_size={batch_size} grad_accum={grad_accum} "
             f"effective={effective_batch}")

    # ----- Load model -----
    log.info("Loading model...")
    try:
        log.info(f"  Trying {PRIMARY_MODEL}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=PRIMARY_MODEL,
            max_seq_length=args.max_seq_len,
            load_in_4bit=True,
            dtype=None,
        )
        log.info("  Primary model loaded OK")
    except Exception as e:
        log.warning(f"  Primary failed: {e}")
        log.info(f"  Trying fallback: {FALLBACK_MODEL}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=FALLBACK_MODEL,
            max_seq_length=args.max_seq_len,
            load_in_4bit=True,
            dtype=None,
        )
        log.info("  Fallback model loaded OK")

    # ----- LoRA -----
    log.info(f"Attaching LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ----- Dataset -----
    log.info(f"Loading dataset from {DATASET_REPO}")
    ds = load_dataset(DATASET_REPO)

    train_raw = ds["train"]
    if args.sample:
        train_raw = train_raw.shuffle(seed=42).select(
            range(min(args.sample, len(train_raw)))
        )
        log.info(f"  Subsampled to {len(train_raw):,} rows")
    else:
        log.info(f"  Using full train set: {len(train_raw):,} rows")

    def to_chat(row):
        user_content = (
            f"### Schema\n{row['schema_context']}\n\n"
            f"### Question\n{row['instruction']}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": row["sql"]},
        ]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )}

    n_proc = min(4, os.cpu_count() or 1)
    log.info(f"Formatting dataset (num_proc={n_proc})...")
    train_ds = train_raw.map(to_chat, remove_columns=train_raw.column_names,
                              num_proc=n_proc, desc="format-train")
    eval_ds = ds["validation"].map(to_chat, remove_columns=ds["validation"].column_names,
                                    num_proc=n_proc, desc="format-eval")

    # ----- Filter by seq length -----
    def fits(row):
        return len(tokenizer.encode(row["text"], add_special_tokens=False)) <= args.max_seq_len

    before = len(train_ds)
    train_ds = train_ds.filter(fits, num_proc=n_proc, desc="filter-train")
    eval_ds = eval_ds.filter(fits, num_proc=n_proc, desc="filter-eval")
    log.info(f"Seq-length filter: kept {len(train_ds):,} / {before:,} "
             f"({100*len(train_ds)/max(1,before):.1f}%)")

    n_steps = len(train_ds) // effective_batch
    log.info(f"Total steps: {n_steps:,} (dataset={len(train_ds):,} / "
             f"effective_batch={effective_batch})")
    log.info(f"=== ESTIMATED TIME: check it/s after first 10 steps ===")

    # ----- Training -----
    training_args = SFTConfig(
        output_dir="sql_generator_adapter",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
        eval_strategy="no",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=training_args,
    )

    log.info("Starting training...")
    try:
        stats = trainer.train()
        log.info(f"Training complete: {stats}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log.error(f"OOM! Try reducing --batch-size. Current: {batch_size}. "
                       f"Suggested: {max(1, batch_size // 2)}")
            log.error(f"Re-run with: --batch-size {max(1, batch_size // 2)}")
            torch.cuda.empty_cache()
        raise

    elapsed = (time.time() - start_time) / 3600
    log.info(f"Total wall time: {elapsed:.1f} hours")

    # ----- Push to Hub -----
    if not args.no_push:
        log.info(f"Pushing LoRA adapter to {args.output_repo}")
        try:
            model.push_to_hub(args.output_repo, save_method="lora")
            tokenizer.push_to_hub(args.output_repo)
        except Exception as e1:
            log.warning(f"push_to_hub failed ({e1}), trying push_to_hub_merged...")
            try:
                model.push_to_hub_merged(
                    args.output_repo, tokenizer, save_method="lora"
                )
            except Exception as e2:
                log.error(f"push_to_hub_merged also failed: {e2}")
                log.info("Saving locally as fallback...")
                model.save_pretrained("sql_generator_adapter_final")
                tokenizer.save_pretrained("sql_generator_adapter_final")
                log.info("Saved to sql_generator_adapter_final/")
                raise
        log.info(f"Done: https://huggingface.co/{args.output_repo}")
    else:
        log.info("Skipping push (--no-push)")

    # ----- Quick test -----
    log.info("Running inference test...")
    FastLanguageModel.for_inference(model)
    sample = ds["test"][0]
    user_content = (
        f"### Schema\n{sample['schema_context']}\n\n"
        f"### Question\n{sample['instruction']}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    out = model.generate(
        input_ids, max_new_tokens=400, do_sample=False, temperature=0
    )
    generated = tokenizer.decode(
        out[0][input_ids.shape[1]:], skip_special_tokens=True
    )
    log.info(f"Generated SQL: {generated[:300]}")
    log.info(f"Gold SQL:      {sample['sql'][:300]}")

    total_time = (time.time() - start_time) / 3600
    log.info(f"=== ALL DONE. Total time: {total_time:.1f} hours ===")


if __name__ == "__main__":
    main()

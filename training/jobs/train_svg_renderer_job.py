# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
#     "unsloth_zoo",
#     "trl>=0.12",
#     "peft",
#     "accelerate",
#     "bitsandbytes",
#     "datasets",
#     "xformers",
#     "huggingface_hub",
#     "torch",
# ]
# ///
"""
HF Jobs training script for the SVG Renderer (DeepSeek Coder 1.3B).

    hf jobs uv run --flavor t4-small --timeout 4h \
        --secrets HF_TOKEN \
        training/jobs/train_svg_renderer_job.py
"""

from __future__ import annotations
import argparse, json, logging, os, sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train-svg")

HF_USER = "DanielRegaladoCardoso"
OUTPUT_REPO = f"{HF_USER}/svg-renderer-deepseek-coder-1.3b-lora"
DATASET_REPO = f"{HF_USER}/svg-chart-render-v1"
PRIMARY_MODEL = "unsloth/deepseek-coder-1.3b-instruct-bnb-4bit"
FALLBACK_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"

SYSTEM_PROMPT = (
    "You are an expert data-visualization engineer. "
    "Given a chart specification as JSON, produce a clean, valid inline SVG."
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--output-repo", default=OUTPUT_REPO)
    p.add_argument("--no-push", action="store_true")
    return p.parse_args()

def detect_gpu():
    import torch
    if not torch.cuda.is_available():
        log.error("No GPU!"); sys.exit(1)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    name = torch.cuda.get_device_name(0)
    log.info(f"GPU: {name} ({vram:.1f} GB)")
    if vram >= 40:    return 8, 2, name
    elif vram >= 20:  return 4, 4, name
    else:             return 2, 8, name

def main():
    args = parse_args()
    import torch
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    auto_bs, auto_ga, gpu_name = detect_gpu()
    batch_size = args.batch_size or auto_bs
    grad_accum = args.grad_accum or auto_ga
    log.info(f"Config: bs={batch_size} ga={grad_accum} eff={batch_size*grad_accum}")

    log.info("Loading model...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=PRIMARY_MODEL, max_seq_length=args.max_seq_len,
            load_in_4bit=True, dtype=None)
    except Exception:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=FALLBACK_MODEL, max_seq_length=args.max_seq_len,
            load_in_4bit=True, dtype=None)

    model = FastLanguageModel.get_peft_model(model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42)

    log.info(f"Loading dataset {DATASET_REPO}")
    ds = load_dataset(DATASET_REPO)
    train_raw = ds["train"]
    if args.sample:
        train_raw = train_raw.shuffle(seed=42).select(range(min(args.sample, len(train_raw))))

    def to_chat(row):
        spec = row["chart_spec"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Render this chart spec as SVG:\n\n{spec}"},
            {"role": "assistant", "content": row["svg_code"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    train_ds = train_raw.map(to_chat, remove_columns=train_raw.column_names, num_proc=4)
    eval_ds = ds["validation"].map(to_chat, remove_columns=ds["validation"].column_names, num_proc=2)

    def fits(row):
        return len(tokenizer.encode(row["text"], add_special_tokens=False)) <= args.max_seq_len
    before = len(train_ds)
    train_ds = train_ds.filter(fits, num_proc=4)
    eval_ds = eval_ds.filter(fits, num_proc=2)
    log.info(f"Filter: {len(train_ds):,}/{before:,} kept")

    trainer = SFTTrainer(model=model, tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds.select(range(min(200, len(eval_ds)))),
        args=SFTConfig(
            output_dir="svg_renderer_adapter",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_ratio=0.03, num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=25, save_steps=args.save_steps, save_total_limit=2,
            optim="adamw_8bit", lr_scheduler_type="cosine", seed=42,
            report_to="none", dataset_text_field="text",
            max_seq_length=args.max_seq_len, packing=False, eval_strategy="no",
        ))

    log.info("Training...")
    trainer.train()

    if not args.no_push:
        log.info(f"Pushing to {args.output_repo}")
        model.push_to_hub(args.output_repo, save_method="lora")
        tokenizer.push_to_hub(args.output_repo)
        log.info(f"Done: https://huggingface.co/{args.output_repo}")

    log.info("All done.")

if __name__ == "__main__":
    main()

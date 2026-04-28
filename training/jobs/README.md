# HF Jobs Training Scripts

UV-executable training scripts that run on [HuggingFace Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs).
Same training logic as the Colab notebooks, packaged as standalone Python scripts with inline dependencies (PEP 723 format).

## Why HF Jobs over Colab

- No session timeouts (Colab disconnects after ~12-24h)
- Pay per actual hour used, not monthly subscription
- Launch from terminal, no browser tab to babysit
- Auto GPU-detection: scripts auto-tune batch size for the assigned GPU
- Hub checkpointing every N steps means lost jobs can be resumed

## Quick reference

| Model | Script | Recommended GPU | Time | Cost |
|---|---|---|---|---|
| SQL Generator (7B) | `train_sql_generator_job.py` | `l40sx1` | 13.5h | ~$24 |
| Chart Reasoner (3.8B) | `train_chart_reasoner_job.py` | `a10g-large` or `t4-small` | 2-4h | $3-6 |
| SVG Renderer (1.3B) | `train_svg_renderer_job.py` | `t4-small` | 1-2h | $1-2 |

> **GPU note**: H200 (`h200`) is *not* recommended despite being the fastest flavor — the default UV image has a CUDA initialization issue with Hopper that prevents Unsloth from detecting the GPU. Use L40S or A100 instead.

## Usage

```bash
# SQL Generator on L40S (full 720k dataset, optimized config)
hf jobs uv run --flavor l40sx1 --timeout 16h \
    --secrets HF_TOKEN=$(cat ~/.cache/huggingface/token) \
    training/jobs/train_sql_generator_job.py \
    --batch-size 16 --grad-accum 1 --save-steps 1000

# SQL Generator smoke test (100 examples on T4, ~10 min)
hf jobs uv run --flavor t4-small --timeout 30m \
    --secrets HF_TOKEN=$(cat ~/.cache/huggingface/token) \
    training/jobs/train_sql_generator_job.py --sample 100

# Resume a previous SQL training from latest Hub checkpoint
hf jobs uv run --flavor l40sx1 --timeout 8h \
    --secrets HF_TOKEN=$(cat ~/.cache/huggingface/token) \
    training/jobs/train_sql_generator_job.py --resume \
    --batch-size 16 --grad-accum 1 --save-steps 1000

# Chart Reasoner on A10G
hf jobs uv run --flavor a10g-large --timeout 6h \
    --secrets HF_TOKEN=$(cat ~/.cache/huggingface/token) \
    training/jobs/train_chart_reasoner_job.py

# SVG Renderer on T4
hf jobs uv run --flavor t4-small --timeout 4h \
    --secrets HF_TOKEN=$(cat ~/.cache/huggingface/token) \
    training/jobs/train_svg_renderer_job.py
```

## Monitoring

```bash
hf jobs ps                # list running jobs
hf jobs logs <job_id>     # stream logs
hf jobs inspect <job_id>  # detailed status
hf jobs cancel <job_id>   # stop a running job
```

## Script flags

All three scripts accept the same flags:

| Flag | Default | Description |
|---|---|---|
| `--sample N` | None (use all) | Subsample N rows for quick test |
| `--max-seq-len N` | 1024 | Max sequence length in tokens (filter drops longer rows) |
| `--batch-size N` | auto | Per-device batch size (auto-detected from GPU VRAM) |
| `--grad-accum N` | auto | Gradient accumulation steps (auto-detected) |
| `--epochs N` | 1 | Number of training epochs |
| `--lr FLOAT` | 1e-4 (SQL) / 2e-4 (others) | Learning rate |
| `--lora-r N` | 16 | LoRA rank |
| `--lora-alpha N` | 32 | LoRA alpha |
| `--save-steps N` | 500 | Save checkpoint every N steps and push to Hub |
| `--output-repo REPO` | auto | HuggingFace repo to push the adapter |
| `--no-push` | false | Skip pushing to Hub (useful for smoke tests) |
| `--resume` | false | Resume from latest Hub checkpoint |

## Optimizations baked into the SQL training script

After empirical tuning, the SQL training script uses:

| Optimization | Effect |
|---|---|
| `packing=True` (TRL) | Concatenates short sequences into 1024-token blocks → ~4.7x compression |
| `gradient_checkpointing=False` | L40S/A100 have plenty of VRAM, disabling saves ~30% per step |
| `max_seq_len=1024` (vs 2048) | Drops only 6.9% of examples, makes attention 2x faster |
| `optim="adamw_8bit"` | bitsandbytes 8-bit AdamW, lower memory pressure |
| `bf16=True` (auto on Ampere+) | Faster + more numerically stable than fp16 |
| `hub_strategy="every_save"` | Pushes checkpoint to Hub after each save_steps interval (no progress lost on cancellation) |

## Output

Each script pushes both a LoRA adapter and a merged 16-bit model to HuggingFace:

- [`DanielRegaladoCardoso/sql-generator-qwen25-coder-7b-lora`](https://huggingface.co/DanielRegaladoCardoso/sql-generator-qwen25-coder-7b-lora) — trained
- `DanielRegaladoCardoso/chart-reasoner-phi3-mini-lora` — planned
- `DanielRegaladoCardoso/svg-renderer-deepseek-coder-1.3b-lora` — planned

## Real run reference (SQL Generator)

The current trained SQL Generator was produced by this exact command:

```bash
hf jobs uv run --flavor l40sx1 --timeout 10h \
    --secrets HF_TOKEN=$(cat ~/.cache/huggingface/token) \
    training/jobs/train_sql_generator_job.py \
    --batch-size 16 --grad-accum 1 --save-steps 1000
```

Final stats from the actual job:

| Metric | Value |
|---|---|
| Total dataset rows | 723,097 |
| Rows kept after seq-len filter | 672,949 (93.1%) |
| Sequences after packing | 154,462 |
| Training steps | 9,654 |
| Wall-clock time | 13.5h |
| Final training loss | 0.2658 |
| GPU cost | ~$24 |

The merged 16-bit model and LoRA adapter are both in the [model repo](https://huggingface.co/DanielRegaladoCardoso/sql-generator-qwen25-coder-7b-lora).

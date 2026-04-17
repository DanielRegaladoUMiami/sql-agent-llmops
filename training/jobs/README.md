# HF Jobs Training Scripts

UV-executable training scripts that run on [HuggingFace Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs).
Same training logic as the Colab notebooks, packaged as standalone Python scripts with inline dependencies.

## Why HF Jobs over Colab?

- No session timeouts (Colab disconnects after ~12-24h)
- Access to H200 (141GB VRAM) — faster than Colab's A100 40GB
- Pay per actual hour used, not monthly subscription
- Launch from terminal — no browser tab to babysit
- Auto GPU-detection: the scripts auto-tune batch size for the assigned GPU

## Quick reference

| Model | Script | Recommended GPU | Est. time | Est. cost |
|-------|--------|-----------------|-----------|-----------|
| SQL Generator (7B) | `train_sql_generator_job.py` | `h200` or `a100-large` | 5-12h | $15-30 |
| Chart Reasoner (3.8B) | `train_chart_reasoner_job.py` | `a100-large` or `t4-small` | 2-4h | $5-10 |
| SVG Renderer (1.3B) | `train_svg_renderer_job.py` | `t4-small` | 1-2h | $1-2 |

## Usage

```bash
# SQL Generator on H200 (full 720k dataset)
hf jobs uv run --flavor h200 --timeout 10h \
    --secrets HF_TOKEN \
    training/jobs/train_sql_generator_job.py

# SQL Generator test run (10k sample on T4)
hf jobs uv run --flavor t4-small --timeout 2h \
    --secrets HF_TOKEN \
    training/jobs/train_sql_generator_job.py --sample 10000

# Chart Reasoner on A100
hf jobs uv run --flavor a100-large --timeout 6h \
    --secrets HF_TOKEN \
    training/jobs/train_chart_reasoner_job.py

# SVG Renderer on T4
hf jobs uv run --flavor t4-small --timeout 4h \
    --secrets HF_TOKEN \
    training/jobs/train_svg_renderer_job.py
```

## Monitoring

```bash
hf jobs ps                    # list running jobs
hf jobs logs <job_id>         # stream logs
hf jobs inspect <job_id>      # detailed status
hf jobs cancel <job_id>       # stop a running job
```

## Script flags

All three scripts accept the same flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--sample N` | None (use all) | Subsample N rows for quick test |
| `--max-seq-len N` | 2048 | Max sequence length in tokens |
| `--batch-size N` | auto | Per-device batch size (auto-detected from GPU VRAM) |
| `--grad-accum N` | auto | Gradient accumulation steps (auto-detected) |
| `--epochs N` | 1 | Number of training epochs |
| `--lr FLOAT` | 1e-4 (SQL) / 2e-4 (others) | Learning rate |
| `--save-steps N` | 500 | Save checkpoint every N steps |
| `--output-repo REPO` | auto | HuggingFace repo to push the adapter |
| `--no-push` | false | Skip pushing to HuggingFace |

## Output

Each script pushes a LoRA adapter to HuggingFace:

- `DanielRegaladoCardoso/sql-generator-qwen25-coder-7b-lora`
- `DanielRegaladoCardoso/chart-reasoner-phi3-mini-lora`
- `DanielRegaladoCardoso/svg-renderer-deepseek-coder-1.3b-lora`

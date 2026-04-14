# 📓 Training Notebooks

One Colab notebook per specialist model. Each one:

1. Installs [Unsloth](https://github.com/unslothai/unsloth)
2. Loads the base model in 4-bit
3. Attaches LoRA adapters
4. Loads the matching dataset from HuggingFace
5. Formats via `tokenizer.apply_chat_template` for SFT
6. Trains with `trl.SFTTrainer`
7. Pushes the LoRA adapter to your HuggingFace

| Notebook | Base model | Dataset | Hardware |
|----------|------------|---------|----------|
| [`train_sql_generator.ipynb`](train_sql_generator.ipynb) | Qwen 2.5 Coder 7B | [`DanielRegaladoCardoso/text-to-sql-mix-v2`](https://huggingface.co/datasets/DanielRegaladoCardoso/text-to-sql-mix-v2) | Colab Pro A100 (T4 for subsample) |
| [`train_chart_reasoner.ipynb`](train_chart_reasoner.ipynb) | Phi-3 Mini 3.8B | [`DanielRegaladoCardoso/chart-reasoning-mix-v1`](https://huggingface.co/datasets/DanielRegaladoCardoso/chart-reasoning-mix-v1) | Colab T4 free ✅ |
| [`train_svg_renderer.ipynb`](train_svg_renderer.ipynb) | DeepSeek Coder 1.3B | [`DanielRegaladoCardoso/svg-chart-render-v1`](https://huggingface.co/datasets/DanielRegaladoCardoso/svg-chart-render-v1) | Colab T4 free ✅ |

## Open in Colab

Click any notebook on GitHub and hit the **"Open in Colab"** badge, or use a URL like:

```
https://colab.research.google.com/github/DanielRegaladoUMiami/sql-agent-llmops/blob/main/training/notebooks/train_svg_renderer.ipynb
```

## What gets produced

Each notebook pushes a LoRA adapter to your HuggingFace profile:

- `DanielRegaladoCardoso/sql-generator-qwen25-coder-7b-lora`
- `DanielRegaladoCardoso/chart-reasoner-phi3-mini-lora`
- `DanielRegaladoCardoso/svg-renderer-deepseek-coder-1.3b-lora`

These get wired into the production SQL Agent in [`sql_agent/`](../../sql_agent/).

## Tips

- Always run cell 1 (`nvidia-smi`) first to confirm you got a GPU.
- T4 free tier has a time limit (~12 h). If you time out, the trainer checkpoints
  every 500–1000 steps — reload from `output_dir/checkpoint-*` and resume.
- For the SQL Generator on free tier, set `SAMPLE_N = 100_000` in cell 7 or you
  won't finish the epoch before the session expires.

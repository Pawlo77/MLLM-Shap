<div align="center">
  <h1>🔬 Experiments Hub</h1>
  <p><strong>Operational layer for reproducible SHAP research runs and analytics.</strong></p>
</div>

# Scope

This area owns the full experiment lifecycle:
- Dataset preparation and TTS synthesis
- Config-driven SHAP experiment execution
- Faithfulness validation (deletion tests)
- Post-run statistical analysis and plotting

# Directory Guide

```
experiments/
├── __init__.py              # Package marker
├── mllm_shapx/             # Main experiment runner (configs, CLI, SLURM scripts)
├── data_preparation/        # Dataset construction, TTS, HuggingFace upload
├── faithfulness/            # HP-1 deletion-based attribution validation
├── interspeech/             # Paper-specific plots, ablation, dashboard
├── analysis/                # Post-run metrics, significance tests, figures
├── ghost_busters/           # Process monitor / zombie-killer utilities
├── logs/                    # Historical run logs
└── wandb/                   # Weights & Biases run artifacts
```

# Standard Workflow

1. **Prepare data** — `data_preparation/` notebooks build and upload datasets.
2. **Configure** — Write JSON configs in `mllm_shapx/configs/`.
3. **Run** — Execute via CLI or SLURM:
   ```bash
   python -m experiments.mllm_shapx.src.cli run --config <path> --resume
   # or batch:
   ./experiments/mllm_shapx/run_configs.sh -c configs/package_grid
   ```
4. **Validate** — Run faithfulness deletion tests (`faithfulness/`).
5. **Analyse** — Build figures and tables in `analysis/` and `interspeech/`.

# Subpackage READMEs

Each subdirectory has its own README with detailed usage:
- [`mllm_shapx/README.md`](mllm_shapx/README.md)
- [`data_preparation/README.md`](data_preparation/README.md)
- [`faithfulness/README.md`](faithfulness/README.md)
- [`interspeech/README.md`](interspeech/README.md)
- [`ghost_busters/README.md`](ghost_busters/README.md)
- [`analysis/README.md`](analysis/README.md)

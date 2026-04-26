<div align="center">
  <h1>🔬 Experiments Hub</h1>
  <p><strong>Operational layer for reproducible SHAP research runs and analytics.</strong></p>
</div>

## 🎯 Scope

This area owns full experiment lifecycle:
- dataset preparation
- runner execution
- output aggregation
- post-run statistical analysis

## 📊 Operations Snapshot

- **1 orchestrator**: `mllm_shapx`
- **2 analysis levels**: per-sample + aggregate summaries
- **multiple dataset pipelines** under `data_preparation/`

## 🗂️ Directory Guide

- `analysis/` - metrics, plots, significance checks
- `data_preparation/` - dataset construction notebooks
- `mllm_shapx/` - config-driven experiment CLI
- `ghost_busters/` - runtime/process helper scripts
- `experiments_output/` - generated run artifacts

## ⚙️ Standard Workflow

1. Prepare data in `data_preparation/`.
2. Validate and run configs via `mllm_shapx`.
3. Inspect outputs in `experiments_output/`.
4. Build figures/tables in `analysis/`.

## 🧩 Environment Note

Loose `.py` and `.sh` files mainly support local cluster operations and ad-hoc execution flows.

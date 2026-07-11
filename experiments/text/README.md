# T2T Experiments

Text-to-text Shapley value experiments for the `mllm_shap` package.

## Documentation

- [scope.md](scope.md) — Full research scope and phased plan
- [plan.md](plan.md) — Experiment execution table, dataset assessment, prerequisites

## Quick Start

Ensure LM Studio is running on localhost with the MLX backend, then:

```bash
# Run all experiments (from repo root)
./experiments/mllm_shapx/run_configs.sh -c "experiments/text/configs/t2t_*.json"

# Run a single experiment group (e.g. T2T-01 only)
./experiments/mllm_shapx/run_configs.sh -c "experiments/text/configs/t2t_01_*.json"

# Dry run to preview what would execute
./experiments/mllm_shapx/run_configs.sh -c "experiments/text/configs/t2t_*.json" --dry-run

# Run with resume (skip completed samples)
./experiments/mllm_shapx/run_configs.sh -c "experiments/text/configs/t2t_*.json" --resume

# Run a specific model only
./experiments/mllm_shapx/run_configs.sh -c "experiments/text/configs/t2t_*_gemma3.json"
```

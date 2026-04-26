<div align="center">
  <h1>🚀 mllm_shapx Runner</h1>
  <p><strong>Configuration-first CLI for reproducible SHAP sweeps at research scale.</strong></p>
</div>

`mllm_shapx` runs reproducible SHAP experiments with resume support and optional Weights & Biases tracking.

## 📊 Runner Snapshot

- **Execution model**: config-driven variant expansion
- **Checkpointing**: built-in resume flow
- **Output granularity**: per-sample + aggregate metrics
- **Supported connectors**: `liquid_audio`, `hf_text`

## Prerequisites

- Python 3.12
- project dependencies installed (`make install`)
- import access to `mllm_shap` (in monorepo setup, configure `MLLM_SHAP_SRC` when needed)

Example environment:

```bash
export MLLM_SHAP_SRC=/path/to/mllm_shap/src
export LOG_LEVEL=INFO
```

See `.env.example` for defaults.

## CLI Entry Point

```bash
python -m mllm_shapx.cli
```

## Common Commands

Validate configuration:

```bash
python -m mllm_shapx.cli validate --config path/to/config.json --check-dataset
```

Run experiments:

```bash
python -m mllm_shapx.cli run --config path/to/config.json --resume
```

Use `--resume` to continue interrupted runs using checkpoints.

## Configuration

Start from templates in `configs/`, especially:

- `configs/mc_minimal.json`
- `configs/single_sentence_grid.json`

The runner supports:
- connector selection (`liquid_audio`, `hf_text`)
- dataset and row selection controls
- SHAP explainer sweeps (`exact`, MC variants, complementary, Neyman, hierarchical)
- W&B logging configuration

For field-level validation and constraints, use the `validate` command before running large jobs.

## Output Layout

Each resolved run writes to:

`{output_root}/{experiment_set_id}/{run_slug}/`

with:
- `spec.json` (resolved config)
- `checkpoint.json` (resume state)
- `samples/sample_XXXXX_result.json` (per-sample outputs)
- `summary/aggregate_metrics.json` (run summary)

## 🛠️ Extension Points

- Register new reducers/normalizers in `config.py`.
- Add new connector wiring in `factory.py`.
- Extend variant expansion/validation in `runner.py` and `config.py` for new explainer types.

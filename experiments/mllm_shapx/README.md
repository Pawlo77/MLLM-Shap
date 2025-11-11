# mllm_shapx — SHAP Experiments runner

A small, production-ready runner that executes **Precise** and **Monte-Carlo** SHAP experiments for `mllm_shap` on the *single_sentence* dataset shard (HuggingFace). It’s modular, documented, and easy to extend.

## Highlights

- ✅ Clean, layered architecture (config → data → factory → runner → storage/serialization)
- ✅ Robust I/O (per-sample results + aggregate summary + checkpoints)
- ✅ Workflows: `validate` (without running) and `run` (with resume)
- ✅ Optional Weights & Biases (metrics + artifacts)
- ✅ MC runs via **lists of `num_samples`** and/or **lists of `fractions`**

## Install

This package expects `mllm_shap` to be importable. In a mono-repo setup you can point at the local sources:

```bash
export MLLM_SHAP_SRC=/path/to/mllm_shap/src
export LOG_LEVEL=INFO
````

Install dependencies in your venv (minimal set):

```bash
uv sync
```

## Usage

Validate a config:

```bash
python -m mllm_shapx.cli validate --config path/to/config.json --check-dataset
```

Run experiments:

```bash
python -m mllm_shapx.cli run --config path/to/config.json [--resume]
```

W&B (optional):

```bash
export WANDB_API_KEY=...
# To log offline:
# export WANDB_MODE=offline  # or use config.wandb.mode
```

## Config (JSON)

```json
{
  "experiment_set_id": "ss_2025_11_03__mc_minimal",
  "output_root": "experiments_output",
  "device": null,
  "dataset": {
    "subset": "single_sentence",
    "split": "test",
    "revision": "refs/convert/parquet",
    "repo_id": "Pawlo77/mllm-shap"
  },
  "selection": {
    "max_samples": 10,
    "shuffle_seed": 0,
    "start_index": 0
  },
  "wandb": {
    "enabled": true,
    "project": "mllm-shap",
    "entity": null,
    "group": null,
    "mode": "offline",
    "tags": ["demo"]
  },
  "generation": {
    "max_new_tokens": 32,
    "text_temperature": 0.2
  },
  "shap": {
    "mode": "CONTEXTUAL",
    "normalizer": "AbsSumNormalizer",
    "reducer": "MeanReducer",
    "similarity": "CosineSimilarity"
  },
  "experiments": [
    { "explainer_type": "exact", "name": "exact" },
    {
      "explainer_type": "mc",
      "name": "mc_sweep",
      "num_samples": [32, 128, 512],
      "fractions": [0.1, 0.25, 0.5, 1.0]
    }
  ]
}
```

**Notes**

* `experiments[].explainer_type`: `"exact"` or `"mc"`.
* For **MC** you may provide any combination of:

  * `num_samples`: **list of ints** (each yields a run). Use `-1` if supported by your explainer semantics.
  * `fractions`: **list of floats** in `(0, 1]` (each yields a run).
* `device`: `null` (auto), `"cuda"` or `"cpu"`.

## Outputs

For each variant:

```
{output_root}/{experiment_set_id}/{run_slug}/
  spec.json
  checkpoint.json
  samples/sample_00000_result.json
  ...
  summary/aggregate_metrics.json
```

* `samples/*_result.json` — per-input conversation with SHAP attributions and per-sample metrics.
* `summary/aggregate_metrics.json` — aggregate stats over processed samples.
* If W&B is enabled:

  * metrics are logged per sample,
  * `aggregate_metrics.json` is uploaded as an Artifact.

## Extending

* **New reducers/normalizers:** register classes in `config.NORMALIZER_MAP` / `config.REDUCER_MAP`, then reference by name in config.
* **New explainer types:** add a branch in `factory.build_explainer_for_variant()` and map it in `runner.expand_variants()`.
* **Datasets:** replace `data.load_single_sentence_df()` with your loader (or add a new `DatasetConfig` + switch on `subset`).
* **Serialization:** tweak `serialization.serialize_conversation()` and `compute_modality_summary()`.

## Troubleshooting

* Set `LOG_LEVEL=DEBUG` to see more detail.
* Monorepo import errors: ensure `MLLM_SHAP_SRC` points to `mllm_shap/src`.
* Resuming issues: delete the run folder (or `checkpoint.json`) to start fresh.

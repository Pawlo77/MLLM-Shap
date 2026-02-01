# mllm_shapx — SHAP Experiments runner

Small, production-quality runner for precise and Monte‑Carlo SHAP experiments used by the
`mllm_shap` project. Focused on the `single_sentence` dataset shard (Hugging Face), the
package implements configuration-driven experiment sweeps, robust I/O, and optional
Weights & Biases integration for logging and artifacts.

**Intended audience:** researchers and engineers running reproducible SHAP experiments
with text/audio connectors supported by `mllm_shapx`.

**Contents:** setup, quick start, CLI usage, and a complete reference for every config option.

**Prerequisites:** Python 3.12+, a virtual environment, and `mllm_shap` importable (monorepo
add `MLLM_SHAP_SRC` to `PYTHONPATH`).

**Quick links**
- **CLI entrypoint:** `python -m mllm_shapx.cli`
- **Default configs:** `experiments/mllm_shapx/configs/`

**Get Started**

- Clone repository and create a virtualenv:

  ```bash
  git clone https://github.com/Pawlo77/MLLM-Shap.git
  uv sync
  ```

- Make `mllm_shap` importable in a monorepo layout:

  ```bash
  export MLLM_SHAP_SRC=/path/to/mllm_shap/src
  export LOG_LEVEL=INFO
  ```

  For default `.env` setup see [this example](./.env.example).

- Use one of the example configs in `configs/` (e.g. `mc_minimal.json`) as a starting point.

**CLI Usage**

- Validate a config (syntax + light semantic checks):

  ```bash
  uv run python -m mllm_shapx.cli validate --config path/to/config.json [--check-dataset]
  ```

- Run experiments defined in a config (sequentially expands variants):

  ```bash
  uv python -m mllm_shapx.cli run --config path/to/config.json [--resume]
  ```

- `--resume` will attempt to continue a partially completed run using existing
  `checkpoint.json` files in the run folder.

> Note: you can also chain runs via standard unix commands, e.g.:

```bash
uv python -m mllm_shapx.cli run --config configs/mc_minimal.json && \
uv python -m mllm_shapx.cli run --config configs/single_sentence_grid.json
```

**Outputs**

- For each variant the runner produces a run folder:

  - ` {output_root}/{experiment_set_id}/{run_slug}/spec.json` — resolved config for the run
  - `checkpoint.json` — checkpoint state for resume
  - `samples/sample_XXXXX_result.json` — per-sample result with attributions and metrics
  - `summary/aggregate_metrics.json` — aggregate results for the run

- If W&B is enabled the runner logs per-sample metrics and uploads `aggregate_metrics.json`
  as an artifact as well as `samples` directory via incremental artifact updates.

**Complete config reference**

The runner loads a JSON file and parses it into an `ExperimentSet` dataclass. Below are
every supported top-level and nested option, their meaning, accepted values, and defaults.

**Top-level**
- `experiment_set_id`: `string` — unique id used to name experiment output directory (required).
- `output_root`: `string` — base output directory (default: `"experiments_output"`).
- `device`: `string | null` — compute device. Allowed: `"cuda"`, `"cpu"`, or `null` (auto).
- `connector`: `string` — backend connector type. Allowed values (see `constants.ConnectorType`):
  - `"liquid_audio"` — Liquid Audio connector
  - `"hf_text"` — Transformers/text connector
  (default: `"liquid_audio"`)

**`dataset` (object)**
- `subset`: `string` — dataset subset identifier. Currently only `"single_sentence"` is
  supported (default: `"single_sentence"`).
- `split`: `string` — dataset split identifier. Currently only `"test"` is
  supported (default: `"test"`).
- `revision`: `string` — git/ref or dataset revision used to pull data (default: repository-specific).
- `repo_id`: `string` — HF dataset repo id (default: `Pawlo77/mllm-shap`).

**`selection` (object)** — controls which rows are processed
- `max_samples`: `int | null` — maximum number of rows to process. `null` = unlimited.
- `shuffle_seed`: `int | null` — seed used when shuffling before selection (default: `0`).
- `start_index`: `int` — zero-based index to start processing from (default: `0`).
- `max_prompt_tokens`: `int | null` — maximum prompt token length filter (optional).
- `min_prompt_tokens`: `int | null` — minimum prompt token length filter (optional).

**`wandb` (object)** — Weights & Biases logging
- `enabled`: `bool` — enable W&B logging (default: `true`).
- `project`: `string` — W&B project name (default: `"mllm-shap"`).
- `entity`: `string | null` — W&B entity (user or team).
- `group`: `string | null` — W&B group for run grouping.
- `mode`: `string | null` — W&B mode. Allowed: `"online"`, `"offline"`, `"disabled"`.
- `tags`: `list[string]` — tags attached to the run.

**`generation` (object)** — text generation parameters used when prompting models
- `max_new_tokens`: `int` — max generated tokens (default: `32`).
- `text_temperature`: `float` — sampling temperature (default: `0.2`).

**`shap` (object)** — SHAP-wide shared knobs
- `mode`: `string` — currently only `"CONTEXTUAL"` is supported.
- `normalizer`: `string` — name of normalizer class. Allowed names (registered in
  `config.NORMALIZER_MAP`): `"AbsSumNormalizer"`, `"IdentityNormalizer"`,
  `"PowerShiftNormalizer"`, `"MinMaxNormalizer"` (default: `"AbsSumNormalizer"`).
- `reducer`: `string` — embedding reducer name (registered in `config.REDUCER_MAP`):
  `"MeanReducer"`, `"MaxReducer"`, `"MinReducer"`, `"SumReducer"`,
  `"FirstReducer"`, `"ZeroReducer"` (default: `"MeanReducer"`).
- `similarity`: `string` — similarity implementation name. Allowed (see
  `constants.SimilarityType`): `"CosineSimilarity"`, `"TfIdfCosineSimilarity"`.

**`embedding` (object)** — optional external embedding model
- `model_id`: `string | null` — HF model id for external embeddings (e.g. `intfloat/e5-small-v2`).
- `revision`: `string | null` — model revision/sha.
- `max_length`: `int` — token length for embeddings (default: `64`).
- `batch_size`: `int` — embedding batch size (default: `64`).
- `l2_normalize`: `bool` — L2 normalize returned vectors (default: `true`).
- `local_files_only`: `bool` — only load local weights if true (default: `false`).

**`experiments`** — list of `ExplainerVariant` objects. Each entry defines one or more
actual runs by sweeping `num_samples` and/or `fractions`.

Fields supported inside each variant:
- `explainer_type`: `string` — one of the `ExplainerType` values:
  - `"exact"` — precise/analytical SHAP where available
  - `"limited_mc"`, `"standard_mc"` — Monte‑Carlo SHAP variants
  - `"limited_cc"`, `"standard_cc"` — complementary (CC) variants
  - `"neyman"` — Neyman-orthogonalized variant
  - `"hierarchical"` — hierarchical explainer (see below)
- `num_samples`: `list[int] | null` — list of integer sample counts. Each entry produces
  a run; `-1` may be used to indicate a particular behavior supported by some explainers.
- `fractions`: `list[float] | null` — list of fraction values in (0, 1]. Each entry produces a run.
- `linear`: `list[float] | null` — alternative fraction-like linear schedule in (0, 1].
- `name`: `string | null` — human-friendly variant name used in `run_slug`.

Hierarchical-specific fields (only for `explainer_type: "hierarchical"`):
- `hierarchical_ks`: `list[int]` — integer k values (>= 2) to sweep for hierarchical partitioning.
- `hierarchical_shap_type`: `string | null` — inner-SHAP type used at deeper levels. Allowed values
  (case-insensitive): `"precise"`, `"limited_mc"`, `"limited_cc"`, `"standard_cc"`, `"neyman"`.
- `hierarchical_shap_num_samples`: `list[int] | null` — sample counts used by the inner SHAP.
- `hierarchical_shap_fractions`: `list[float] | null` — fraction sweeps for the inner SHAP.
- `hierarchical_first_layer_type`: `string | null` — optional first-layer explainer type; allowed: `"none"`, `"precise"`,
  `"limited_mc"`, `"limited_cc"`, `"standard_cc"`, `"neyman"`.
- `hierarchical_first_layer_num_samples`: `list[int] | null` — num_samples for the first layer.
- `hierarchical_first_layer_fractions`: `list[float] | null` — fractions for the first layer.
- `hierarchical_use_importance_sampling`: `bool` — importance sampling toggle (must be `true`).
- `hierarchical_importance_min_fractions`: `list[float] | null` — min fraction sweep values in (0,1].

Validation rules (what `validate` checks)

- `dataset.subset` must be `"single_sentence"` and `dataset.split` must be `"test"`.
- `selection.max_samples` if present must be > 0; `start_index` must be >= 0.
- `wandb.mode` if present must be one of: `online`, `offline`, `disabled`.
- `shap.mode` must be `CONTEXTUAL`.
- `shap.normalizer` / `shap.reducer` must be one of the registered names shown above.
- `shap.similarity` must be `CosineSimilarity` or `TfIdfCosineSimilarity`.
- `experiments` must not be empty. For MC-like explainers (`limited_mc`, `standard_mc`, `limited_cc`,
  `standard_cc`, `neyman`) each variant must provide at least one of `num_samples`, `fractions`, or `linear`.
- `num_samples` must be a non-empty list of ints; values must be `-1` or positive integers.
- `fractions` and `linear` entries must be floats in (0, 1].
- For `hierarchical`, `hierarchical_ks` must be present and each k >= 2; `hierarchical_use_importance_sampling`
  must be `true`.

Example variant entries (typical patterns):

```json
{ "explainer_type": "limited_mc", "num_samples": [50], "name": "minimal_mc" }

{
  "explainer_type": "standard_mc",
  "num_samples": [100, 300],
  "fractions": [0.1, 0.5],
  "name": "mc_grid"
}

{
  "explainer_type": "hierarchical",
  "hierarchical_ks": [2,4,8],
  "hierarchical_shap_type": "limited_mc",
  "hierarchical_shap_num_samples": [50],
  "hierarchical_first_layer_type": "precise"
}
```

**Examples**

- Test config shipped with the package: `configs/mc_minimal.json`.
- Grid example: `configs/single_sentence_grid.json`.

**Extending the runner**

- To add a new normalizer or reducer: register the class in `config.NORMALIZER_MAP` or
  `config.REDUCER_MAP`, and reference its name in `shap.normalizer` / `shap.reducer`.
- To add a new connector: extend `ConnectorType` and implement the connector factory in
  `factory.py`.
- To add a new explainer type: implement the explainer class and map it in
  `factory.build_explainer_for_variant()`; ensure `runner.expand_variants()` and
  `config.validate_config()` understand its required knobs.

**Troubleshooting**

- If imports fail, ensure `MLLM_SHAP_SRC` is set to the `mllm_shap/src` folder or install
  `mllm_shap` into your environment.
- For verbose logs: `export LOG_LEVEL=DEBUG`.
- If runs fail mid-way, remove `checkpoint.json` in the run folder to restart clean.

**Next steps / Tips for experiments**

- Start with `configs/mc_minimal.json` to verify environment and W&B setup.
- Use small `selection.max_samples` (10–20) for fast iteration, then scale up.
- When using `hierarchical` sweeps, prefer `MinMaxNormalizer` for stable behavior (the
  runner warns if another normalizer is used).

<div align="center">
  <h1>🚀 mllm_shapx Runner</h1>
  <p><strong>Configuration-first CLI for reproducible SHAP sweeps at research scale.</strong></p>
</div>

`mllm_shapx` runs reproducible SHAP experiments with resume support, **MLflow** tracking, and optional LM Studio / OpenAI-compatible servers via the `openai_compat_text` connector.

# 📊 Runner Snapshot

- **Execution model**: config-driven variant expansion with per-variant overrides
- **Checkpointing**: built-in resume flow with versioned checkpoints
- **Output granularity**: per-sample + aggregate metrics
- **Supported connectors**: `liquid_audio`, `hf_text`, `openai_compat_text` (extensible via registry)
- **Dataset sources**: HuggingFace Parquet, HuggingFace Datasets, local Parquet, local CSV
- **Sharding**: native `--shard-index` / `--num-shards` for SLURM parallelism

# Package Layout

```
mllm_shapx/
├── __init__.py          # package marker
├── src/                 # core library modules
│   ├── __init__.py      # public API exports
│   ├── cli.py           # CLI: validate, plan, run subcommands
│   ├── config/          # configuration subsystem
│   │   ├── models.py    # Pydantic config models, inheritance
│   │   ├── registry.py  # normalizer/reducer/similarity maps
│   │   └── validation.py
│   ├── runner/          # execution engine
│   │   ├── types.py     # ExpandedVariant dataclass
│   │   ├── variants.py  # expand_variants() logic
│   │   ├── execution.py # run_single_sentence_variant orchestrator
│   │   ├── stages.py    # RowSelector, ChatBuilder, ExplainerRunner, ResultWriter
│   │   ├── helpers.py   # _LinearSampleScaler, _try_set_num_samples
│   │   └── io_utils.py  # checkpoint load/save, result serialization
│   ├── constants.py     # enums, frozensets, modality helpers
│   ├── data.py          # dataset loading, filtering, row selection
│   ├── factory.py       # connector registry, explainer/chat builders
│   ├── discovery.py     # runtime explainer/connector discovery
│   ├── mlflow_tracker.py
│   ├── analysis_snapshot.py
│   ├── audio_utils.py   # audio artifact helpers
│   ├── lm_studio.py     # LM Studio lifecycle (download/load/unload)
│   └── hf_model_download.py  # standalone model download helper
├── tests/               # test suite
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_constants.py
│   ├── test_data.py
│   ├── test_discovery.py
│   ├── test_factory.py
│   ├── test_runner.py
│   ├── test_runner_helpers.py
│   └── test_result_writer_mlflow.py
├── configs/             # experiment configuration files
│   └── example.json     # annotated example config
├── sbatchs/             # SLURM submission scripts
│   └── example.sbatch   # annotated example sbatch
├── gui/                 # Streamlit MLflow monitor
│   ├── app.py           # main entry point
│   ├── state.py         # session state management
│   ├── components/      # reusable UI components
│   └── views/           # page views
├── launch_shards_ssh.sh # cluster helper for SSH sharding
└── run_configs.sh       # local batch runner with retry logic
```

# Prerequisites

- Python 3.12
- project dependencies installed (`make install`)
- import access to `mllm_shap` (in monorepo setup, configure `MLLM_SHAP_SRC` when needed)

Example environment:

```bash
export MLLM_SHAP_SRC=/path/to/mllm_shap/src
export LOG_LEVEL=INFO
```

See `.example.env` for defaults.

# CLI Entry Point

```bash
# From project root:
python -m experiments.mllm_shapx.src.cli --help
```

# Common Commands

Validate configuration:

```bash
python -m experiments.mllm_shapx.src.cli validate --config path/to/config.json --check-dataset
```

Preview experiment plan (dry run):

```bash
python -m experiments.mllm_shapx.src.cli plan --config path/to/config.json
```

Run with MLflow (set a tracking URI in the environment or in JSON `mlflow.tracking_uri`):

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5050"
python -m experiments.mllm_shapx.src.cli run --config path/to/config.json --resume
```

### Streamlit monitor

```bash
uv run streamlit run experiments/mllm_shapx/gui/app.py
```

### SSH shard launcher

Use `launch_shards_ssh.sh` to run `--shard-index` / `--num-shards` across hosts (see script header). All workers should share the same `MLFLOW_TRACKING_URI`. Supports `--resume` (default), `--no-resume`, and `--max-samples`.

```bash
export MLFLOW_TRACKING_URI="http://mlflow.internal:5050"
./experiments/mllm_shapx/launch_shards_ssh.sh hosts.txt configs/mc_minimal.json
./experiments/mllm_shapx/launch_shards_ssh.sh --no-resume --max-samples 50 hosts.txt configs/mc_minimal.json
```

Run with sharding (SLURM):

```bash
python -m experiments.mllm_shapx.src.cli run --config path/to/config.json \
    --shard-index $SLURM_ARRAY_TASK_ID --num-shards $SLURM_ARRAY_TASK_COUNT --resume
```

Batch mode with glob:

```bash
python -m experiments.mllm_shapx.src.cli run --config "configs/package_grid/*.json" --resume
```

Use `--resume` to continue interrupted runs using checkpoints.

# Configuration

Start from the annotated example:

- `configs/example.json`

## Key Features

- **Config inheritance**: Use `"base": "path/to/parent.json"` to share common settings
- **Per-variant overrides**: Set `shap_override`, `generation_override`, `embedding_override` per experiment
- **Dataset sources**: `hf_parquet`, `hf_datasets`, `local_parquet`, `local_csv`
- **Generic filters**: Filter dataset rows with `{"column": "lang", "op": "in", "value": ["en", "de"]}`
- **Full generation control**: `text_temperature`, `text_top_k`, `audio_temperature`, `audio_top_k`
- **Token filter**: `exclude_punctuation` or `none`
- **Connector registry**: Register custom connectors at runtime

## Connector Selection

| Connector | Value | Description |
|-----------|-------|-------------|
| Liquid Audio | `liquid_audio` | Multi-modal audio+text model |
| Transformers Text | `hf_text` | Text-only HuggingFace causal LM |
| OpenAI-compatible | `openai_compat_text` | LM Studio / vLLM / any OpenAI-API server |

## Explainer Types

| Type | Value | Parameters |
|------|-------|------------|
| Exact (Precise) | `exact` | — |
| Limited MC | `limited_mc` / `mc` | `num_samples`, `fractions`, `linear` |
| Standard MC | `standard_mc` | `num_samples`, `fractions`, `linear` |
| Limited Complementary | `limited_cc` / `cc` | `num_samples`, `fractions`, `linear` |
| Standard Complementary | `standard_cc` | `num_samples`, `fractions`, `linear` |
| Limited Neyman | `limited_neyman` / `neyman` | `num_samples`, `fractions`, `linear` |
| Standard Neyman | `standard_neyman` | `num_samples`, `fractions`, `linear` |
| Hierarchical | `hierarchical` | `hierarchical: {ks, shap_type, ...}` |

For field-level validation and constraints, use the `validate` command before running large jobs.

# Output Layout

Each resolved run writes to:

`{output_root}/{experiment_set_id}/{run_slug}/`

with:
- `spec.json` (resolved config)
- `checkpoint.json` (resume state, versioned)
- `samples/sample_XXXXX_result.json` (per-sample outputs)
- `summary/aggregate_metrics.json` (run summary)

# Running Tests

```bash
python -m pytest experiments/mllm_shapx/tests/ -v
```

# 🛠️ Extension Points

- Register new connectors at runtime via `register_connector(name, factory)`.
- Register new reducers/normalizers in `src/config.py` maps.
- Add new explainer types by extending `expand_variants()` in `src/runner.py`.
- Use config inheritance (`"base"` key) to share settings across experiment configs.
- Add per-variant overrides for shap, generation, or embedding settings.

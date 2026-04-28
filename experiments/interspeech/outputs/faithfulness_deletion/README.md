# HP-1 Faithfulness Deletion Validation

This experiment validates whether the SGPA Shapley values identify causally important audio segments. It is a post-hoc deletion test over existing `mllm_shapx` SGPA runs, not a re-estimation of Shapley values.

## Exact Approach

For each serialized `mllm_shapx` sample:

1. Load the original dataset row using the `spec.json` stored with the run. This preserves the exact Hugging Face dataset revision, subset, split, row ordering, and shuffle seed used by the original SGPA run.
2. Read the audio Shapley values from `samples/sample_XXXXX_result.json`.
3. Select the top segment as `argmax(abs(SV_i))`.
4. Reconstruct the SGPA word segmentation from the original audio and transcript using `SpectrogramGuidedAligner`.
5. Silence the top-SV segment in the waveform while preserving total audio duration.
6. Pick one random non-top SGPA segment and silence an equal-duration interval centered on that random segment. Equal duration keeps the deletion amount matched.
7. Run LFM2-Audio three times with the same audio-only prompt style as the original run:
   - original audio,
   - top-SV deletion,
   - random equal-duration deletion.
8. Score response preservation with `TfIdfCosineSimilarity`, the same response-token similarity objective used by the existing audio-output `mllm_shapx` runs.
9. Compute drops:
   - `top_drop = original_similarity - top_deleted_similarity`
   - `random_drop = original_similarity - random_deleted_similarity`
   - `drop_difference = top_drop - random_drop`
10. Run a paired t-test comparing `top_drop` against `random_drop`, and report Cohen's dz.

The validation claim is supported if top-SV deletion causes a larger response change than random equal-duration deletion, i.e. mean `drop_difference > 0`.

## Why This Reuses `mllm_shapx`

`mllm_shapx` is reused for the pieces that must match the original experiments:

- dataset loading and row ordering,
- `spec.json` configuration,
- LiquidAudio model construction,
- chat construction and generation style,
- existing SGPA Shapley outputs.

The deletion evaluation is implemented separately in `experiments/interspeech/src/faithfulness_deletion.py` because it is not a generic Shapley estimation run. It consumes completed Shapley runs and produces paired deletion statistics.

## Outputs

For non-partitioned local/debug runs:

- `audio__male_results.csv` or `audio__female_results.csv`
- `audio__male_failures.csv` or `audio__female_failures.csv`
- `audio__male_summary.json` or `audio__female_summary.json`

For partitioned SLURM runs:

- `audio__male_partK-ofN_results.csv`
- `audio__female_partK-ofN_results.csv`
- matching `*_failures.csv` and `*_summary.json` files

After all array tasks finish, combine partitions with:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m experiments.interspeech.src.faithfulness_deletion \
  --output-dir experiments/interspeech/outputs/faithfulness_deletion \
  --combine-only
```

This writes:

- `audio__male_combined_results.csv`
- `audio__female_combined_results.csv`
- `combined_results.csv`
- voice-level and combined summary JSON files.

## Local Validation

## Lem Environment Setup

On Lem, the repository is cloned fresh and the virtual environment must be created there. A failed `uv sync` can leave a partial `.venv`; if basic imports such as `numpy` fail, the sync did not complete.

Recommended one-time setup from the repository root:

```bash
cd ~/MLLM-Shap
export PATH="$HOME/.local/bin:$PATH"
export UV_HTTP_TIMEOUT=300
export UV_CONCURRENT_DOWNLOADS=2
rm -rf .venv
uv sync --group dev
```

Why this matters: `--group dev` is needed because the experiment utilities import dependencies from the dev group, including Hugging Face dataset tooling. The NVIDIA wheel timeout during `nvidia-cusparse` download means the environment was not installed; running `.venv/bin/python` after that failure will produce `ModuleNotFoundError` errors.

After setup, verify the environment:

```bash
.venv/bin/python - <<'PY'
import numpy, pandas, torch
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
```

Use the same `.venv` for interactive validation and `sbatch`. Do not run dependency installation inside each SLURM array task.

Use preflight mode to validate dataset access, row ordering, SV extraction, SGPA alignment, deletion interval selection, and audio re-encoding without loading LFM2:

```bash
PYTHONPATH="$PWD" UV_PROJECT_ENVIRONMENT="$PWD/.venv" \
uv run --project "$PWD" --group dev \
python -m experiments.interspeech.src.faithfulness_deletion \
  --run-dir experiments/experiments_output/single_sentence_2026_01_03/audio_male_audio_limited_neyman_lin3_0 \
  --output-dir experiments/interspeech/outputs/faithfulness_deletion \
  --max-samples 2 \
  --preflight-only
```

Full local model validation is:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m experiments.interspeech.src.faithfulness_deletion \
  --run-dir experiments/experiments_output/single_sentence_2026_01_03/audio_male_audio_limited_neyman_lin3_0 \
  --output-dir experiments/interspeech/outputs/faithfulness_deletion \
  --max-samples 1 \
  --max-new-tokens 1 \
  --device cuda \
  --aligner-device cpu \
  --resume
```

On the current local machine, CUDA initialization fails because the NVIDIA driver is older than the installed PyTorch/CUDA build. The full generation run should therefore be validated on Lem's interactive GPU partition before launching a large production array.

## SLURM Strategy

The production script is `experiments/run_hp1_faithfulness.sbatch`.

The experiment is embarrassingly parallel over samples, so the best cluster strategy is one LFM2 replica per GPU and many independent array tasks. The script requests one H100 per array task and splits each voice into `PARTITIONS_PER_VOICE` partitions:

- default `PARTITIONS_PER_VOICE=8`,
- two voices,
- total default array tasks: `2 * 8 = 16`,
- array directive: `#SBATCH --array=0-15%16`.

This uses up to 16 H100 GPUs concurrently and lets Slurm pack jobs across the available H100 nodes. For bigger sample counts and more available GPUs, increase `PARTITIONS_PER_VOICE` and update the array directive accordingly:

- `PARTITIONS_PER_VOICE=16` -> `#SBATCH --array=0-31%32`
- `PARTITIONS_PER_VOICE=32` -> `#SBATCH --array=0-63%64`
- `PARTITIONS_PER_VOICE=64` -> `#SBATCH --array=0-127%128`

Because each task uses only one GPU, do not request `--gres=gpu:4` unless the Python runner is changed to launch four independent workers inside one Slurm job. For the current code, one GPU per task gives better scheduling flexibility and near-linear scaling.

Before submitting, replace:

```bash
#SBATCH -A plg<insert_grant_name_here>
```

with the actual PLGrid grant, for example `#SBATCH -A plgYOUR_GRANT`.

Submit production:

```bash
sbatch experiments/run_hp1_faithfulness.sbatch
```

Recommended interactive validation on Lem:

```bash
srun -A plgYOUR_GRANT -p lem-gpu-interactive --gres=gpu:hopper:1 --cpus-per-task=16 --mem=40G --time=01:00:00 --pty bash
```

Then run the full one-sample command from the repository root.

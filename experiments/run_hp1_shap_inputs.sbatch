#!/bin/bash
#SBATCH -J hp1_shap
#SBATCH -A plg<insert_grant_name_here>
#SBATCH -p plgrid-lem-gpu-h100
# Default: 100 samples/voice at one sample per task -> 200 tasks for male+female.
# For larger runs, override --array and MAX_SAMPLES_PER_VOICE at submission time.
#SBATCH --array=0-199%32
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MAX_SAMPLES_PER_VOICE="${MAX_SAMPLES_PER_VOICE:-100}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-1}"

CONFIG_PATHS=(
  "experiments/interspeech/configs/hp1_shap_inputs_audio_male.json"
  "experiments/interspeech/configs/hp1_shap_inputs_audio_female.json"
)

VOICE_COUNT="${#CONFIG_PATHS[@]}"
SHARDS_PER_VOICE=$(((MAX_SAMPLES_PER_VOICE + SAMPLES_PER_TASK - 1) / SAMPLES_PER_TASK))
TOTAL_TASKS=$((VOICE_COUNT * SHARDS_PER_VOICE))
TASK_ID="${SLURM_ARRAY_TASK_ID}"

if (( TASK_ID >= TOTAL_TASKS )); then
  echo "Task ${TASK_ID} is outside TOTAL_TASKS=${TOTAL_TASKS}; exiting cleanly."
  exit 0
fi

VOICE_INDEX=$((TASK_ID / SHARDS_PER_VOICE))
SHARD_INDEX=$((TASK_ID % SHARDS_PER_VOICE))
CONFIG_PATH="${CONFIG_PATHS[$VOICE_INDEX]}"
START_INDEX=$((SHARD_INDEX * SAMPLES_PER_TASK))
REMAINING=$((MAX_SAMPLES_PER_VOICE - START_INDEX))
TASK_MAX_SAMPLES="$SAMPLES_PER_TASK"
if (( REMAINING < TASK_MAX_SAMPLES )); then
  TASK_MAX_SAMPLES="$REMAINING"
fi

cd "$PROJECT_DIR"
mkdir -p logs

# Uncomment/load site-specific modules if your Lem environment needs them.
# module purge
# module load CUDA
# module load Python

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
  PYTHON_RUNNER="$PROJECT_DIR/.venv/bin/python"
else
  echo "Missing $PROJECT_DIR/.venv/bin/python"
  echo "Create the environment once before sbatch with:"
  echo "  cd $PROJECT_DIR"
  echo "  uv sync --group dev"
  exit 2
fi
export PYTHON_RUNNER CONFIG_PATH START_INDEX TASK_MAX_SAMPLES

TMP_CONFIG="${SLURM_TMPDIR:-/tmp}/hp1_shap_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
export TMP_CONFIG

echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "MAX_SAMPLES_PER_VOICE=$MAX_SAMPLES_PER_VOICE"
echo "SAMPLES_PER_TASK=$SAMPLES_PER_TASK"
echo "SHARDS_PER_VOICE=$SHARDS_PER_VOICE"
echo "VOICE_INDEX=$VOICE_INDEX"
echo "SHARD_INDEX=$SHARD_INDEX"
echo "START_INDEX=$START_INDEX"
echo "TASK_MAX_SAMPLES=$TASK_MAX_SAMPLES"

"$PYTHON_RUNNER" - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["CONFIG_PATH"])
cfg = json.loads(src.read_text(encoding="utf-8"))
cfg["selection"]["start_index"] = int(os.environ["START_INDEX"])
cfg["selection"]["max_samples"] = int(os.environ["TASK_MAX_SAMPLES"])

dst = Path(os.environ["TMP_CONFIG"])
dst.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(f"Wrote shard config: {dst}")
PY

srun --export=ALL --gpu-bind=single:1 bash -lc '
  echo "Inside step:"
  echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  nvidia-smi || { echo "nvidia-smi failed in step"; exit 2; }

  "$PYTHON_RUNNER" -u -m experiments.mllm_shapx.cli run \
      --config "$TMP_CONFIG"
'

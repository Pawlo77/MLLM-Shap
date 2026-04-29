#!/bin/bash
#SBATCH -J rebuttal_shap
#SBATCH -A plg<insert_grant_name_here>
#SBATCH -p plgrid-lem-gpu-h100
# Submit with --array sized for CONFIG_LIST * shards_per_config.
# Example: 3 configs * 100 one-sample shards -> --array=0-299%32.
#SBATCH --array=0-299%32
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG_LIST="${CONFIG_LIST:-experiments/interspeech/configs/rebuttal_stage1_1k_sgpa_configs.txt}"
MAX_SAMPLES_PER_CONFIG="${MAX_SAMPLES_PER_CONFIG:-100}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-1}"

cd "$PROJECT_DIR"
mkdir -p logs

mapfile -t CONFIG_PATHS < "$CONFIG_LIST"
CONFIG_COUNT="${#CONFIG_PATHS[@]}"
SHARDS_PER_CONFIG=$(((MAX_SAMPLES_PER_CONFIG + SAMPLES_PER_TASK - 1) / SAMPLES_PER_TASK))
TOTAL_TASKS=$((CONFIG_COUNT * SHARDS_PER_CONFIG))
TASK_ID="${SLURM_ARRAY_TASK_ID}"

if (( TASK_ID >= TOTAL_TASKS )); then
  echo "Task ${TASK_ID} is outside TOTAL_TASKS=${TOTAL_TASKS}; exiting cleanly."
  exit 0
fi

CONFIG_INDEX=$((TASK_ID / SHARDS_PER_CONFIG))
SHARD_INDEX=$((TASK_ID % SHARDS_PER_CONFIG))
CONFIG_PATH="${CONFIG_PATHS[$CONFIG_INDEX]}"
START_INDEX=$((SHARD_INDEX * SAMPLES_PER_TASK))
REMAINING=$((MAX_SAMPLES_PER_CONFIG - START_INDEX))
TASK_MAX_SAMPLES="$SAMPLES_PER_TASK"
if (( REMAINING < TASK_MAX_SAMPLES )); then
  TASK_MAX_SAMPLES="$REMAINING"
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MLLM_SHAPX_DISABLE_CHECKPOINT=1

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

TMP_CONFIG="${SLURM_TMPDIR:-/tmp}/rebuttal_shap_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
export TMP_CONFIG

echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "CONFIG_LIST=$CONFIG_LIST"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "MAX_SAMPLES_PER_CONFIG=$MAX_SAMPLES_PER_CONFIG"
echo "SAMPLES_PER_TASK=$SAMPLES_PER_TASK"
echo "CONFIG_INDEX=$CONFIG_INDEX"
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

srun --export=ALL bash -lc '
  echo "Inside step:"
  echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  nvidia-smi || { echo "nvidia-smi failed in step"; exit 2; }

  "$PYTHON_RUNNER" -u -m experiments.mllm_shapx.cli run \
      --config "$TMP_CONFIG" \
      --resume
'

#!/bin/bash
# Example SLURM sbatch script for running mllm_shapx experiments.
# Adjust #SBATCH directives, CONFIG, and paths to match your cluster.
#
# Usage:
#   Single config:  sbatch run_mllm_shapx.sbatch
#   Array (sharded): sbatch --array=0-3%4 run_mllm_shapx.sbatch
#
#SBATCH -J mllm_shapx
#SBATCH -A <your_grant>
#SBATCH -p <your_partition>
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────────
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG="${CONFIG:-mllm_shapx/configs/your_experiment.json}"
MAX_SAMPLES="${MAX_SAMPLES:-}"          # leave empty to use config value
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

# ── Environment ─────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"
mkdir -p logs

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# ── Python interpreter ──────────────────────────────────────────────────────
if [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
  PYTHON="$PROJECT_DIR/.venv/bin/python"
else
  echo "ERROR: Missing $PROJECT_DIR/.venv/bin/python"
  echo "Create the environment before submitting:"
  echo "  cd $PROJECT_DIR && uv sync --group dev"
  exit 2
fi

# ── Run ─────────────────────────────────────────────────────────────────────
echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0}"
echo "CONFIG=$CONFIG"

CLI_ARGS=(run --config "$CONFIG" --resume)

if [[ -n "$MAX_SAMPLES" ]]; then
  CLI_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

if (( NUM_SHARDS > 1 )); then
  CLI_ARGS+=(--shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS")
fi

exec "$PYTHON" -m mllm_shapx.src.cli "${CLI_ARGS[@]}"

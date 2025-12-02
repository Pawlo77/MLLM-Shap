#!/bin/bash
#SBATCH -J mllm_shapx
#SBATCH -A ganzha_23
#SBATCH -p short,long
#SBATCH --array=0-1%2
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

WORKDIR="$HOME/projects/MLLM-Shap/experiments"
CONFIG_DIR="mllm_shapx/configs/package_exp"

CONFIGS=(
  "$CONFIG_DIR/complementary.json"
  "$CONFIG_DIR/exact.json"
)

IDX=${SLURM_ARRAY_TASK_ID}
CONFIG="${CONFIGS[$IDX]}"

cd "$WORKDIR"
mkdir -p logs

# W&B env
if [ -f "$HOME/.secrets/wandb.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.secrets/wandb.sh"
fi
export WANDB_MODE="${WANDB_MODE:-online}"

export TOKENIZERS_PARALLELISM=false
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"

echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} cfg=$(basename "$CONFIG")"

# Optional per-config tee
CFG_BASENAME="$(basename "$CONFIG" .json)"
TEE_LOG="logs/${CFG_BASENAME}__${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

srun uv run python -m mllm_shapx.cli run --config "$CONFIG" 2>&1 | tee "$TEE_LOG"

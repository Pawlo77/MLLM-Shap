#!/bin/bash
#SBATCH -J mllm_shapx_exact
#SBATCH -A ganzha_23
#SBATCH -p short,long
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --export=ALL

set -euo pipefail

WORKDIR="$HOME/projects/MLLM-Shap/experiments"
CONFIG="mllm_shapx/configs/package_exp/exact.json"

cd "$WORKDIR"
mkdir -p logs

# W&B
if [ -f "$HOME/.secrets/wandb.sh" ]; then
  source "$HOME/.secrets/wandb.sh"
fi
export WANDB_MODE="${WANDB_MODE:-online}"

# Caches & PATH
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
TEE_LOG="logs/exact__${SLURM_JOB_ID}.log"

echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID} cfg=$(basename "$CONFIG")"

# Do EVERYTHING inside the step (so GPU binding and env are identical)
srun --export=ALL --gpu-bind=single:1 bash -lc '
  echo "Inside step:"
  echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  nvidia-smi || { echo "nvidia-smi failed in step"; exit 2; }

  echo "Torch CUDA sanity:"
  python - <<PY
import os, torch
print("  torch.__version__   =", torch.__version__)
print("  torch.version.cuda  =", torch.version.cuda)
print("  is_available        =", torch.cuda.is_available())
print("  device_count        =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("  gpu0 name           =", torch.cuda.get_device_name(0))
PY

  echo "Running pipeline…"
  source ~/projects/MLLM-Shap/.venv/bin/activate
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
  python -u -m mllm_shapx.src.cli run --config mllm_shapx/configs/package_exp/exact.json --resume 2>&1 | tee "$TEE_LOG"
'

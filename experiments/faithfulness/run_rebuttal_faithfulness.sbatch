#!/bin/bash
#SBATCH -J rebut_faith
#SBATCH -A plg<insert_grant_name_here>
#SBATCH -p plgrid-lem-gpu-h100
# 5 runs * PARTITIONS_PER_RUN=8 => 40 tasks. If you change the run list or
# partitions, override with: sbatch --array=0-$((RUN_COUNT*PARTITIONS_PER_RUN-1))
#SBATCH --array=0-39%20
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN_LIST="${RUN_LIST:-experiments/faithfulness/configs/rebuttal_current_runs.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/faithfulness/outputs/rebuttal_new_data}"
PARTITIONS_PER_RUN="${PARTITIONS_PER_RUN:-8}"
ALIGNER_DEVICE="${ALIGNER_DEVICE:-cpu}"
MODE="${MODE:-rankwise}" # rankwise | aggregate

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_ROOT"

mapfile -t RUN_ROWS < <(awk 'NF && $1 !~ /^#/' "$RUN_LIST")
RUN_COUNT="${#RUN_ROWS[@]}"
TOTAL_TASKS=$((RUN_COUNT * PARTITIONS_PER_RUN))
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID >= TOTAL_TASKS )); then
  echo "Task ${TASK_ID} is outside TOTAL_TASKS=${TOTAL_TASKS}; exiting cleanly."
  exit 0
fi

RUN_INDEX=$((TASK_ID / PARTITIONS_PER_RUN))
PARTITION_INDEX=$((TASK_ID % PARTITIONS_PER_RUN))
IFS=$'\t' read -r LABEL RUN_DIR SPEC_PATH MAX_SAMPLES <<< "${RUN_ROWS[$RUN_INDEX]}"

case "$MODE" in
  rankwise)
    MODE_DIR="rankwise"
    ;;
  aggregate)
    MODE_DIR="aggregate"
    ;;
  *)
    echo "MODE must be 'rankwise' or 'aggregate', got: $MODE" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="$OUTPUT_ROOT/$MODE_DIR/$LABEL"
mkdir -p "$OUTPUT_DIR"

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PROJECT_DIR RUN_LIST OUTPUT_ROOT PARTITIONS_PER_RUN ALIGNER_DEVICE MODE
export LABEL RUN_DIR SPEC_PATH MAX_SAMPLES OUTPUT_DIR PARTITION_INDEX

if [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
  PYTHON_RUNNER="$PROJECT_DIR/.venv/bin/python"
else
  echo "Missing $PROJECT_DIR/.venv/bin/python"
  echo "Create the environment once before sbatch with:"
  echo "  cd $PROJECT_DIR"
  echo "  uv sync --group dev"
  exit 2
fi
export PYTHON_RUNNER

echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID:-manual} task=${TASK_ID}"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "MODE=$MODE"
echo "LABEL=$LABEL"
echo "RUN_DIR=$RUN_DIR"
echo "SPEC_PATH=$SPEC_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "PARTITIONS_PER_RUN=$PARTITIONS_PER_RUN"
echo "PARTITION_INDEX=$PARTITION_INDEX"

srun --export=ALL bash -lc '
  echo "Inside step:"
  echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  nvidia-smi || { echo "nvidia-smi failed in step"; exit 2; }

  extra_args=()
  if [ "$MODE" = "rankwise" ]; then
    extra_args+=(--all-rank-deletions)
  fi

  "$PYTHON_RUNNER" -u -m experiments.faithfulness.src.run \
      --run-dir "$RUN_DIR" \
      --spec-path "$SPEC_PATH" \
      --output-dir "$OUTPUT_DIR" \
      --max-samples "$MAX_SAMPLES" \
      --device cuda \
      --aligner-device "$ALIGNER_DEVICE" \
      --partition-index "$PARTITION_INDEX" \
      --num-partitions "$PARTITIONS_PER_RUN" \
      --resume \
      "${extra_args[@]}"
'

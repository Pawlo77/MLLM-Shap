#!/bin/bash
#SBATCH -J hp1_faith
#SBATCH -A plg<insert_grant_name_here>
#SBATCH -p plgrid-lem-gpu-h100
# For PARTITIONS_PER_VOICE=8 and two voices, use 16 array tasks.
# If you change PARTITIONS_PER_VOICE, update this to 0-$((2*PARTITIONS_PER_VOICE-1)).
#SBATCH --array=0-15%16
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/MLLM-Shap}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/interspeech/outputs/faithfulness_deletion}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
ALIGNER_DEVICE="${ALIGNER_DEVICE:-cpu}"
PARTITIONS_PER_VOICE="${PARTITIONS_PER_VOICE:-8}"

RUN_DIRS=(
  "experiments/experiments_output/single_sentence_2026_01_03/audio_male_audio_limited_neyman_lin3_0"
  "experiments/experiments_output/single_sentence_2026_01_03/audio_female_audio_limited_neyman_lin3_0"
)
SPEC_PATHS=(
  "experiments/interspeech/configs/hp1_faithfulness_audio_male_spec.json"
  "experiments/interspeech/configs/hp1_faithfulness_audio_female_spec.json"
)

VOICE_COUNT="${#RUN_DIRS[@]}"
TOTAL_TASKS=$((VOICE_COUNT * PARTITIONS_PER_VOICE))
TASK_ID="${SLURM_ARRAY_TASK_ID}"

if (( TASK_ID >= TOTAL_TASKS )); then
  echo "Task ${TASK_ID} is outside TOTAL_TASKS=${TOTAL_TASKS}; exiting cleanly."
  exit 0
fi

VOICE_INDEX=$((TASK_ID / PARTITIONS_PER_VOICE))
PARTITION_INDEX=$((TASK_ID % PARTITIONS_PER_VOICE))
RUN_DIR="${RUN_DIRS[$VOICE_INDEX]}"
SPEC_PATH="${SPEC_PATHS[$VOICE_INDEX]}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"

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
export PROJECT_DIR OUTPUT_DIR MAX_SAMPLES ALIGNER_DEVICE RUN_DIR SPEC_PATH
export PARTITIONS_PER_VOICE PARTITION_INDEX

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

echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "SPEC_PATH=$SPEC_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "PARTITIONS_PER_VOICE=$PARTITIONS_PER_VOICE"
echo "PARTITION_INDEX=$PARTITION_INDEX"

srun --export=ALL --gpu-bind=single:1 bash -lc '
  echo "Inside step:"
  echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  nvidia-smi || { echo "nvidia-smi failed in step"; exit 2; }

  "$PYTHON_RUNNER" -u -m experiments.interspeech.src.faithfulness_deletion \
      --run-dir "$RUN_DIR" \
      --spec-path "$SPEC_PATH" \
      --output-dir "$OUTPUT_DIR" \
      --max-samples "$MAX_SAMPLES" \
      --device cuda \
      --aligner-device "$ALIGNER_DEVICE" \
      --partition-index "$PARTITION_INDEX" \
      --num-partitions "$PARTITIONS_PER_VOICE" \
      --resume
'

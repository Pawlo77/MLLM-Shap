#!/usr/bin/env bash
# Second-model (Voxtral-Mini-3B, Mistral family) SGPA exact-Shapley pipeline.
#
# Same 7-condition suite as run_queue.sh but with --backend voxtral via mm_faith.
# Voxtral is native in transformers 5.4 (no trust_remote_code), so there is no
# version conflict and no isolated env needed.
# Waits for the GPU to be free before each job, so it can be launched now and
# will transparently start after the Qwen queue (run_queue.sh) finishes.
# Every job uses --resume (safe to re-run).
#
# IMPORTANT: validate the backend first with (needs a free GPU):
#   .venv/bin/python -m experiments.faithfulness.src.voxtral_audio_backend
# Only then launch this for the overnight run.
set -u
cd "$(dirname "$0")"

export PYTHONPATH="$PWD/mllm_shap/src"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=.venv/bin/python
QDIR="logs/queue_voxtral"
mkdir -p "$QDIR"
STATUS="$QDIR/status.tsv"
BACKEND="voxtral"

R1K="experiments/experiments_output/rebuttal_single_sentence_1k_sgpa"
ORIG="experiments/experiments_output/aaai27_fixed_500_original/audio_original_audio_sgpa_limited_neyman_lin3_0"
OUT="experiments/faithfulness/outputs/voxtral"

# Sample selection matches run_queue.sh: exactly 100 utterances, word count
# [4,7]. LibriSpeech original + ablations keep the token-balanced 100; TTS voice
# sets use --full-pool for a word-banded 100 (identical ids across voices).
# name | run_dir | output_dir | extra_flags
JOBS=(
  "voxtral_original|$ORIG|$OUT/exact_shapley_original|--max-samples 100"
  "voxtral_male|$R1K/audio_male_audio_sgpa_limited_neyman_lin3_0|$OUT/exact_shapley_male|--max-samples 100 --full-pool"
  "voxtral_female|$R1K/audio_female_audio_sgpa_limited_neyman_lin3_0|$OUT/exact_shapley_female|--max-samples 100 --full-pool"
  "voxtral_tts_original|$R1K/audio_original_audio_sgpa_limited_neyman_lin3_0|$OUT/exact_shapley_ttsorig|--max-samples 100 --full-pool"
  "voxtral_stage3off|$ORIG|$OUT/exact_shapley_stage3off|--max-samples 100 --stage3-off"
  "voxtral_mask_noise|$ORIG|$OUT/exact_shapley_mask_noise|--max-samples 60 --mask-mode noise"
  "voxtral_mask_concat|$ORIG|$OUT/exact_shapley_mask_concat|--max-samples 60 --mask-mode concat"
)

set_status() {
  local name="$1" state="$2" extra="${3:-}"
  local tmp="$STATUS.$$"
  touch "$STATUS"
  grep -v "^${name} " "$STATUS" > "$tmp" 2>/dev/null || true
  echo "${name} ${state} $(date +%H:%M:%S) ${extra}" >> "$tmp"
  mv "$tmp" "$STATUS"
}

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1
}

wait_for_gpu() {
  local used
  while true; do
    used="$(gpu_used_mib)"
    if [ "${used:-9999}" -lt 1500 ]; then
      sleep 3
      used="$(gpu_used_mib)"
      [ "${used:-9999}" -lt 1500 ] && return 0
    fi
    sleep 15
  done
}

for spec in "${JOBS[@]}"; do
  set_status "${spec%%|*}" QUEUED ""
done

for spec in "${JOBS[@]}"; do
  IFS='|' read -r name rundir outdir extra <<< "$spec"
  if [ ! -d "$rundir" ]; then
    set_status "$name" FAILED "missing-run-dir"
    continue
  fi
  set_status "$name" WAIT_GPU ""
  wait_for_gpu
  set_status "$name" RUNNING "$outdir"
  log="$QDIR/${name}.log"
  # shellcheck disable=SC2086
  $PY -u -m experiments.faithfulness.src.mm_faith \
      --backend "$BACKEND" \
      --run-dir "$rundir" \
      --output-dir "$outdir" \
      --max-players 7 --resume $extra > "$log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    set_status "$name" DONE "rc=0"
  else
    set_status "$name" FAILED "rc=$rc"
  fi
done

echo "voxtral queue finished at $(date +%H:%M:%S)"

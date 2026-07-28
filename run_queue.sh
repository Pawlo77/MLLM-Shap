#!/usr/bin/env bash
# Complete single-model (Qwen2-Audio) SGPA exact-Shapley pipeline for AAAI-27.
#
# The 12 GB GPU fits one Qwen job at a time, so jobs run strictly sequentially.
# Before each GPU job the runner waits until the GPU is free, so it also queues
# behind anything already running. Every job uses --resume (safe to re-run).
# After all GPU jobs it runs the CPU-only consolidated analysis.
#
# Usage:
#   ./run_queue.sh                 # run full pipeline
#   watch -n 5 ./monitor_jobs.sh   # (another shell) live status
set -u
cd "$(dirname "$0")"

export PYTHONPATH="$PWD/mllm_shap/src"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=.venv/bin/python
QDIR="logs/queue"
mkdir -p "$QDIR"
STATUS="$QDIR/status.tsv"

R1K="experiments/experiments_output/rebuttal_single_sentence_1k_sgpa"
ORIG="experiments/experiments_output/aaai27_fixed_500_original/audio_original_audio_sgpa_limited_neyman_lin3_0"
OUT="experiments/faithfulness/outputs"

# Sample selection: exactly 100 utterances with word count in [4,7].
#  - LibriSpeech "original" + its ablations keep the finished token-balanced 100
#    (already all in-band); the word band is a no-op there.
#  - TTS voice sets (male/female/tts_original) use --full-pool to draw a
#    word-banded 100 from the whole 854-utterance dataset (identical id set
#    across the three voices for a fair paired comparison).
# One job per line: name | run_dir | output_dir | extra_flags
# NOTE: qwen_original already completed (token-balanced 100, all in [4,7] words)
# and is intentionally NOT re-run here -- its outputs are preserved on disk and
# still picked up by qwen_analyze.
JOBS=(
  "qwen_male|$R1K/audio_male_audio_sgpa_limited_neyman_lin3_0|$OUT/qwen_exact_shapley_male|--max-samples 100 --full-pool"
  "qwen_female|$R1K/audio_female_audio_sgpa_limited_neyman_lin3_0|$OUT/qwen_exact_shapley_female|--max-samples 100 --full-pool"
  "qwen_tts_original|$R1K/audio_original_audio_sgpa_limited_neyman_lin3_0|$OUT/qwen_exact_shapley_ttsorig|--max-samples 100 --full-pool"
  "qwen_stage3off|$ORIG|$OUT/qwen_exact_shapley_stage3off|--max-samples 100 --stage3-off"
  "qwen_mask_noise|$ORIG|$OUT/qwen_exact_shapley_mask_noise|--max-samples 60 --mask-mode noise"
  "qwen_mask_concat|$ORIG|$OUT/qwen_exact_shapley_mask_concat|--max-samples 60 --mask-mode concat"
)

set_status() { # name state extra
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
  $PY -u -m experiments.faithfulness.src.qwen_faith \
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

# Consolidated CPU analysis (tables + AOPC figures) over whatever completed.
set_status "analysis" RUNNING ""
$PY -u -m experiments.faithfulness.src.qwen_analyze > "$QDIR/analysis.log" 2>&1
if [ $? -eq 0 ]; then set_status "analysis" DONE "rc=0"; else set_status "analysis" FAILED "see analysis.log"; fi

echo "queue finished at $(date +%H:%M:%S)"

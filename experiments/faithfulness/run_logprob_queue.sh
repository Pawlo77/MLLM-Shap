#!/usr/bin/env bash
# Log-probability faithfulness endpoint (7.1) for Qwen + Voxtral on LibriSpeech and
# TTS-original. Reuses cached SGPA segments + exact SVs; only scoring is new.
# Run detached from the mllm-shap repo root:
#   setsid bash experiments/faithfulness/run_logprob_queue.sh > /tmp/logprob_queue.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/../.."   # -> mllm-shap repo root

PY=.venv/bin/python
OUT=experiments/faithfulness/outputs/logprob_qwen_voxtral
LIBRI=experiments/experiments_output/aaai27_fixed_500_original/audio_original_audio_sgpa_limited_neyman_lin3_0
TTS=experiments/experiments_output/rebuttal_single_sentence_1k_sgpa/audio_original_audio_sgpa_limited_neyman_lin3_0
QC=experiments/faithfulness/outputs
VC=experiments/faithfulness/outputs/voxtral

run () {  # model condition run_dir cached_results extra_flags
  echo "=================================================================="
  echo ">> $1 / $2  ($(date '+%H:%M:%S'))"
  echo "=================================================================="
  $PY -m experiments.faithfulness.src.logprob_endpoint \
    --model "$1" --condition "$2" --run-dir "$3" --cached-results "$4" \
    --output-dir "$OUT" --max-samples 100 --resume $5 || echo "!! $1/$2 FAILED"
}

run qwen    librispeech_original "$LIBRI" "$QC/qwen_exact_shapley_original/qwen_exact_shapley_results.csv" ""
run qwen    tts_original         "$TTS"   "$QC/qwen_exact_shapley_ttsorig/qwen_exact_shapley_results.csv"  "--full-pool"
run voxtral librispeech_original "$LIBRI" "$VC/exact_shapley_original/exact_shapley_results.csv"            ""
run voxtral tts_original         "$TTS"   "$VC/exact_shapley_ttsorig/exact_shapley_results.csv"             "--full-pool"

echo "=================================================================="
echo ">> ALL LOGPROB CONDITIONS DONE ($(date '+%H:%M:%S'))"
echo "=================================================================="

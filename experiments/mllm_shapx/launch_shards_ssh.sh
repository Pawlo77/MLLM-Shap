#!/usr/bin/env bash
# Fan-out mllm_shapx shards over SSH with a shared MLflow tracking URI.
#
# Usage:
#   export MLFLOW_TRACKING_URI="http://mlflow.example.com:5050"
#   export REPO_ROOT="$HOME/projects/MLLM-Shap"   # optional
#   ./launch_shards_ssh.sh [options] hosts.txt /path/to/config.json
#
# Options:
#   --resume            Pass --resume to each shard (default: enabled)
#   --no-resume         Do not pass --resume
#   --max-samples N     Override selection.max_samples on each shard
#
# hosts.txt: one SSH target per line (user@host or host). Lines starting with # are skipped.
# Shard index matches the line order (0 .. N-1).

set -euo pipefail

# ─── Defaults ──────────────────────────────────────────────────────────────────

RESUME=true
MAX_SAMPLES=""

# ─── Argument parsing ──────────────────────────────────────────────────────────

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)       RESUME=true; shift ;;
    --no-resume)    RESUME=false; shift ;;
    --max-samples)  MAX_SAMPLES="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^$/s/^# \?//p' "$0"
      exit 0
      ;;
    -*)             echo "unknown option: $1" >&2; exit 2 ;;
    *)              POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ ${#POSITIONAL[@]} -lt 2 ]]; then
  echo "usage: $0 [options] hosts.txt /path/to/config.json" >&2
  exit 2
fi

HOSTS_FILE="${POSITIONAL[0]}"
CONFIG_PATH="${POSITIONAL[1]}"

if [[ ! -f "$HOSTS_FILE" ]]; then
  echo "hosts file not found: $HOSTS_FILE" >&2
  exit 2
fi

mapfile -t HOSTS < <(grep -v '^[[:space:]]*#' "$HOSTS_FILE" | grep -v '^[[:space:]]*$' || true)
NUM_SHARDS="${#HOSTS[@]}"
if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "no hosts in $HOSTS_FILE" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-$HOME/projects/MLLM-Shap}"

if [[ -z "${MLFLOW_TRACKING_URI:-}" ]]; then
  echo "warning: MLFLOW_TRACKING_URI is not set." >&2
fi

# ─── Build CLI flags ──────────────────────────────────────────────────────────

CLI_EXTRA=""
[[ "$RESUME" = true ]] && CLI_EXTRA+=" --resume"
[[ -n "$MAX_SAMPLES" ]] && CLI_EXTRA+=" --max-samples $MAX_SAMPLES"

# ─── Launch shards ────────────────────────────────────────────────────────────

for i in "${!HOSTS[@]}"; do
  H="${HOSTS[$i]}"
  echo "==> shard $i/$NUM_SHARDS on $H"
  # shellcheck disable=SC2029
  ssh -o BatchMode=yes "$H" \
    "export MLFLOW_TRACKING_URI=$(printf '%q' "${MLFLOW_TRACKING_URI:-}") TOKENIZERS_PARALLELISM=false; \
     cd $(printf '%q' "$REPO_ROOT") && \
     uv run python -m experiments.mllm_shapx.src.cli run \
       --config $(printf '%q' "$CONFIG_PATH") \
       --shard-index $i --num-shards $NUM_SHARDS$CLI_EXTRA" &
done

wait
echo "all shards finished."

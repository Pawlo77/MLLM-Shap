#!/usr/bin/env bash
# Run all JSON configs in a directory with retry-on-failure and structured logs.
# Usage examples:
#   ./run_configs.sh -c experiments/configs/package_grid -l ./logs --resume --repeat
#   ./run_configs.sh -c experiments/configs/package_grid --no-resume --no-repeat

set -u

SCRIPT_NAME=$(basename "$0")

print_usage() {
  cat <<-USAGE
Usage: $SCRIPT_NAME -c <configs_dir> [-l <log_dir>] [--resume|--no-resume] [--repeat|--no-repeat]

Options:
  -c, --configs    Directory containing JSON config files (required)
  -l, --logdir     Directory to write logs into (default: ./logs)
  --resume         Pass --resume to the command (default: enabled)
  --no-resume      Do not pass --resume
  --repeat         If a run fails, retry endlessly until success (default: enabled)
  --no-repeat      If a run fails, do not retry and proceed to next config
  -h, --help       Show this help message
USAGE
}

# Defaults
CONFIGS_DIR=""
LOG_DIR="./logs"
RESUME=true
REPEAT_ON_FAIL=true

# Simple CLI parsing for long and short options
if [ "$#" -eq 0 ]; then
  print_usage
  exit 1
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    -c|--configs)
      CONFIGS_DIR="$2"
      shift 2
      ;;
    -l|--logdir)
      LOG_DIR="$2"
      shift 2
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --no-resume)
      RESUME=false
      shift
      ;;
    --repeat)
      REPEAT_ON_FAIL=true
      shift
      ;;
    --no-repeat)
      REPEAT_ON_FAIL=false
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      print_usage
      exit 2
      ;;
  esac
done

if [ -z "$CONFIGS_DIR" ]; then
  echo "Error: configs directory is required (-c)."
  print_usage
  exit 2
fi

if [ ! -d "$CONFIGS_DIR" ]; then
  echo "Error: configs directory '$CONFIGS_DIR' does not exist or is not a directory."
  exit 2
fi

mkdir -p "$LOG_DIR"

# Create run session directory with timestamp
SESSION_DIR="$LOG_DIR/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SESSION_DIR"

echo "Configs dir: $CONFIGS_DIR"
echo "Logs dir: $SESSION_DIR"
echo "Resume flag: $RESUME"
echo "Repeat-on-fail: $REPEAT_ON_FAIL"

shopt -s nullglob

for cfg in "$CONFIGS_DIR"/*.json; do
  cfg_basename=$(basename -- "$cfg")
  logfile="$SESSION_DIR/${cfg_basename%.json}.log"

  echo "----------------------------------------" | tee -a "$logfile"
  echo "Config: $cfg" | tee -a "$logfile"
  echo "Logfile: $logfile" | tee -a "$logfile"
  echo "Start time: $(date --iso-8601=seconds)" | tee -a "$logfile"

  attempt=1
  while true; do
    echo "\n[${cfg_basename}] Attempt $attempt: $(date --iso-8601=seconds)" | tee -a "$logfile"

    # Build command
    CMD=(uv run python -m mllm_shapx.cli run --config "$cfg")
    if [ "$RESUME" = true ]; then
      CMD+=(--resume)
    fi

    # Run and capture exit code while tee-ing output to logfile
    ("${CMD[@]}" 2>&1) | tee -a "$logfile"
    rc=${PIPESTATUS[0]}

    if [ "$rc" -eq 0 ]; then
      echo "[${cfg_basename}] Completed successfully (exit code 0)" | tee -a "$logfile"
      echo "End time: $(date --iso-8601=seconds)" | tee -a "$logfile"
      break
    else
      echo "[${cfg_basename}] Failed with exit code $rc" | tee -a "$logfile"
      if [ "$REPEAT_ON_FAIL" = true ]; then
        attempt=$((attempt+1))
        echo "Retrying after 5s..." | tee -a "$logfile"
        sleep 5
        continue
      else
        echo "Not retrying. Moving to next config." | tee -a "$logfile"
        break
      fi
    fi
  done
done

echo "All configs processed. Logs in: $SESSION_DIR"

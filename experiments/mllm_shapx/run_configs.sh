#!/usr/bin/env bash
# Run JSON experiment configs with retry-on-failure, structured logs, and summary.
# Works on both Linux (GNU) and macOS (BSD).
#
# Usage examples:
#   ./run_configs.sh -c configs/package_grid
#   ./run_configs.sh -c configs/package_grid -l ./logs --max-retries 5
#   ./run_configs.sh -c "configs/package_grid/*.json" --dry-run
#   ./run_configs.sh -c configs/package_grid --no-resume --no-repeat

set -uo pipefail

SCRIPT_NAME=$(basename "$0")

# ─── Colors (disabled if not a terminal) ───────────────────────────────────────

if [[ -t 1 ]]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
  CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; RESET=''
fi

# ─── Helpers ───────────────────────────────────────────────────────────────────

iso_now() {
  # Portable ISO-8601 timestamp (works on both GNU and BSD date)
  date '+%Y-%m-%dT%H:%M:%S%z'
}

log() { printf "${CYAN}▶${RESET} %s\n" "$*"; }
warn() { printf "${YELLOW}⚠${RESET} %s\n" "$*" >&2; }
err() { printf "${RED}✖${RESET} %s\n" "$*" >&2; }

print_usage() {
  cat <<USAGE
Usage: $SCRIPT_NAME -c <configs> [options]

Options:
  -c, --configs      Directory or glob pattern for JSON configs (required)
  -l, --logdir       Base directory for logs (default: ./logs)
  --resume           Pass --resume to the CLI (default: enabled)
  --no-resume        Do not pass --resume
  --repeat           Retry failed runs (default: enabled)
  --no-repeat        Skip retries on failure
  --max-retries N    Maximum retry attempts per config (default: unlimited, 0=no limit)
  --retry-delay N    Seconds to wait between retries (default: 5)
  --dry-run          Print commands without executing
  -h, --help         Show this help message

Examples:
  $SCRIPT_NAME -c configs/package_grid
  $SCRIPT_NAME -c "configs/**/*.json" --max-retries 3
  $SCRIPT_NAME -c configs/package_grid --dry-run
USAGE
}

# ─── Defaults ──────────────────────────────────────────────────────────────────

CONFIGS_INPUT=""
LOG_DIR="./logs"
RESUME=true
REPEAT_ON_FAIL=true
MAX_RETRIES=0  # 0 = unlimited
RETRY_DELAY=5
DRY_RUN=false

# ─── Argument parsing ─────────────────────────────────────────────────────────

if [[ $# -eq 0 ]]; then
  print_usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--configs)     CONFIGS_INPUT="$2"; shift 2 ;;
    -l|--logdir)      LOG_DIR="$2"; shift 2 ;;
    --resume)         RESUME=true; shift ;;
    --no-resume)      RESUME=false; shift ;;
    --repeat)         REPEAT_ON_FAIL=true; shift ;;
    --no-repeat)      REPEAT_ON_FAIL=false; shift ;;
    --max-retries)    MAX_RETRIES="$2"; shift 2 ;;
    --retry-delay)    RETRY_DELAY="$2"; shift 2 ;;
    --dry-run)        DRY_RUN=true; shift ;;
    -h|--help)        print_usage; exit 0 ;;
    *)                err "Unknown argument: $1"; print_usage; exit 2 ;;
  esac
done

if [[ -z "$CONFIGS_INPUT" ]]; then
  err "Configs directory or pattern is required (-c)."
  print_usage
  exit 2
fi

# ─── Resolve config files ─────────────────────────────────────────────────────

shopt -s nullglob

CONFIG_FILES=()
if [[ -d "$CONFIGS_INPUT" ]]; then
  # Directory: collect all .json files (non-recursive)
  for f in "$CONFIGS_INPUT"/*.json; do
    CONFIG_FILES+=("$f")
  done
else
  # Glob pattern or single file
  for f in $CONFIGS_INPUT; do
    [[ -f "$f" ]] && CONFIG_FILES+=("$f")
  done
fi

if [[ ${#CONFIG_FILES[@]} -eq 0 ]]; then
  err "No JSON config files found matching: $CONFIGS_INPUT"
  exit 2
fi

# Sort for deterministic order
IFS=$'\n' CONFIG_FILES=($(sort <<<"${CONFIG_FILES[*]}")); unset IFS

# ─── Session setup ─────────────────────────────────────────────────────────────

SESSION_DIR="$LOG_DIR/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SESSION_DIR"

# ─── Trap for graceful interrupt ──────────────────────────────────────────────

INTERRUPTED=false
trap 'INTERRUPTED=true; warn "Interrupted — finishing current config..."' INT TERM

# ─── Print plan ───────────────────────────────────────────────────────────────

printf "\n${BOLD}═══ Run Plan ═══${RESET}\n"
log "Configs:       ${#CONFIG_FILES[@]} files from $CONFIGS_INPUT"
log "Logs:          $SESSION_DIR"
log "Resume:        $RESUME"
log "Retry:         $REPEAT_ON_FAIL (max=$( [[ $MAX_RETRIES -eq 0 ]] && echo "∞" || echo "$MAX_RETRIES" ), delay=${RETRY_DELAY}s)"
[[ "$DRY_RUN" = true ]] && log "Mode:          ${YELLOW}DRY RUN${RESET}"
printf "${BOLD}════════════════${RESET}\n\n"

# ─── Main loop ────────────────────────────────────────────────────────────────

PASSED=0
FAILED=0
SKIPPED=0
declare -a FAILED_CONFIGS=()

for cfg in "${CONFIG_FILES[@]}"; do
  [[ "$INTERRUPTED" = true ]] && { SKIPPED=$((${#CONFIG_FILES[@]} - PASSED - FAILED)); break; }

  cfg_basename=$(basename -- "$cfg")
  logfile="$SESSION_DIR/${cfg_basename%.json}.log"

  printf "${BOLD}────────────────────────────────────────${RESET}\n"
  log "Config: $cfg"
  log "Start:  $(iso_now)"

  # Build command
  CMD=(uv run python -m mllm_shapx.src.cli run --config "$cfg")
  [[ "$RESUME" = true ]] && CMD+=(--resume)

  if [[ "$DRY_RUN" = true ]]; then
    log "${YELLOW}[dry-run]${RESET} ${CMD[*]}"
    PASSED=$((PASSED + 1))
    continue
  fi

  {
    echo "# Config: $cfg"
    echo "# Command: ${CMD[*]}"
    echo "# Started: $(iso_now)"
    echo ""
  } >> "$logfile"

  attempt=1
  while true; do
    [[ "$INTERRUPTED" = true ]] && break

    printf "  ${CYAN}attempt %d${RESET} " "$attempt"
    [[ $MAX_RETRIES -gt 0 ]] && printf "/ %d " "$MAX_RETRIES"
    printf "@ %s\n" "$(iso_now)"

    echo "--- Attempt $attempt @ $(iso_now) ---" >> "$logfile"

    # Run and capture exit code (pipefail ensures we get the real exit code)
    "${CMD[@]}" 2>&1 | tee -a "$logfile"
    rc=${PIPESTATUS[0]}

    if [[ $rc -eq 0 ]]; then
      printf "  ${GREEN}✔ Success${RESET} (%s)\n" "$(iso_now)"
      echo "--- SUCCESS @ $(iso_now) ---" >> "$logfile"
      PASSED=$((PASSED + 1))
      break
    else
      printf "  ${RED}✖ Failed (exit %d)${RESET}\n" "$rc"
      echo "--- FAILED (exit $rc) @ $(iso_now) ---" >> "$logfile"

      if [[ "$REPEAT_ON_FAIL" = true ]]; then
        if [[ $MAX_RETRIES -gt 0 && $attempt -ge $MAX_RETRIES ]]; then
          warn "Max retries ($MAX_RETRIES) reached for $cfg_basename"
          FAILED=$((FAILED + 1))
          FAILED_CONFIGS+=("$cfg_basename")
          break
        fi
        attempt=$((attempt + 1))
        log "Retrying in ${RETRY_DELAY}s..."
        sleep "$RETRY_DELAY"
      else
        FAILED=$((FAILED + 1))
        FAILED_CONFIGS+=("$cfg_basename")
        break
      fi
    fi
  done
done

# ─── Summary ──────────────────────────────────────────────────────────────────

printf "\n${BOLD}═══ Summary ═══${RESET}\n"
printf "  ${GREEN}Passed:${RESET}  %d\n" "$PASSED"
printf "  ${RED}Failed:${RESET}  %d\n" "$FAILED"
[[ $SKIPPED -gt 0 ]] && printf "  ${YELLOW}Skipped:${RESET} %d (interrupted)\n" "$SKIPPED"
printf "  Logs:    %s\n" "$SESSION_DIR"

if [[ ${#FAILED_CONFIGS[@]} -gt 0 ]]; then
  printf "\n  ${RED}Failed configs:${RESET}\n"
  for name in "${FAILED_CONFIGS[@]}"; do
    printf "    - %s\n" "$name"
  done
fi

printf "${BOLD}════════════════${RESET}\n"

# Exit with non-zero if any failed
[[ $FAILED -gt 0 || "$INTERRUPTED" = true ]] && exit 1
exit 0

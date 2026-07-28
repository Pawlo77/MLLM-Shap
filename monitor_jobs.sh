#!/usr/bin/env bash
# One-shot status dashboard for the AAAI-27 SGPA GPU jobs.
#
# Usage:
#   ./monitor_jobs.sh            # print status once
#   watch -n 5 ./monitor_jobs.sh # refresh every 5s
cd "$(dirname "$0")"

QDIR="logs/queue"
BOLD=$'\e[1m'; DIM=$'\e[2m'; RST=$'\e[0m'

echo "${BOLD}=== GPU ===${RST}"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"

echo
echo "${BOLD}=== Running python jobs ===${RST}"
found=$(ps -eo pid,etime,%cpu,%mem,args 2>/dev/null \
  | grep -E -- "-m experiments\.(faithfulness|mllm_shapx)" \
  | grep -v -E "zsh -c|grep -E" \
  | awk '{
      cmd=substr($0, index($0,$5));
      sub(/.*\/\.venv\/bin\/python[^ ]* /, "python ", cmd);
      sub(/ *--run-dir /, " ", cmd);
      printf "  pid=%-7s up=%-9s cpu=%-6s mem=%-5s %s\n", $1,$2,$3,$4, cmd
    }')
if [ -n "$found" ]; then echo "$found"; else echo "  (none)"; fi

echo
echo "${BOLD}=== Queue status ===${RST}"
shown=0
for q in "$QDIR" logs/queue_voxtral; do
  if [ -f "$q/status.tsv" ]; then
    echo "${DIM}[$q]${RST}"
    { echo "JOB STATE SINCE EXTRA"; sort "$q/status.tsv"; } | column -t
    shown=1
  fi
done
[ "$shown" = "0" ] && echo "  (queue not started — run ./run_queue.sh)"

echo
echo "${BOLD}=== Progress (latest line per log) ===${RST}"
progress() { # logfile label
  local f="$1" lbl="$2"
  [ -f "$f" ] || return 0
  local last done_flag
  last="$(tr '\r' '\n' < "$f" | grep -oE '[0-9]+/[0-9]+ \[[^]]*\]' | tail -1)"
  if grep -q '"completed_samples"' "$f" 2>/dev/null; then
    done_flag=" ${DIM}[summary written]${RST}"
  else
    done_flag=""
  fi
  printf "  %-26s %s%s\n" "$lbl" "${last:-starting/loading...}" "$done_flag"
}

# queued jobs (both queues)
for q in "$QDIR" logs/queue_voxtral; do
  [ -d "$q" ] || continue
  for f in "$q"/*.log; do
    [ -e "$f" ] || continue
    progress "$f" "$(basename "$f" .log)"
  done
done

echo
echo "${BOLD}=== Outputs on disk (completed samples) ===${RST}"
for d in experiments/faithfulness/outputs/qwen_exact_shapley*; do
  [ -d "$d" ] || continue
  csv="$d/qwen_exact_shapley_results.csv"
  if [ -f "$csv" ]; then
    n=$(tail -n +2 "$csv" | cut -d, -f1 | sort -u | wc -l)
    printf "  %-42s %s samples\n" "$(basename "$d")" "$n"
  fi
done

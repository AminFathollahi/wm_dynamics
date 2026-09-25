#!/usr/bin/env bash
# Live status display for the four information-benchmark seed workers.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="$ROOT/results/info_benchmark_seeds"
INTERVAL="${1:-10}"

if ! [[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Usage: %s [refresh-seconds]\n' "$0" >&2
  exit 2
fi

while true; do
  clear
  printf 'Information benchmark monitor — %s (refresh: %ss)\n\n' "$(date --iso-8601=seconds)" "$INTERVAL"

  printf 'Workers\n'
  printf '%-5s %-9s %-11s %-7s %-7s %-10s %-5s %s\n' \
    'seed' 'PID' 'elapsed' 'CPU%' 'MEM%' 'RSS MiB' 'stat' 'latest log line'
  for seed in 0 1 2 3; do
    pid_file="$RESULTS/seed_${seed}.pid"
    pid='—'
    elapsed='—'
    cpu='—'
    mem='—'
    rss='—'
    stat='—'
    if [[ -f "$pid_file" ]]; then
      pid="$(tr -d '[:space:]' < "$pid_file")"
      row="$(ps -p "$pid" -o etime=,pcpu=,pmem=,rss=,stat= 2>/dev/null || true)"
      if [[ -n "$row" ]]; then
        read -r elapsed cpu mem rss_kb stat <<< "$row"
        rss="$((rss_kb / 1024))"
      else
        stat='stopped'
      fi
    else
      stat='no PID file'
    fi
    latest="$(awk 'NF {line=$0} END {print line}' "$RESULTS/seed_${seed}.log" 2>/dev/null)"
    latest="${latest:0:100}"
    printf '%-5s %-9s %-11s %-7s %-7s %-10s %-5s %s\n' \
      "$seed" "$pid" "$elapsed" "$cpu" "$mem" "$rss" "$stat" "$latest"
  done

  printf '\nHost memory\n'
  free -h

  printf '\nGPU\n'
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory \
      --format=csv,noheader 2>&1
    printf '\nGPU compute processes\n'
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 || true
  else
    printf 'nvidia-smi is unavailable\n'
  fi

  printf '\nCtrl-C exits the monitor; it does not stop any worker.\n'
  sleep "$INTERVAL"
done

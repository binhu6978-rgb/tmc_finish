#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CIFAR100_ROOT:?Set CIFAR100_ROOT}"
mkdir -p logs
for d in results/performance_300/b0 results/performance_300/b1; do
  if [[ -e "$d" ]]; then echo "Output already exists: $d"; exit 2; fi
done
# Read-only memory query. No extra smoke runs/tests or automatic parameter changes.
"${PYTHON:-python}" -u -B -m src.resources --gpu 1 --performance
stamp=$(date +%Y%m%d_%H%M%S)
cleanup() { for pid in $(jobs -pr); do kill -TERM "$pid" 2>/dev/null || true; done; }
trap 'cleanup; exit 130' INT TERM
bash scripts/run_b0_gpu1.sh >"logs/b0_300_${stamp}.log" 2>&1 & pa=$!
bash scripts/run_b1_gpu1.sh >"logs/b1_300_${stamp}.log" 2>&1 & pb=$!
printf 'b0_pid=%s\nb1_pid=%s\n' "$pa" "$pb" >"logs/pids_300_${stamp}.txt"
if wait "$pa"; then ea=0; else ea=$?; fi
if wait "$pb"; then eb=0; else eb=$?; fi
trap - INT TERM
printf 'b0_exit=%s\nb1_exit=%s\n' "$ea" "$eb" >"logs/exits_300_${stamp}.txt"
if [[ "$ea" -ne 0 || "$eb" -ne 0 ]]; then echo "EXPERIMENT_FAILED: b0=$ea b1=$eb"; exit 1; fi
bash scripts/analyze_performance_300.sh

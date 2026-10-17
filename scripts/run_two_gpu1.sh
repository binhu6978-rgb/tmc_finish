#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CIFAR100_ROOT:?Set CIFAR100_ROOT}"
mkdir -p logs
# No silent serial fallback. Includes short, full-local-budget memory measurements.
"${PYTHON:-python}" -u -B -m src.resources --gpu 1
for d in results/original_semantic results/fixed_class_semantic; do
  if [[ -e "$d" ]]; then echo "Output already exists: $d; choose/archive outputs explicitly"; exit 2; fi
done
stamp=$(date +%Y%m%d_%H%M%S)
cleanup() {
  # jobs reports only live children of THIS shell, avoiding stale/reused stored PIDs.
  for pid in $(jobs -pr); do kill -TERM "$pid" 2>/dev/null || true; done
}
trap 'cleanup; exit 130' INT TERM
bash scripts/run_original_gpu1.sh >"logs/original_${stamp}.log" 2>&1 & pa=$!
bash scripts/run_fixed_gpu1.sh >"logs/fixed_${stamp}.log" 2>&1 & pb=$!
printf 'original_pid=%s\nfixed_pid=%s\n' "$pa" "$pb" >"logs/pids_${stamp}.txt"
if wait "$pa"; then ea=0; else ea=$?; fi
if wait "$pb"; then eb=0; else eb=$?; fi
printf 'original_exit=%s\nfixed_exit=%s\n' "$ea" "$eb" >"logs/exits_${stamp}.txt"
trap - INT TERM
if [[ "$ea" -ne 0 || "$eb" -ne 0 ]]; then
  echo "EXPERIMENT_FAILED: original=$ea fixed=$eb; inspect logs and status.json"; exit 1
fi
echo "Both50-round runs complete. Run bash scripts/analyze_results.sh"

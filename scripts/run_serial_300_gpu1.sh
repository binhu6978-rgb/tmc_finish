#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for d in results/performance_300/b0 results/performance_300/b1; do
  if [[ -e "$d" ]]; then echo "Output already exists: $d"; exit 2; fi
done
mkdir -p logs
stamp=$(date +%Y%m%d_%H%M%S)
bash scripts/run_b0_gpu1.sh >"logs/b0_300_${stamp}.log" 2>&1
bash scripts/run_b1_gpu1.sh >"logs/b1_300_${stamp}.log" 2>&1
bash scripts/analyze_performance_300.sh

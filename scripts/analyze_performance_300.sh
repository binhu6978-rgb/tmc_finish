#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python}" -u -B -m src.performance_analysis \
  --b0 "${1:-results/performance_300/b0}" --b1 "${2:-results/performance_300/b1}" \
  --output "${3:-results/performance_300/comparison}"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Explicit user-selected serial mode, never selected by the parallel launcher.
bash scripts/check_gpu1.sh
bash scripts/run_original_gpu1.sh
bash scripts/run_fixed_gpu1.sh

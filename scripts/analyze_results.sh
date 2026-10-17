#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python}" -B -m src.analyze --original "${1:-results/original_semantic}" --fixed "${2:-results/fixed_class_semantic}" --output "${3:-results/comparison}"

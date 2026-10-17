#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CIFAR100_ROOT:?Set CIFAR100_ROOT}"
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "${PYTHON:-python}" -u -B -m src.performance --arm b1 --output "${1:-results/performance_300/b1}"

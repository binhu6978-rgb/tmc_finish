#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CIFAR100_ROOT:?Set CIFAR100_ROOT}"
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "${PYTHON:-python}" -u -B -m src.main --arm fixed_class --output "${1:-results/fixed_class_semantic}"

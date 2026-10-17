#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CIFAR100_ROOT:?Set CIFAR100_ROOT to the directory containing cifar-100-python}"
"${PYTHON:-python}" -u -B -m src.resources --gpu 1

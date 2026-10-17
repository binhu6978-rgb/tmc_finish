#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python}" -c 'import torch,torchvision,numpy,yaml; print("torch",torch.__version__,"torchvision",torchvision.__version__); print("CIFAR100_ROOT must point to the directory containing cifar-100-python")'
"${PYTHON:-python}" -B -m unittest discover -s tests -p test_probe.py -v

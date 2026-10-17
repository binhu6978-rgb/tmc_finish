#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python}" -B -m src.resources --gpu 1 --inspect-only

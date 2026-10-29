#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/python app.py "$@"

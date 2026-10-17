#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PYTHON="${PYTHON:-/home/public/tbiom/.venv/bin/python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p outputs/logs
bash scripts/check_dataset_size.sh ../../datasets/market1501 > outputs/logs/dataset_size.json || true
exec "$PYTHON" -u -m poanet.train --config configs/default.yaml --device cuda 2>&1 | tee outputs/logs/train_full.log

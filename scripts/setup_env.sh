#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MONO="$(cd "$ROOT/../../.." && pwd)"
PYTHON="${PYTHON:-$MONO/.venv/bin/python}"
"$PYTHON" -m pip install -q -r "$ROOT/requirements.txt"
echo "OK: deps in $PYTHON"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/python:${PYTHONPATH:-}"

OUT_DIR="${OUT_DIR:-profile_results/correctness}"
mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

python3 benchmark/python/tests/test_pybind_vs_official_correctness.py \
  --dtype "${DTYPE:-fp16}" \
  --input_mode "${INPUT_MODE:-random}" \
  --seed "${SEED:-1234}" \
  --cases "${CASES:-2x4x128x128,2x4x256x128,2x4x512x128}" \
  --official_path "${OFFICIAL_PATH:-$HOME/flash-attention}" \
  --out_json "$OUT_DIR/pybind_vs_official_${TS}.json" \
  --out_csv "$OUT_DIR/pybind_vs_official_${TS}.csv" \
  "$@"


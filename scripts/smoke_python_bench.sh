#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/python:${PYTHONPATH:-}"

echo "[1/4] python syntax check"
python3 -m py_compile \
  benchmark/python/benchmark_pybind.py \
  benchmark/python/tests/test_pybind_interface.py \
  benchmark/python/tests/test_pybind_correctness.py

echo "[2/4] cuda availability check"
python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for smoke benchmark")
print("cuda ok:", torch.cuda.get_device_name(0))
PY

echo "[3/4] pybind interface smoke"
python3 benchmark/python/tests/test_pybind_interface.py

echo "[4/4] correctness + tiny benchmark"
./scripts/run_correctness_gate.sh \
  --cases "2x4x128x64,2x4x256x64,2x4x128x128,2x4x256x128" \
  --fail_fast
python3 benchmark/python/benchmark_pybind.py \
  --d_heads 64 \
  --seq_lens 512 \
  --warmups 2 \
  --repeats 5 \
  --csv

echo "smoke benchmark done"

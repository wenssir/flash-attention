#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/python:${PYTHONPATH:-}"

SEQ_LENS="${SEQ_LENS:-4096}"
D_HEADS="${D_HEADS:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_HEADS="${N_HEADS:-16}"
WARMUPS="${WARMUPS:-10}"
REPEATS="${REPEATS:-50}"
BASELINE="${BASELINE:-pytorch}"
PYBIND_DTYPE="${PYBIND_DTYPE:-fp16}"
OFFICIAL_DTYPE="${OFFICIAL_DTYPE:-fp16}"
OFFICIAL_PYTORCH_DTYPE="${OFFICIAL_PYTORCH_DTYPE:-match}"
INPUT_MODE="${INPUT_MODE:-random}"
SEED="${SEED:-1234}"
REPLAY_DIR="${REPLAY_DIR:-benchmark/data/replay}"
REPLAY_FILE="${REPLAY_FILE:-}"
RUN_OFFICIAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seq_lens) SEQ_LENS="$2"; shift 2 ;;
    --d_heads) D_HEADS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --n_heads) N_HEADS="$2"; shift 2 ;;
    --warmups) WARMUPS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --pybind_dtype) PYBIND_DTYPE="$2"; shift 2 ;;
    --official_dtype) OFFICIAL_DTYPE="$2"; shift 2 ;;
    --official_pytorch_dtype) OFFICIAL_PYTORCH_DTYPE="$2"; shift 2 ;;
    --input_mode) INPUT_MODE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --replay_dir) REPLAY_DIR="$2"; shift 2 ;;
    --replay_file) REPLAY_FILE="$2"; shift 2 ;;
    --run_official) RUN_OFFICIAL=1; shift 1 ;;
    --skip_official) RUN_OFFICIAL=0; shift 1 ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--seq_lens 4096] [--d_heads 128] [--batch_size 8] [--n_heads 16] [--warmups 10] [--repeats 50] [--baseline pytorch|fa2_pybind] [--pybind_dtype fp32|fp16] [--official_dtype fp16|bf16] [--official_pytorch_dtype match|fp32] [--input_mode random|structured|stress|replay] [--seed 1234] [--replay_dir benchmark/data/replay] [--replay_file data.pt] [--run_official|--skip_official]"
      exit 1
      ;;
  esac
done

mkdir -p profile_results
TS="$(date +%Y%m%d_%H%M%S)"
PYBIND_CSV="profile_results/pybind_${TS}.csv"
OFFICIAL_CSV="profile_results/official_${TS}.csv"

echo "[1/3] Running pybind benchmark..."
python3 benchmark/python/benchmark_pybind.py \
  --seq_lens "$SEQ_LENS" \
  --d_heads "$D_HEADS" \
  --batch_size "$BATCH_SIZE" \
  --n_heads "$N_HEADS" \
  --warmups "$WARMUPS" \
  --repeats "$REPEATS" \
  --baseline "$BASELINE" \
  --dtype "$PYBIND_DTYPE" \
  --input_mode "$INPUT_MODE" \
  --seed "$SEED" \
  --replay_dir "$REPLAY_DIR" \
  --replay_file "$REPLAY_FILE" \
  --csv \
  --out_csv "$PYBIND_CSV"

if [[ "$RUN_OFFICIAL" -eq 1 ]]; then
  echo "[2/3] Running official benchmark..."
  if python3 benchmark/python/benchmark_official_flashattn.py \
      --seq_lens "$SEQ_LENS" \
      --d_heads "$D_HEADS" \
      --batch_size "$BATCH_SIZE" \
      --n_heads "$N_HEADS" \
      --warmups "$WARMUPS" \
      --repeats "$REPEATS" \
      --dtype "$OFFICIAL_DTYPE" \
      --pytorch_dtype "$OFFICIAL_PYTORCH_DTYPE" \
      --input_mode "$INPUT_MODE" \
      --seed "$SEED" \
      --replay_dir "$REPLAY_DIR" \
      --replay_file "$REPLAY_FILE" \
      --baseline pytorch \
      --csv \
      --out_csv "$OFFICIAL_CSV"; then
    echo "[3/3] Merging benchmark CSV files..."
    python3 scripts/merge_bench_csv.py \
      --pybind_csv "$PYBIND_CSV" \
      --official_csv "$OFFICIAL_CSV" \
      --out_csv "profile_results/merged_${TS}.csv"
  else
    echo "Official benchmark failed. Pybind results are still saved at: $PYBIND_CSV"
    exit 1
  fi
else
  echo "[2/2] Skipped official benchmark. Pybind results saved at: $PYBIND_CSV"
fi

echo "Done."

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

DEFAULT_CONFIG="$SCRIPT_DIR/local_perf_gate.example"
CONFIG_FILE=""

# Load default config first so CLI flags can override it.
if [[ -f "$DEFAULT_CONFIG" ]]; then
  # shellcheck disable=SC1090
  source "$DEFAULT_CONFIG"
fi

BUILD_DIR="${BUILD_DIR:-build}"
OUT_DIR="${OUT_DIR:-profile_results/gate}"
SEQ_LEN="${SEQ_LEN:-4096}"
D_HEAD="${D_HEAD:-128}"
BATCH_SIZE="${BATCH_SIZE:-16}"
N_HEADS="${N_HEADS:-16}"
WARMUPS="${WARMUPS:-10}"
REPEATS="${REPEATS:-50}"
DTYPE="${DTYPE:-fp16}"
INPUT_MODE="${INPUT_MODE:-random}"
SEED="${SEED:-1234}"
CUDA_ARCH="${CUDA_ARCH:-}"
RUN_NCU="${RUN_NCU:-0}"
CLEAN_EXT="${CLEAN_EXT:-1}"
NCU_CMD="${NCU_CMD:-ncu}"
GATE_FILE="${GATE_FILE:-profile_results/gate_history.csv}"
PREFIX="${PREFIX:-gate}"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_local_perf_gate.sh [options]

Options:
  --build_dir <dir>          CMake build dir (default: build)
  --config <path>            Source config file (KEY=VALUE), optional.
                             Default: scripts/local_perf_gate.example (if exists)
  --out_dir <dir>            Output dir (default: profile_results/gate)
  --prefix <name>            Run directory prefix (default: gate)
  --seq_len <int>            Sequence length (default: 4096)
  --d_head <int>             Head dim (default: 128)
  --batch_size <int>         Batch size (default: 16)
  --n_heads <int>            Number of heads (default: 16)
  --warmups <int>            Warmup iterations for python benchmark (default: 10)
  --repeats <int>            Repeat iterations for python benchmark (default: 50)
  --dtype <fp16|fp32>        Pybind benchmark input dtype (default: fp16)
  --input_mode <mode>        random|structured|stress|replay (default: random)
  --seed <int>               Input seed (default: 1234)
  --cuda_arch <arch>         e.g. 80, 120 (optional)
  --run_ncu <0|1>            Ignored for now (NCU disabled in this gate script)
  --clean_ext <0|1>          Remove torch extension cache first (default: 1)
  --ncu_cmd <cmd>            ncu runner command (default: ncu)
  --gate_file <path>         Aggregate CSV path (default: profile_results/gate_history.csv)
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build_dir) BUILD_DIR="$2"; shift 2 ;;
    --config)
      CONFIG_FILE="$2"
      if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "Config file not found: $CONFIG_FILE"
        exit 1
      fi
      # shellcheck disable=SC1090
      source "$CONFIG_FILE"
      shift 2
      ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --seq_len) SEQ_LEN="$2"; shift 2 ;;
    --d_head) D_HEAD="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --n_heads) N_HEADS="$2"; shift 2 ;;
    --warmups) WARMUPS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --input_mode) INPUT_MODE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --cuda_arch) CUDA_ARCH="$2"; shift 2 ;;
    --run_ncu) RUN_NCU="$2"; shift 2 ;;
    --clean_ext) CLEAN_EXT="$2"; shift 2 ;;
    --ncu_cmd) NCU_CMD="$2"; shift 2 ;;
    --gate_file) GATE_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$OUT_DIR"
mkdir -p "$(dirname "$GATE_FILE")"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUT_DIR/${PREFIX}_${TS}"
mkdir -p "$RUN_DIR"
COMPILE_LOG="$RUN_DIR/compile.log"
PYBENCH_CSV="$RUN_DIR/pybind.csv"
PYBENCH_LOG="$RUN_DIR/pybind.log"
NCU_RAW="$RUN_DIR/ncu.csv"
SUMMARY_CSV="$RUN_DIR/summary.csv"

export PYTHONPATH="$PROJECT_ROOT/python:${PYTHONPATH:-}"

if [[ "$CLEAN_EXT" == "1" ]]; then
  rm -rf "$HOME/.cache/torch_extensions/"*flash_attention_v2_kernels* 2>/dev/null || true
fi

cmake_args=(
  -S .
  -B "$BUILD_DIR"
  -DFA_BUILD_UTILS=ON
  -DFA_BUILD_TESTS=OFF
  -DFA_BUILD_EXAMPLES=OFF
)
if [[ -n "$CUDA_ARCH" ]]; then
  cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
fi

cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" -j --target profile_kernel 2>&1 | tee "$COMPILE_LOG"

python3 benchmark/python/benchmark_pybind.py \
  --seq_lens "$SEQ_LEN" \
  --d_heads "$D_HEAD" \
  --batch_size "$BATCH_SIZE" \
  --n_heads "$N_HEADS" \
  --warmups "$WARMUPS" \
  --repeats "$REPEATS" \
  --dtype "$DTYPE" \
  --input_mode "$INPUT_MODE" \
  --seed "$SEED" \
  --baseline pytorch \
  --csv \
  --out_csv "$PYBENCH_CSV" \
  2>&1 | tee "$PYBENCH_LOG"

if [[ "$RUN_NCU" == "1" ]]; then
  echo "[INFO] --run_ncu is currently disabled in this script; skip NCU."
fi

python3 - "$COMPILE_LOG" "$PYBENCH_CSV" "$NCU_RAW" "$SUMMARY_CSV" "$GATE_FILE" "$TS" "$SEQ_LEN" "$D_HEAD" "$BATCH_SIZE" "$N_HEADS" <<'PY'
import csv
import os
import re
import sys
from statistics import mean

compile_log, py_csv, ncu_raw, summary_csv, gate_file, ts, seq_len, d_head, batch_size, n_heads = sys.argv[1:]

def parse_compile(path: str):
    regs = None
    spills_store = 0
    spills_load = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    # Prefer v3 kernel-specific register line if present.
    v3_block = re.findall(
        r"flash_attention_forward_v3_tensor_core.*?(?:Used\s+(\d+)\s+registers).*?(?=ptxas info\s+: Compiling entry function|$)",
        txt,
        re.S,
    )
    if v3_block:
        try:
            regs = max(int(x) for x in v3_block if x)
        except Exception:
            regs = None
    if regs is None:
        all_regs = [int(x) for x in re.findall(r"Used\s+(\d+)\s+registers", txt)]
        regs = max(all_regs) if all_regs else None

    for s, l in re.findall(r"(\d+)\s+bytes spill stores,\s+(\d+)\s+bytes spill loads", txt):
        spills_store = max(spills_store, int(s))
        spills_load = max(spills_load, int(l))
    return regs, spills_store, spills_load

def parse_pybench(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    tgt = [
        r for r in rows
        if r.get("d_head") == d_head
        and r.get("seq_len") == seq_len
        and r.get("batch_size") == batch_size
        and r.get("n_heads") == n_heads
        and r.get("kernel") in ("pytorch", "fa2_pybind")
    ]
    out = {}
    for r in tgt:
        out[r["kernel"]] = r
    if "fa2_pybind" not in out:
        raise RuntimeError("fa2_pybind row not found in benchmark CSV")
    return out

def try_float(s):
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return None

def parse_ncu(path: str):
    if not os.path.exists(path):
        return None, None, None, None

    metric_values = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "," not in line:
                continue
            cols = [c.strip().strip('"') for c in line.strip().split(",")]
            if len(cols) < 2:
                continue
            # detect metric token in row
            metric = None
            for c in cols:
                if "__" in c and (".sum" in c or ".pct" in c or ".avg" in c):
                    metric = c
                    break
            if metric is None:
                continue
            val = None
            for c in reversed(cols):
                v = try_float(c)
                if v is not None:
                    val = v
                    break
            if val is None:
                continue
            metric_values.setdefault(metric, []).append(val)

    def mget(prefix):
        vals = []
        for k, v in metric_values.items():
            if k.startswith(prefix):
                vals.extend(v)
        return mean(vals) if vals else None

    occ = mget("sm__warps_active.avg.pct_of_peak_sustained_active")
    ld_bytes = mget("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum")
    ld_secs = mget("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
    st_bytes = mget("l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum")
    st_secs = mget("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")

    load_bps = (ld_bytes / ld_secs) if ld_bytes and ld_secs else None
    store_bps = (st_bytes / st_secs) if st_bytes and st_secs else None

    stall_metrics = {
        "barrier": mget("smsp__warp_issue_stalled_barrier_per_warp_active.pct"),
        "long_scoreboard": mget("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"),
        "short_scoreboard": mget("smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct"),
        "math_pipe_throttle": mget("smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"),
        "mio_throttle": mget("smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct"),
        "not_selected": mget("smsp__warp_issue_stalled_not_selected_per_warp_active.pct"),
        "wait": mget("smsp__warp_issue_stalled_wait_per_warp_active.pct"),
    }
    stall_items = [(k, v) for k, v in stall_metrics.items() if v is not None]
    top_stall = max(stall_items, key=lambda kv: kv[1])[0] if stall_items else None
    return occ, load_bps, store_bps, top_stall

regs, spill_store, spill_load = parse_compile(compile_log)
bench = parse_pybench(py_csv)
occ, load_bps, store_bps, top_stall = parse_ncu(ncu_raw)

py = bench["fa2_pybind"]
pt = bench.get("pytorch")

row = {
    "time": ts,
    "commit": py.get("git_commit", ""),
    "gpu": py.get("gpu_name", ""),
    "shape": f"B{batch_size}_H{n_heads}_N{seq_len}_D{d_head}",
    "variant": "fa2_pybind_vs_pytorch",
    "mean_ms_pybind": py.get("mean_ms", ""),
    "mean_ms_pytorch": (pt or {}).get("mean_ms", ""),
    "attn_tflops_pybind": py.get("attn_tflops", ""),
    "attn_tflops_pytorch": (pt or {}).get("attn_tflops", ""),
    "rel_perf_pct": py.get("rel_perf_pct", ""),
    "regs": "" if regs is None else str(regs),
    "spill_store_B": str(spill_store),
    "spill_load_B": str(spill_load),
    "achieved_occ_pct": "" if occ is None else f"{occ:.3f}",
    "global_load_B_per_sector": "" if load_bps is None else f"{load_bps:.3f}",
    "global_store_B_per_sector": "" if store_bps is None else f"{store_bps:.3f}",
    "top_stall": top_stall or "",
}

fields = list(row.keys())
for p in (summary_csv, gate_file):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    exists = os.path.exists(p)
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)

print("Summary:")
for k in fields:
    print(f"  {k}: {row[k]}")
print(f"saved summary: {summary_csv}")
print(f"updated gate history: {gate_file}")
PY

echo ""
echo "Done."
echo "run dir     : $RUN_DIR"
echo "compile log : $COMPILE_LOG"
echo "bench csv   : $PYBENCH_CSV"
echo "summary csv : $SUMMARY_CSV"
echo "gate file   : $GATE_FILE"

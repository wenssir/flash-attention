#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_FILE="$PROJECT_ROOT/scripts/local_config.example"
BLOCK_MS="32,64,128"
PROFILE_SHAPE="16x16x4096x128"
WARMUPS="10"
REPEATS="50"
KEEP_BEST="0"

usage() {
  cat <<'USAGE'
Usage: ./scripts/local/autotune_blockm.sh [options]

Options:
  --config <path>           local run config file (default: scripts/local_config.example)
  --block-ms <list>         comma-separated BlockM candidates (default: 32,64,128)
  --profile-shape BxHxNxD   benchmark shape (default: 16x16x4096x128)
  --warmups <int>           warmup iters (default: 10)
  --repeats <int>           repeat iters (default: 50)
  --keep-best               keep best BlockM in src/common/common.h (default: restore original)
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --block-ms) BLOCK_MS="$2"; shift 2 ;;
    --profile-shape) PROFILE_SHAPE="$2"; shift 2 ;;
    --warmups) WARMUPS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --keep-best) KEEP_BEST="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

COMMON_H="$PROJECT_ROOT/src/common/common.h"
if [[ ! -f "$COMMON_H" ]]; then
  echo "Missing file: $COMMON_H" >&2
  exit 1
fi

ORIG_BLOCK_M="$(sed -n 's/^[[:space:]]*constexpr int V3_BLOCK_M = \([0-9][0-9]*\);/\1/p' "$COMMON_H" | head -n 1)"
if [[ -z "$ORIG_BLOCK_M" ]]; then
  echo "Failed to parse original V3_BLOCK_M from $COMMON_H" >&2
  exit 1
fi

set_block_m() {
  local bm="$1"
  sed -i -E "s/^([[:space:]]*constexpr int V3_BLOCK_M = )[0-9]+(;)$/\1${bm}\2/" "$COMMON_H"
}

restore_original() {
  if [[ "$KEEP_BEST" == "0" ]]; then
    set_block_m "$ORIG_BLOCK_M"
    echo "[autotune] restored V3_BLOCK_M=$ORIG_BLOCK_M"
  fi
}
trap restore_original EXIT

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$PROJECT_ROOT/profile_results/autotune_blockm_${TS}"
mkdir -p "$OUT_DIR"
SUMMARY_CSV="$OUT_DIR/summary.csv"

cat > "$SUMMARY_CSV" <<'CSV'
block_m,run_id,run_dir,mean_ms,attn_tflops,achieved_gflops,throughput_tokens_per_s,est_bw_gbps
CSV

BEST_BM=""
BEST_MS=""

IFS=',' read -r -a BM_LIST <<< "$BLOCK_MS"
for bm_raw in "${BM_LIST[@]}"; do
  bm="$(echo "$bm_raw" | xargs)"
  if [[ -z "$bm" ]]; then
    continue
  fi
  if ! [[ "$bm" =~ ^[0-9]+$ ]]; then
    echo "[autotune] skip invalid BlockM: $bm" >&2
    continue
  fi

  echo "[autotune] testing BlockM=$bm"
  set_block_m "$bm"

  RUN_INFO="$OUT_DIR/runinfo_bm${bm}.json"
  ./scripts/local/run.sh \
    --config "$CONFIG_FILE" \
    --run-name "autotune_bm${bm}" \
    --run-info "$RUN_INFO" \
    --profile-benchmark \
    --skip-ncu \
    --profile-shape "$PROFILE_SHAPE" \
    --warmups "$WARMUPS" \
    --repeats "$REPEATS"

  RUN_DIR="$(python3 - <<'PY' "$RUN_INFO"
import json,sys
obj=json.load(open(sys.argv[1]))
print(obj.get("run_dir",""))
PY
)"
  if [[ -z "$RUN_DIR" ]]; then
    echo "[autotune] missing run_dir for BlockM=$bm" >&2
    continue
  fi

  OURS_CSV="$RUN_DIR/tables/ours_cpp.csv"
  if [[ ! -f "$OURS_CSV" ]]; then
    echo "[autotune] missing ours_cpp.csv for BlockM=$bm: $OURS_CSV" >&2
    continue
  fi

  LINE="$(python3 - <<'PY' "$OURS_CSV"
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1], newline='', encoding='utf-8')))
if not rows:
    print("")
else:
    r=rows[0]
    print(",".join([
        r.get("run_id",""),
        r.get("mean_ms",""),
        r.get("attn_tflops",""),
        r.get("achieved_gflops",""),
        r.get("throughput_tokens_per_s",""),
        r.get("est_bw_gbps",""),
    ]))
PY
)"
  if [[ -z "$LINE" ]]; then
    echo "[autotune] empty metrics row for BlockM=$bm" >&2
    continue
  fi

  IFS=',' read -r run_id mean_ms attn_tflops achieved_gflops throughput_tokens_per_s est_bw_gbps <<< "$LINE"
  if ! awk "BEGIN{exit !($mean_ms > 0.01)}"; then
    echo "[autotune] skip suspicious result for BlockM=$bm: mean_ms=$mean_ms" >&2
    continue
  fi

  echo "$bm,$run_id,$RUN_DIR,$mean_ms,$attn_tflops,$achieved_gflops,$throughput_tokens_per_s,$est_bw_gbps" >> "$SUMMARY_CSV"

  if [[ -z "$BEST_MS" ]] || awk "BEGIN{exit !($mean_ms < $BEST_MS)}"; then
    BEST_MS="$mean_ms"
    BEST_BM="$bm"
  fi
done

if [[ -z "$BEST_BM" ]]; then
  echo "[autotune] no valid results, see: $SUMMARY_CSV" >&2
  exit 1
fi

echo "[autotune] summary: $SUMMARY_CSV"
echo "[autotune] best BlockM=$BEST_BM mean_ms=$BEST_MS"

if [[ "$KEEP_BEST" == "1" ]]; then
  set_block_m "$BEST_BM"
  trap - EXIT
  echo "[autotune] kept best V3_BLOCK_M=$BEST_BM in src/common/common.h"
fi

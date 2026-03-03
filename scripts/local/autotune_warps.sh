#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_FILE="$PROJECT_ROOT/scripts/local_config.example"
WARPS_LIST="2,4,8"
PROFILE_SHAPE="16x16x4096x128"
WARMUPS="10"
REPEATS="50"
KEEP_BEST="0"

usage() {
  cat <<'USAGE'
Usage: ./scripts/local/autotune_warps.sh [options]

Options:
  --config <path>           local run config file (default: scripts/local_config.example)
  --warps <list>            comma-separated NWarps candidates (default: 2,4,8)
  --profile-shape BxHxNxD   benchmark shape (default: 16x16x4096x128)
  --warmups <int>           warmup iters (default: 10)
  --repeats <int>           repeat iters (default: 50)
  --keep-best               keep best V3_WARPS in src/common/common.h (default: restore original)
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --warps) WARPS_LIST="$2"; shift 2 ;;
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

ORIG_WARPS="$(sed -n 's/^[[:space:]]*constexpr int V3_WARPS = \([0-9][0-9]*\);/\1/p' "$COMMON_H" | head -n 1)"
if [[ -z "$ORIG_WARPS" ]]; then
  echo "Failed to parse original V3_WARPS from $COMMON_H" >&2
  exit 1
fi

set_warps() {
  local w="$1"
  sed -i -E "s/^([[:space:]]*constexpr int V3_WARPS = )[0-9]+(;)$/\1${w}\2/" "$COMMON_H"
}

restore_original() {
  if [[ "$KEEP_BEST" == "0" ]]; then
    set_warps "$ORIG_WARPS"
    echo "[autotune] restored V3_WARPS=$ORIG_WARPS"
  fi
}
trap restore_original EXIT

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$PROJECT_ROOT/profile_results/autotune_warps_${TS}"
mkdir -p "$OUT_DIR"
SUMMARY_CSV="$OUT_DIR/summary.csv"

cat > "$SUMMARY_CSV" <<'CSV'
n_warps,run_id,run_dir,mean_ms,attn_tflops,achieved_gflops,throughput_tokens_per_s,est_bw_gbps
CSV

BEST_WARPS=""
BEST_MS=""

IFS=',' read -r -a WLIST <<< "$WARPS_LIST"
for w_raw in "${WLIST[@]}"; do
  w="$(echo "$w_raw" | xargs)"
  if [[ -z "$w" ]]; then
    continue
  fi
  if ! [[ "$w" =~ ^[0-9]+$ ]]; then
    echo "[autotune] skip invalid warps: $w" >&2
    continue
  fi

  echo "[autotune] testing NWarps=$w"
  set_warps "$w"

  RUN_INFO="$OUT_DIR/runinfo_w${w}.json"
  ./scripts/local/run.sh \
    --config "$CONFIG_FILE" \
    --run-name "autotune_w${w}" \
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
    echo "[autotune] missing run_dir for NWarps=$w" >&2
    continue
  fi

  OURS_CSV="$RUN_DIR/tables/ours_cpp.csv"
  if [[ ! -f "$OURS_CSV" ]]; then
    echo "[autotune] missing ours_cpp.csv for NWarps=$w: $OURS_CSV" >&2
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
    echo "[autotune] empty metrics row for NWarps=$w" >&2
    continue
  fi

  IFS=',' read -r run_id mean_ms attn_tflops achieved_gflops throughput_tokens_per_s est_bw_gbps <<< "$LINE"
  if ! awk "BEGIN{exit !($mean_ms > 0.01)}"; then
    echo "[autotune] skip suspicious result for NWarps=$w: mean_ms=$mean_ms" >&2
    continue
  fi

  echo "$w,$run_id,$RUN_DIR,$mean_ms,$attn_tflops,$achieved_gflops,$throughput_tokens_per_s,$est_bw_gbps" >> "$SUMMARY_CSV"

  if [[ -z "$BEST_MS" ]] || awk "BEGIN{exit !($mean_ms < $BEST_MS)}"; then
    BEST_MS="$mean_ms"
    BEST_WARPS="$w"
  fi
done

if [[ -z "$BEST_WARPS" ]]; then
  echo "[autotune] no valid results, see: $SUMMARY_CSV" >&2
  exit 1
fi

echo "[autotune] summary: $SUMMARY_CSV"
echo "[autotune] best NWarps=$BEST_WARPS mean_ms=$BEST_MS"

if [[ "$KEEP_BEST" == "1" ]]; then
  set_warps "$BEST_WARPS"
  trap - EXIT
  echo "[autotune] kept best V3_WARPS=$BEST_WARPS in src/common/common.h"
fi

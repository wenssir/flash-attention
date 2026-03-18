#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_FILE=""
RUN_INFO=""
DEFAULT_CONFIG_FILE="$PROJECT_ROOT/scripts/local_config.example"
ORIGINAL_ARGS=("$@")

# Defaults
RUN_NAME="bench"
OUTPUT_DIR="profile_results"
RUN_ID=""

SEQ_LENS="4096"
D_HEADS="128"
BATCH_SIZE="16"
N_HEADS="16"
PROFILE_SHAPE="16x16x4096x128"
PROFILE_SHAPE_SET="0"

WARMUPS="10"
REPEATS="50"
DTYPE="fp16"
INPUT_MODE="random"
SEED="1234"
REPLAY_DIR="benchmark/data/replay"
REPLAY_FILE=""
BASELINE="pytorch"   # pytorch|official

RUN_PYBIND="1"
RUN_OFFICIAL="0"
RUN_PYTORCH="0"
RUN_PROFILE_BENCHMARK="1"
RUN_NCU="0"
STRICT_MODE="0"
SELECTED_ANY="0"

OFFICIAL_DTYPE="fp16"
OFFICIAL_PYTORCH_DTYPE="match"
OFFICIAL_PATH="$HOME/flash-attention"

BUILD_DIR="build"
CUDA_ARCH=""
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"
CUDA_HOME="${CUDA_HOME:-}"
CUDACXX="${CUDACXX:-}"
NCU_CMD="ncu"
NCU_SET="full"
NCU_LAUNCH_SKIP=""
NCU_LAUNCH_COUNT=""
NCU_USE_SUDO="0"
NCU_TARGET="profile_kernel"

# Auto-load local default config if present.
if [[ -f "$DEFAULT_CONFIG_FILE" ]]; then
  CONFIG_FILE="$DEFAULT_CONFIG_FILE"
  # shellcheck disable=SC1090
  source "$DEFAULT_CONFIG_FILE"
fi

usage() {
  cat <<'USAGE'
Usage: ./scripts/local/run.sh [options]

Options:
  --config <path>              Source config (KEY=VALUE)
                               If omitted, auto-loads scripts/local_config.example when present.
  --run-info <path>            Write run info JSON to this path
  --run-name <name>
  --output-dir <dir>
  --run-id <id>

  Selectors (presence means enabled; if any selector is present, only selected ones run):
  --pybind                     enable pybind benchmark
  --official                   enable official flash-attn benchmark
  --pytorch                    enable pytorch benchmark
  --profile-benchmark          enable profile_kernel benchmark
  --ncu                        enable ncu profiling

  --seq-lens <list>
  --d-heads <list>
  --batch-size <int>
  --n-heads <int>
  --warmups <int>
  --repeats <int>
  --dtype <fp16|fp32>
  --input-mode <random|structured|stress|replay>
  --seed <int>
  --replay-dir <path>
  --replay-file <path>
  --baseline <pytorch|official>

  --official-dtype <fp16|bf16>
  --official-pytorch-dtype <match|fp32>
  --official-path <path>

  --build-dir <dir>
  --cuda-arch <arch>
  --torch-cuda-arch-list <list>
  --cuda-home <path>
  --cudacxx <path>
  --ncu-cmd <path>
  --ncu-set <name>
  --ncu-launch-skip <int>
  --ncu-launch-count <int>
  --ncu-use-sudo <0|1>
  --ncu-target <profile_kernel>

  --profile-shape BxHxNxD
  --skip-ncu                 compatibility alias, force disable ncu
  --strict
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_FILE="$2"
      if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "Config file not found: $CONFIG_FILE" >&2
        exit 1
      fi
      # shellcheck disable=SC1090
      source "$CONFIG_FILE"
      shift 2
      ;;
    --run-info) RUN_INFO="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;

    --pybind) SELECTED_ANY="1"; shift ;;
    --official) SELECTED_ANY="1"; shift ;;
    --pytorch) SELECTED_ANY="1"; shift ;;
    --profile-benchmark) SELECTED_ANY="1"; shift ;;
    --profile-benchmark-v5-narrow) SELECTED_ANY="1"; shift ;;
    --ncu) SELECTED_ANY="1"; shift ;;

    --seq-lens) SEQ_LENS="$2"; shift 2 ;;
    --d-heads) D_HEADS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --n-heads) N_HEADS="$2"; shift 2 ;;
    --warmups) WARMUPS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --input-mode) INPUT_MODE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --replay-dir) REPLAY_DIR="$2"; shift 2 ;;
    --replay-file) REPLAY_FILE="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;

    --official-dtype) OFFICIAL_DTYPE="$2"; shift 2 ;;
    --official-pytorch-dtype) OFFICIAL_PYTORCH_DTYPE="$2"; shift 2 ;;
    --official-path) OFFICIAL_PATH="$2"; shift 2 ;;

    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --cuda-arch) CUDA_ARCH="$2"; shift 2 ;;
    --torch-cuda-arch-list) TORCH_CUDA_ARCH_LIST="$2"; shift 2 ;;
    --cuda-home) CUDA_HOME="$2"; shift 2 ;;
    --cudacxx) CUDACXX="$2"; shift 2 ;;
    --ncu-cmd) NCU_CMD="$2"; shift 2 ;;
    --ncu-set) NCU_SET="$2"; shift 2 ;;
    --ncu-launch-skip) NCU_LAUNCH_SKIP="$2"; shift 2 ;;
    --ncu-launch-count) NCU_LAUNCH_COUNT="$2"; shift 2 ;;
    --ncu-use-sudo) NCU_USE_SUDO="$2"; shift 2 ;;
    --ncu-target) NCU_TARGET="$2"; shift 2 ;;

    --profile-shape) PROFILE_SHAPE="$2"; PROFILE_SHAPE_SET="1"; shift 2 ;;
    --skip-ncu) RUN_NCU="0"; shift ;;
    --strict) STRICT_MODE="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$BASELINE" != "pytorch" && "$BASELINE" != "official" ]]; then
  echo "Invalid --baseline: $BASELINE (expected pytorch|official)" >&2
  exit 1
fi

# Backward-compatible config mapping when no explicit selectors provided.
if [[ "$SELECTED_ANY" == "0" && -n "${RUN_TARGET:-}" ]]; then
  case "$RUN_TARGET" in
    all) RUN_PYBIND="1"; RUN_PROFILE_BENCHMARK="1"; RUN_OFFICIAL="1"; RUN_PYTORCH="1" ;;
    ours) RUN_PYBIND="0"; RUN_PROFILE_BENCHMARK="1"; RUN_OFFICIAL="0"; RUN_PYTORCH="0" ;;
    official) RUN_PYBIND="0"; RUN_PROFILE_BENCHMARK="0"; RUN_OFFICIAL="1"; RUN_PYTORCH="0" ;;
    pytorch) RUN_PYBIND="0"; RUN_PROFILE_BENCHMARK="0"; RUN_OFFICIAL="0"; RUN_PYTORCH="1" ;;
    *) echo "Invalid RUN_TARGET=$RUN_TARGET" >&2; exit 1 ;;
  esac
fi

# Legacy config keys
[[ -n "${OFFICIAL_SEQ_LENS:-}" ]] && SEQ_LENS="$OFFICIAL_SEQ_LENS"
[[ -n "${OFFICIAL_D_HEADS:-}" ]] && D_HEADS="$OFFICIAL_D_HEADS"
[[ -n "${OFFICIAL_BATCH_SIZE:-}" ]] && BATCH_SIZE="$OFFICIAL_BATCH_SIZE"
[[ -n "${OFFICIAL_N_HEADS:-}" ]] && N_HEADS="$OFFICIAL_N_HEADS"
[[ -n "${OFFICIAL_WARMUPS:-}" ]] && WARMUPS="$OFFICIAL_WARMUPS"
[[ -n "${OFFICIAL_REPEATS:-}" ]] && REPEATS="$OFFICIAL_REPEATS"
[[ -n "${OFFICIAL_DTYPE:-}" ]] && OFFICIAL_DTYPE="$OFFICIAL_DTYPE"
[[ -n "${OFFICIAL_PYTORCH_DTYPE:-}" ]] && OFFICIAL_PYTORCH_DTYPE="$OFFICIAL_PYTORCH_DTYPE"
[[ -n "${OFFICIAL_INPUT_MODE:-}" ]] && INPUT_MODE="$OFFICIAL_INPUT_MODE"
[[ -n "${OFFICIAL_SEED:-}" ]] && SEED="$OFFICIAL_SEED"
[[ -n "${OFFICIAL_REPLAY_DIR:-}" ]] && REPLAY_DIR="$OFFICIAL_REPLAY_DIR"
[[ -n "${OFFICIAL_REPLAY_FILE:-}" ]] && REPLAY_FILE="$OFFICIAL_REPLAY_FILE"
if [[ -n "${OURS_BENCH_CONFIG:-}" ]]; then
  PROFILE_SHAPE="$OURS_BENCH_CONFIG"
  PROFILE_SHAPE_SET="1"
fi

if [[ "$SELECTED_ANY" == "0" ]]; then
  [[ -n "${RUN_PYBIND:-}" ]] && RUN_PYBIND="$RUN_PYBIND"
  [[ -n "${RUN_OFFICIAL:-}" ]] && RUN_OFFICIAL="$RUN_OFFICIAL"
  [[ -n "${RUN_PYTORCH:-}" ]] && RUN_PYTORCH="$RUN_PYTORCH"
  [[ -n "${RUN_PROFILE_BENCHMARK:-}" ]] && RUN_PROFILE_BENCHMARK="$RUN_PROFILE_BENCHMARK"
  [[ -n "${RUN_BENCHMARK:-}" ]] && RUN_PROFILE_BENCHMARK="$RUN_BENCHMARK"
  [[ -n "${RUN_NCU:-}" ]] && RUN_NCU="$RUN_NCU"
fi
[[ -n "${REMOTE_ARCH:-}" ]] && CUDA_ARCH="$REMOTE_ARCH"
[[ -n "${NCU_SET:-}" ]] && NCU_SET="$NCU_SET"
[[ -n "${NCU_LAUNCH_SKIP:-}" ]] && NCU_LAUNCH_SKIP="$NCU_LAUNCH_SKIP"
[[ -n "${NCU_LAUNCH_COUNT:-}" ]] && NCU_LAUNCH_COUNT="$NCU_LAUNCH_COUNT"
[[ -n "${NCU_TARGET:-}" ]] && NCU_TARGET="$NCU_TARGET"

# For NCU on remote hosts, prefer absolute path and sudo.
if [[ "$RUN_NCU" == "1" ]]; then
  if [[ "$NCU_CMD" != /* && -x "/usr/local/cuda/bin/ncu" ]]; then
    NCU_CMD="/usr/local/cuda/bin/ncu"
  fi
  if [[ "$NCU_USE_SUDO" == "1" && "$NCU_CMD" != /* ]]; then
    echo "When --ncu-use-sudo=1, --ncu-cmd must be an absolute path (current: $NCU_CMD)" >&2
    exit 1
  fi
fi

if [[ "$PROFILE_SHAPE_SET" == "1" && -n "${PROFILE_SHAPE:-}" ]]; then
  IFS='x' read -r b h n d <<< "$PROFILE_SHAPE"
  if [[ "$b" =~ ^[0-9]+$ && "$h" =~ ^[0-9]+$ && "$n" =~ ^[0-9]+$ && "$d" =~ ^[0-9]+$ ]]; then
    BATCH_SIZE="$b"
    N_HEADS="$h"
    SEQ_LENS="$n"
    D_HEADS="$d"
  fi
fi
if [[ "$PROFILE_SHAPE_SET" != "1" ]]; then
  SHAPE_SEQ_LEN="$SEQ_LENS"
  if [[ "$SHAPE_SEQ_LEN" == *,* ]]; then
    SHAPE_SEQ_LEN="${SHAPE_SEQ_LEN%%,*}"
  fi
  PROFILE_SHAPE="${BATCH_SIZE}x${N_HEADS}x${SHAPE_SEQ_LEN}x${D_HEADS}"
fi

# If any explicit selector flag was used, run only selected ones.
if [[ "$SELECTED_ANY" == "1" ]]; then
  RUN_PYBIND="0"
  RUN_OFFICIAL="0"
  RUN_PYTORCH="0"
  RUN_PROFILE_BENCHMARK="0"
  RUN_NCU="0"
  for a in "${ORIGINAL_ARGS[@]}"; do
    case "$a" in
      --pybind) RUN_PYBIND="1" ;;
      --official) RUN_OFFICIAL="1" ;;
      --pytorch) RUN_PYTORCH="1" ;;
      --profile-benchmark) RUN_PROFILE_BENCHMARK="1" ;;
      --ncu) RUN_NCU="1" ;;
    esac
  done
fi

mkdir -p "$OUTPUT_DIR"
if [[ -n "$TORCH_CUDA_ARCH_LIST" ]]; then
  export TORCH_CUDA_ARCH_LIST
fi
if [[ -n "$CUDA_HOME" ]]; then
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "$CUDACXX" ]]; then
  export CUDACXX
fi

cmd=(
  python3 -m benchmark.api.runner
  --run-name "$RUN_NAME"
  --output-dir "$OUTPUT_DIR"
  --seq-lens "$SEQ_LENS"
  --d-heads "$D_HEADS"
  --batch-size "$BATCH_SIZE"
  --n-heads "$N_HEADS"
  --profile-shape "$PROFILE_SHAPE"
  --warmups "$WARMUPS"
  --repeats "$REPEATS"
  --dtype "$DTYPE"
  --input-mode "$INPUT_MODE"
  --seed "$SEED"
  --replay-dir "$REPLAY_DIR"
  --replay-file "$REPLAY_FILE"
  --baseline "$BASELINE"
  --run-pybind "$RUN_PYBIND"
  --run-official "$RUN_OFFICIAL"
  --run-pytorch "$RUN_PYTORCH"
  --run-profile-benchmark "$RUN_PROFILE_BENCHMARK"
  --run-ncu "$RUN_NCU"
  --strict "$STRICT_MODE"
  --official-path "$OFFICIAL_PATH"
  --official-dtype "$OFFICIAL_DTYPE"
  --official-pytorch-dtype "$OFFICIAL_PYTORCH_DTYPE"
  --build-dir "$BUILD_DIR"
  --cuda-arch "$CUDA_ARCH"
  --ncu-cmd "$NCU_CMD"
  --ncu-set "$NCU_SET"
  --ncu-launch-skip "$NCU_LAUNCH_SKIP"
  --ncu-launch-count "$NCU_LAUNCH_COUNT"
  --ncu-use-sudo "$NCU_USE_SUDO"
  --ncu-target "$NCU_TARGET"
)

[[ -n "$RUN_ID" ]] && cmd+=(--run-id "$RUN_ID")
[[ -n "$RUN_INFO" ]] && cmd+=(--run-info "$RUN_INFO")

printf '[local] %s\n' "${cmd[*]}"
"${cmd[@]}"

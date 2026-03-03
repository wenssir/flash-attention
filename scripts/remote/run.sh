#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

REMOTE_ADDR=""
REMOTE_PORT="22"
REMOTE_PATH="/tmp/flash_attention_v2_profile"
SSH_KEY_PATH="$HOME/.ssh/id_rsa"
VENV_PATH=""
CONFIG_FILE=""
DEFAULT_CONFIG_FILE="$PROJECT_ROOT/scripts/remote_config.example"
FORWARD_ARGS=()
REMOTE_TIMEOUT_SEC="0"

usage() {
  cat <<'USAGE'
Usage: ./scripts/remote/run.sh [remote_addr] [options] [pipeline options...]

Options:
  --config <path>          Source config file (KEY=VALUE)
                           If omitted, auto-loads scripts/remote_config.example when present.
  --host <user@ip>         Remote host
  --port <22>
  --remote-path <path>
  --ssh-key <path>
  --venv-path <path>       Remote virtualenv path
  --timeout-sec <int>      Remote wait timeout, 0 means no timeout

Any unknown --xxx options are forwarded to remote scripts/local/run.sh.
USAGE
}

# Auto-load remote default config when available.
if [[ -f "$DEFAULT_CONFIG_FILE" ]]; then
  CONFIG_FILE="$DEFAULT_CONFIG_FILE"
  # shellcheck disable=SC1090
  source "$DEFAULT_CONFIG_FILE"
  [[ -n "${REMOTE_ADDR:-}" ]] && REMOTE_ADDR="$REMOTE_ADDR"
  [[ -n "${REMOTE_PORT:-}" ]] && REMOTE_PORT="$REMOTE_PORT"
  [[ -n "${REMOTE_PATH:-}" ]] && REMOTE_PATH="$REMOTE_PATH"
  [[ -n "${SSH_KEY_PATH:-}" ]] && SSH_KEY_PATH="$SSH_KEY_PATH"
  [[ -n "${VENV_PATH:-}" ]] && VENV_PATH="$VENV_PATH"
  [[ -n "${REMOTE_TIMEOUT_SEC:-}" ]] && REMOTE_TIMEOUT_SEC="$REMOTE_TIMEOUT_SEC"
fi

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
      [[ -n "${REMOTE_ADDR:-}" ]] && REMOTE_ADDR="$REMOTE_ADDR"
      [[ -n "${REMOTE_PORT:-}" ]] && REMOTE_PORT="$REMOTE_PORT"
      [[ -n "${REMOTE_PATH:-}" ]] && REMOTE_PATH="$REMOTE_PATH"
      [[ -n "${SSH_KEY_PATH:-}" ]] && SSH_KEY_PATH="$SSH_KEY_PATH"
      [[ -n "${VENV_PATH:-}" ]] && VENV_PATH="$VENV_PATH"
      [[ -n "${REMOTE_TIMEOUT_SEC:-}" ]] && REMOTE_TIMEOUT_SEC="$REMOTE_TIMEOUT_SEC"
      shift 2
      ;;
    --host) REMOTE_ADDR="$2"; shift 2 ;;
    --port) REMOTE_PORT="$2"; shift 2 ;;
    --remote-path) REMOTE_PATH="$2"; shift 2 ;;
    --ssh-key) SSH_KEY_PATH="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --timeout-sec) REMOTE_TIMEOUT_SEC="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*)
      FORWARD_ARGS+=("$1")
      if [[ $# -gt 1 && "$2" != --* ]]; then
        FORWARD_ARGS+=("$2")
        shift 2
      else
        shift
      fi
      ;;
    *)
      if [[ -z "$REMOTE_ADDR" ]]; then
        REMOTE_ADDR="$1"
        shift
      else
        echo "Unknown positional arg: $1" >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$REMOTE_ADDR" ]]; then
  echo "REMOTE_ADDR is required (use --host or config REMOTE_ADDR)" >&2
  exit 1
fi
if [[ ! -f "$SSH_KEY_PATH" ]]; then
  echo "SSH key not found: $SSH_KEY_PATH" >&2
  exit 1
fi
if ! [[ "$REMOTE_TIMEOUT_SEC" =~ ^[0-9]+$ ]]; then
  echo "Invalid --timeout-sec: $REMOTE_TIMEOUT_SEC" >&2
  exit 1
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
SRC_TGZ="/tmp/fa_v2_src_${RUN_TS}.tgz"
REMOTE_CFG="$REMOTE_PATH/remote_config.env"
REMOTE_RUN_INFO="$REMOTE_PATH/run_info.json"
REMOTE_JOB_LOG="$REMOTE_PATH/remote_job.log"
REMOTE_JOB_PID="$REMOTE_PATH/remote_job.pid"
LOCAL_FETCH_DIR="$PROJECT_ROOT/profile_results/remote_${RUN_TS}"
mkdir -p "$LOCAL_FETCH_DIR"

SSH_OPTS=(
  -p "$REMOTE_PORT"
  -i "$SSH_KEY_PATH"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=20
  -o TCPKeepAlive=yes
)
SCP_OPTS=( -P "$REMOTE_PORT" -i "$SSH_KEY_PATH" )

echo "[remote] target=$REMOTE_ADDR path=$REMOTE_PATH"

tar czf "$SRC_TGZ" \
  --exclude=.git \
  --exclude=build \
  --exclude='build_*' \
  --exclude=profile_results \
  --exclude='*.ncu-rep' \
  --exclude='__pycache__' \
  .

ssh "${SSH_OPTS[@]}" "$REMOTE_ADDR" "mkdir -p '$REMOTE_PATH'"
scp "${SCP_OPTS[@]}" "$SRC_TGZ" "$REMOTE_ADDR:$REMOTE_PATH/src.tgz"
rm -f "$SRC_TGZ"

if [[ -n "$CONFIG_FILE" ]]; then
  scp "${SCP_OPTS[@]}" "$CONFIG_FILE" "$REMOTE_ADDR:$REMOTE_CFG"
else
  ssh "${SSH_OPTS[@]}" "$REMOTE_ADDR" "cat > '$REMOTE_CFG' <<'CFG'
RUN_NAME=remote
OUTPUT_DIR=profile_results
RUN_PYBIND=1
RUN_OFFICIAL=0
RUN_PYTORCH=0
RUN_PROFILE_BENCHMARK=1
RUN_NCU=0
CFG"
fi

echo "[remote] starting detached job"
ssh "${SSH_OPTS[@]}" "$REMOTE_ADDR" bash -s -- \
  "$REMOTE_PATH" \
  "$REMOTE_RUN_INFO" \
  "$VENV_PATH" \
  "$REMOTE_JOB_LOG" \
  "$REMOTE_JOB_PID" \
  "${FORWARD_ARGS[@]}" <<'EOS'
set -e
REMOTE_PATH="$1"
REMOTE_RUN_INFO="$2"
VENV_PATH="$3"
REMOTE_JOB_LOG="$4"
REMOTE_JOB_PID="$5"
shift 5

cd "$REMOTE_PATH"
rm -f "$REMOTE_RUN_INFO"
rm -rf src
mkdir -p src
cd src
tar xzf ../src.tgz

# Build command with safely escaped args.
cmd=(bash scripts/local/run.sh --config ../remote_config.env --run-info "$REMOTE_RUN_INFO")
for a in "$@"; do
  cmd+=("$a")
done

run_file="$REMOTE_PATH/run_remote_local.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -e'
  if [ -n "$VENV_PATH" ]; then
    printf 'if [ -f %q ]; then source %q; else echo %q >&2; exit 1; fi\n' "$VENV_PATH/bin/activate" "$VENV_PATH/bin/activate" "[remote] venv not found: $VENV_PATH"
  else
    echo 'if [ -f "$HOME/fa-venv/bin/activate" ]; then source "$HOME/fa-venv/bin/activate"; fi'
  fi
  printf 'cd %q\n' "$REMOTE_PATH/src"
  printf '%q' "${cmd[0]}"
  for ((i=1;i<${#cmd[@]};i++)); do
    printf ' %q' "${cmd[$i]}"
  done
  printf '\n'
} > "$run_file"
chmod +x "$run_file"

nohup bash "$run_file" > "$REMOTE_JOB_LOG" 2>&1 < /dev/null &
echo $! > "$REMOTE_JOB_PID"
echo "STARTED PID=$(cat "$REMOTE_JOB_PID")"
EOS

echo "[remote] monitoring..."
start_ts="$(date +%s)"
while true; do
  status="$(ssh "${SSH_OPTS[@]}" "$REMOTE_ADDR" "if [ -f '$REMOTE_JOB_PID' ]; then pid=\$(cat '$REMOTE_JOB_PID'); if kill -0 \$pid 2>/dev/null; then echo RUNNING; else echo DONE; fi; else echo NO_PID; fi" || echo SSH_FAIL)"
  case "$status" in
    RUNNING)
      echo "[remote] job running..."
      ;;
    DONE)
      echo "[remote] job finished"
      break
      ;;
    NO_PID)
      echo "[remote] job pid missing"
      break
      ;;
    SSH_FAIL)
      echo "[remote] ssh check failed, retrying..."
      ;;
    *)
      echo "[remote] status=$status"
      ;;
  esac

  if [[ "$REMOTE_TIMEOUT_SEC" != "0" ]]; then
    now_ts="$(date +%s)"
    elapsed=$((now_ts - start_ts))
    if (( elapsed > REMOTE_TIMEOUT_SEC )); then
      echo "[remote] timeout reached: ${REMOTE_TIMEOUT_SEC}s" >&2
      break
    fi
  fi
  sleep 15
done

echo "[remote] fetch remote log"
scp "${SCP_OPTS[@]}" "$REMOTE_ADDR:$REMOTE_JOB_LOG" "$LOCAL_FETCH_DIR/remote_job.log" || true

if ! scp "${SCP_OPTS[@]}" "$REMOTE_ADDR:$REMOTE_RUN_INFO" "$LOCAL_FETCH_DIR/run_info.json"; then
  echo "[remote] run_info not found. last remote log:" >&2
  tail -n 120 "$LOCAL_FETCH_DIR/remote_job.log" >&2 || true
  exit 1
fi

REMOTE_RESULT_DIR="$(python3 - <<'PY' "$LOCAL_FETCH_DIR/run_info.json"
import json,sys
p=sys.argv[1]
obj=json.load(open(p))
print(obj.get('run_dir',''))
PY
)"

if [[ -z "$REMOTE_RESULT_DIR" ]]; then
  echo "[remote] failed to read remote run_dir" >&2
  exit 1
fi

if ! ssh "${SSH_OPTS[@]}" "$REMOTE_ADDR" "test -d '$REMOTE_RESULT_DIR'"; then
  echo "[remote] run_dir missing on remote: $REMOTE_RESULT_DIR" >&2
  echo "[remote] last remote log:" >&2
  tail -n 120 "$LOCAL_FETCH_DIR/remote_job.log" >&2 || true
  exit 1
fi

echo "[remote] syncing results from $REMOTE_RESULT_DIR"
scp -r "${SCP_OPTS[@]}" "$REMOTE_ADDR:$REMOTE_RESULT_DIR" "$LOCAL_FETCH_DIR/"

LOCAL_RESULT_DIR="$(python3 - <<'PY' "$LOCAL_FETCH_DIR/run_info.json"
import json,sys,os
obj=json.load(open(sys.argv[1]))
print(os.path.basename(obj.get('run_dir','')))
PY
)"

echo "[remote] done"
echo "RUN_INFO=$LOCAL_FETCH_DIR/run_info.json"
echo "RESULT_DIR=$LOCAL_FETCH_DIR/$LOCAL_RESULT_DIR"

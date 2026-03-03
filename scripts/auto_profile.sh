#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/auto_profile.sh local  [args...]
  ./scripts/auto_profile.sh remote [args...]
  ./scripts/auto_profile.sh --local  [args...]
  ./scripts/auto_profile.sh --remote [args...]

This is a compatibility wrapper.
- local  -> scripts/local/run.sh
- remote -> scripts/remote/run.sh
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

mode="$1"
shift

case "$mode" in
  --local)
    if [[ $# -eq 0 && -f "$SCRIPT_DIR/local_config.example" ]]; then
      exec "$SCRIPT_DIR/local/run.sh" --config "$SCRIPT_DIR/local_config.example"
    fi
    exec "$SCRIPT_DIR/local/run.sh" "$@"
    ;;
  --remote)
    if [[ $# -eq 0 && -f "$SCRIPT_DIR/remote_config.example" ]]; then
      exec "$SCRIPT_DIR/remote/run.sh" --config "$SCRIPT_DIR/remote_config.example"
    fi
    exec "$SCRIPT_DIR/remote/run.sh" "$@"
    ;;
  local)
    exec "$SCRIPT_DIR/local/run.sh" "$@"
    ;;
  remote)
    if [[ $# -eq 0 && -f "$SCRIPT_DIR/remote_config.example" ]]; then
      exec "$SCRIPT_DIR/remote/run.sh" --config "$SCRIPT_DIR/remote_config.example"
    fi
    exec "$SCRIPT_DIR/remote/run.sh" "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    usage
    exit 1
    ;;
esac

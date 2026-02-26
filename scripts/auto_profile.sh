#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
PROFILE_DIR="$PROJECT_ROOT/profile_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NCU_AVAILABLE=true
DEFAULT_REMOTE_ADDR="ubuntu@149.36.1.223"
DEFAULT_SSH_KEY="$HOME/.ssh/id_rsa"
NCU_SET="${NCU_SET:-full}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-}"
BENCH_CONFIG="${BENCH_CONFIG:-all}"
RUN_BENCHMARK="${RUN_BENCHMARK:-0}"
RUN_NCU="${RUN_NCU:-1}"
NCU_CONFIG="${NCU_CONFIG:-large}"
RUN_OFFICIAL="${RUN_OFFICIAL:-0}"
RUN_PYTORCH="${RUN_PYTORCH:-0}"
STRICT_MODE="${STRICT_MODE:-0}"
OFFICIAL_DTYPE="${OFFICIAL_DTYPE:-fp16}"
OFFICIAL_PYTORCH_DTYPE="${OFFICIAL_PYTORCH_DTYPE:-match}"
OFFICIAL_SEQ_LENS="${OFFICIAL_SEQ_LENS:-4096}"
OFFICIAL_D_HEADS="${OFFICIAL_D_HEADS:-128}"
OFFICIAL_BATCH_SIZE="${OFFICIAL_BATCH_SIZE:-8}"
OFFICIAL_N_HEADS="${OFFICIAL_N_HEADS:-16}"
OFFICIAL_WARMUPS="${OFFICIAL_WARMUPS:-10}"
OFFICIAL_REPEATS="${OFFICIAL_REPEATS:-50}"
OFFICIAL_INPUT_MODE="${OFFICIAL_INPUT_MODE:-random}"
OFFICIAL_SEED="${OFFICIAL_SEED:-1234}"
OFFICIAL_REPLAY_DIR="${OFFICIAL_REPLAY_DIR:-benchmark/data/replay}"
OFFICIAL_REPLAY_FILE="${OFFICIAL_REPLAY_FILE:-}"

mkdir -p "$PROFILE_DIR"
mkdir -p "$BUILD_DIR"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

detect_system_info() {
    print_section "Detecting Local System Information"

    if command -v nvcc &> /dev/null; then
        NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        log_info "NVCC Version: $NVCC_VERSION"
    else
        log_error "nvcc not found"
        exit 1
    fi

    if command -v ncu &> /dev/null; then
        NCU_VERSION=$(ncu --version | head -n1)
        log_info "NCU Version: $NCU_VERSION"
    else
        log_warning "ncu not found, profiling export will be skipped"
        NCU_AVAILABLE=false
    fi

    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
        GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
        log_info "Local GPU: $GPU_NAME"
        log_info "Local Compute Capability: $GPU_ARCH"
    else
        log_error "nvidia-smi not found"
        exit 1
    fi
}

compile_project() {
    local cuda_arch_override="$1"

    print_section "Compiling Flash Attention V2"
    cd "$BUILD_DIR"

    local cuda_arch="${GPU_ARCH}"
    if [ -n "$cuda_arch_override" ]; then
        cuda_arch="$cuda_arch_override"
    fi

    log_info "Configuring CMake with CUDA_ARCHITECTURES=$cuda_arch"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch" \
        -DFA_BUILD_TESTS=OFF \
        -DFA_BUILD_UTILS=ON

    log_info "Building project"
    cmake --build . -j"$(nproc)"

    if [ -f "profile_kernel" ]; then
        log_success "Compilation successful"
        log_info "Executable: $BUILD_DIR/profile_kernel"
    else
        log_error "Compilation failed, profile_kernel not found"
        exit 1
    fi

    cd "$PROJECT_ROOT"
}

run_local_profile() {
    print_section "Running Local Profile"
    cd "$BUILD_DIR"
    log_info "Local working directory: $(pwd)"

    local bench_args=()
    if [ -n "$BENCH_CONFIG" ] && [ "$BENCH_CONFIG" != "all" ]; then
        bench_args=("$BENCH_CONFIG")
    fi
    local ncu_args=()
    if [ -n "$NCU_CONFIG" ] && [ "$NCU_CONFIG" != "all" ]; then
        ncu_args=("$NCU_CONFIG")
    fi

    if [ "$RUN_BENCHMARK" = "1" ]; then
        log_info "Running benchmark command: ./profile_kernel --mode benchmark ${bench_args[*]}"
        ./profile_kernel --mode benchmark "${bench_args[@]}"
    else
        log_info "RUN_BENCHMARK=0, skipping plain benchmark run; keeping NCU profiling"
    fi

    if [ "$RUN_NCU" = "1" ] && [ "$NCU_AVAILABLE" = true ]; then
        local out="$PROFILE_DIR/ncu_local_${TIMESTAMP}.ncu-rep"
        local ncu_run="ncu"
        if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
            ncu_run="sudo ncu"
        fi
        log_info "Local NCU runner: $ncu_run"
        local ncu_cmd="$ncu_run --set \"$NCU_SET\" --export \"$out\" --force-overwrite"
        if [ -n "$NCU_LAUNCH_SKIP" ]; then
            ncu_cmd="$ncu_cmd --launch-skip \"$NCU_LAUNCH_SKIP\""
        fi
        if [ -n "$NCU_LAUNCH_COUNT" ]; then
            ncu_cmd="$ncu_cmd --launch-count \"$NCU_LAUNCH_COUNT\""
        fi
        ncu_cmd="$ncu_cmd ./profile_kernel --mode ncu ${ncu_args[*]}"
        log_info "Running NCU command: $ncu_cmd"
        eval "$ncu_cmd"
        if [ ! -f "$out" ]; then
            log_error "NCU finished but report not found: $out"
            exit 1
        fi
        log_success "NCU report saved: $out"
    else
        if [ "$RUN_NCU" = "1" ]; then
            log_warning "ncu unavailable; only plain benchmark run is possible"
        else
            log_info "RUN_NCU=0, skip local ncu profiling"
        fi
    fi

    cd "$PROJECT_ROOT"
}

run_remote_profile() {
    print_section "Running Remote Profile"

    local remote_addr=""
    shift 1

    local remote_port="22"
    local remote_path="/tmp/flash_attention_v2_profile_${TIMESTAMP}"
    local remote_bench_pid_file="/tmp/flash_attention_v2_bench_${TIMESTAMP}.pid"
    local remote_bench_log_file="/tmp/flash_attention_v2_bench_${TIMESTAMP}.log"
    local remote_ncu_pid_file="/tmp/flash_attention_v2_ncu_${TIMESTAMP}.pid"
    local remote_ncu_log_file="/tmp/flash_attention_v2_ncu_${TIMESTAMP}.log"
    local remote_arch_override=""
    local ssh_key_override=""
    local config_file=""
    local bench_config="$BENCH_CONFIG"
    local run_benchmark="$RUN_BENCHMARK"
    local run_ncu="$RUN_NCU"
    local ncu_config="$NCU_CONFIG"
    local run_ours="1"
    local run_official="$RUN_OFFICIAL"
    local run_pytorch="$RUN_PYTORCH"
    local strict_mode="$STRICT_MODE"
    local official_dtype="$OFFICIAL_DTYPE"
    local official_pytorch_dtype="$OFFICIAL_PYTORCH_DTYPE"
    local official_seq_lens="$OFFICIAL_SEQ_LENS"
    local official_d_heads="$OFFICIAL_D_HEADS"
    local official_batch_size="$OFFICIAL_BATCH_SIZE"
    local official_n_heads="$OFFICIAL_N_HEADS"
    local official_warmups="$OFFICIAL_WARMUPS"
    local official_repeats="$OFFICIAL_REPEATS"
    local official_input_mode="$OFFICIAL_INPUT_MODE"
    local official_seed="$OFFICIAL_SEED"
    local official_replay_dir="$OFFICIAL_REPLAY_DIR"
    local official_replay_file="$OFFICIAL_REPLAY_FILE"

    while [ $# -gt 0 ]; do
        case "$1" in
            -p|--port)
                remote_port="$2"
                shift 2
                ;;
            --remote-path)
                remote_path="$2"
                shift 2
                ;;
            --remote-arch)
                remote_arch_override="$2"
                shift 2
                ;;
            --ssh-key)
                ssh_key_override="$2"
                shift 2
                ;;
            --config)
                config_file="$2"
                if [ ! -f "$config_file" ]; then
                    log_error "Config file not found: $config_file"
                    exit 1
                fi
                # shellcheck disable=SC1090
                source "$config_file"
                [ -n "${REMOTE_ADDR:-}" ] && remote_addr="$REMOTE_ADDR"
                [ -n "${REMOTE_PORT:-}" ] && remote_port="$REMOTE_PORT"
                [ -n "${REMOTE_PATH:-}" ] && remote_path="$REMOTE_PATH"
                [ -n "${REMOTE_ARCH:-}" ] && remote_arch_override="$REMOTE_ARCH"
                [ -n "${SSH_KEY_PATH:-}" ] && ssh_key_override="$SSH_KEY_PATH"
                [ -n "${OURS_BENCH_CONFIG:-}" ] && bench_config="$OURS_BENCH_CONFIG"
                [ -n "${OURS_NCU_CONFIG:-}" ] && ncu_config="$OURS_NCU_CONFIG"
                [ -n "${RUN_BENCHMARK:-}" ] && run_benchmark="$RUN_BENCHMARK"
                [ -n "${RUN_NCU:-}" ] && run_ncu="$RUN_NCU"
                [ -n "${RUN_OURS:-}" ] && run_ours="$RUN_OURS"
                [ -n "${RUN_OFFICIAL:-}" ] && run_official="$RUN_OFFICIAL"
                [ -n "${RUN_PYTORCH:-}" ] && run_pytorch="$RUN_PYTORCH"
                [ -n "${STRICT_MODE:-}" ] && strict_mode="$STRICT_MODE"
                [ -n "${NCU_SET:-}" ] && NCU_SET="$NCU_SET"
                [ -n "${NCU_LAUNCH_SKIP:-}" ] && NCU_LAUNCH_SKIP="$NCU_LAUNCH_SKIP"
                [ -n "${NCU_LAUNCH_COUNT:-}" ] && NCU_LAUNCH_COUNT="$NCU_LAUNCH_COUNT"
                [ -n "${OFFICIAL_DTYPE:-}" ] && official_dtype="$OFFICIAL_DTYPE"
                [ -n "${OFFICIAL_PYTORCH_DTYPE:-}" ] && official_pytorch_dtype="$OFFICIAL_PYTORCH_DTYPE"
                [ -n "${OFFICIAL_SEQ_LENS:-}" ] && official_seq_lens="$OFFICIAL_SEQ_LENS"
                [ -n "${OFFICIAL_D_HEADS:-}" ] && official_d_heads="$OFFICIAL_D_HEADS"
                [ -n "${OFFICIAL_BATCH_SIZE:-}" ] && official_batch_size="$OFFICIAL_BATCH_SIZE"
                [ -n "${OFFICIAL_N_HEADS:-}" ] && official_n_heads="$OFFICIAL_N_HEADS"
                [ -n "${OFFICIAL_WARMUPS:-}" ] && official_warmups="$OFFICIAL_WARMUPS"
                [ -n "${OFFICIAL_REPEATS:-}" ] && official_repeats="$OFFICIAL_REPEATS"
                [ -n "${OFFICIAL_INPUT_MODE:-}" ] && official_input_mode="$OFFICIAL_INPUT_MODE"
                [ -n "${OFFICIAL_SEED:-}" ] && official_seed="$OFFICIAL_SEED"
                [ -n "${OFFICIAL_REPLAY_DIR:-}" ] && official_replay_dir="$OFFICIAL_REPLAY_DIR"
                [ -n "${OFFICIAL_REPLAY_FILE:-}" ] && official_replay_file="$OFFICIAL_REPLAY_FILE"
                if [ -n "${RUN_TARGET:-}" ]; then
                    case "$RUN_TARGET" in
                        all) run_ours=1; run_official=1; run_pytorch=1 ;;
                        ours) run_ours=1; run_official=0; run_pytorch=0 ;;
                        official) run_ours=0; run_official=1; run_pytorch=0 ;;
                        pytorch) run_ours=0; run_official=0; run_pytorch=1 ;;
                        *)
                            log_error "Invalid RUN_TARGET in config: $RUN_TARGET (expected all|ours|official|pytorch)"
                            exit 1
                            ;;
                    esac
                fi
                if [ -n "${PROFILE_SHAPE:-}" ]; then
                    IFS='x' read -r ps_b ps_h ps_n ps_d <<< "$PROFILE_SHAPE"
                    if ! echo "$ps_b $ps_h $ps_n $ps_d" | grep -Eq '^[0-9]+ [0-9]+ [0-9]+ [0-9]+$'; then
                        log_error "Invalid PROFILE_SHAPE in config: $PROFILE_SHAPE (expected BxHxNxD)"
                        exit 1
                    fi
                    bench_config="${ps_b}x${ps_h}x${ps_n}x${ps_d}"
                    ncu_config="${ps_b}x${ps_h}x${ps_n}x${ps_d}"
                    official_batch_size="$ps_b"
                    official_n_heads="$ps_h"
                    official_seq_lens="$ps_n"
                    official_d_heads="$ps_d"
                fi
                shift 2
                ;;
            --bench-config)
                bench_config="$2"
                shift 2
                ;;
            --run-benchmark)
                run_benchmark=1
                shift 1
                ;;
            --skip-benchmark)
                run_benchmark=0
                shift 1
                ;;
            --run-ncu)
                run_ncu=1
                shift 1
                ;;
            --skip-ncu)
                run_ncu=0
                shift 1
                ;;
            --ncu-config)
                ncu_config="$2"
                shift 2
                ;;
            --run-official)
                run_official=1
                shift 1
                ;;
            --run-pytorch)
                run_pytorch=1
                shift 1
                ;;
            --strict)
                strict_mode=1
                shift 1
                ;;
            --only-ours)
                run_ours=1
                run_official=0
                run_pytorch=0
                shift 1
                ;;
            --only-official)
                run_ours=0
                run_official=1
                run_benchmark=0
                run_pytorch=0
                shift 1
                ;;
            --only-pytorch)
                run_ours=0
                run_official=0
                run_benchmark=0
                run_pytorch=1
                shift 1
                ;;
            --skip-official)
                run_official=0
                shift 1
                ;;
            --skip-pytorch)
                run_pytorch=0
                shift 1
                ;;
            --official-dtype)
                official_dtype="$2"
                shift 2
                ;;
            --official-pytorch-dtype)
                official_pytorch_dtype="$2"
                shift 2
                ;;
            --official-seq-lens)
                official_seq_lens="$2"
                shift 2
                ;;
            --official-d-heads)
                official_d_heads="$2"
                shift 2
                ;;
            --official-batch-size)
                official_batch_size="$2"
                shift 2
                ;;
            --official-n-heads)
                official_n_heads="$2"
                shift 2
                ;;
            --official-warmups)
                official_warmups="$2"
                shift 2
                ;;
            --official-repeats)
                official_repeats="$2"
                shift 2
                ;;
            --official-input-mode)
                official_input_mode="$2"
                shift 2
                ;;
            --official-seed)
                official_seed="$2"
                shift 2
                ;;
            --official-replay-dir)
                official_replay_dir="$2"
                shift 2
                ;;
            --official-replay-file)
                official_replay_file="$2"
                shift 2
                ;;
            -*)
                log_error "Unknown argument: $1"
                log_info "Usage: $0 remote [user@host] [-p port] [--config path] [--ssh-key ~/.ssh/id_rsa] [--remote-path path] [--remote-arch 80|90] [--bench-config tiny|small|medium|large|BxHxNxD|all] [--ncu-config tiny|small|medium|large|BxHxNxD|all] [--run-benchmark|--skip-benchmark] [--run-ncu|--skip-ncu] [--run-official|--skip-official|--run-pytorch|--skip-pytorch|--only-ours|--only-official|--only-pytorch] [--strict] [--official-dtype fp16|bf16] [--official-pytorch-dtype match|fp32] [--official-seq-lens 4096] [--official-d-heads 128] [--official-batch-size 8] [--official-n-heads 16] [--official-warmups 10] [--official-repeats 50] [--official-input-mode random|structured|stress|replay] [--official-seed 1234] [--official-replay-dir benchmark/data/replay] [--official-replay-file replay.pt]"
                exit 1
                ;;
            *)
                if [ -z "$remote_addr" ]; then
                    remote_addr="$1"
                    shift 1
                else
                    log_error "Unknown positional argument: $1"
                    exit 1
                fi
                ;;
        esac
    done

    if [ -z "$bench_config" ]; then
        bench_config="all"
    fi
    if ! echo "$bench_config" | grep -Eq '^(all|tiny|small|medium|large|[1-9][0-9]*x[1-9][0-9]*x[1-9][0-9]*x[1-9][0-9]*)$'; then
        log_error "Invalid --bench-config: $bench_config"
        exit 1
    fi
    if ! echo "$ncu_config" | grep -Eq '^(all|tiny|small|medium|large|[1-9][0-9]*x[1-9][0-9]*x[1-9][0-9]*x[1-9][0-9]*)$'; then
        log_error "Invalid --ncu-config: $ncu_config"
        exit 1
    fi
    for f in "$run_ours" "$run_official" "$run_pytorch" "$run_benchmark" "$run_ncu" "$strict_mode"; do
        if ! echo "$f" | grep -Eq '^[01]$'; then
            log_error "Run flags must be 0/1, got: $f"
            exit 1
        fi
    done
    if [ "$run_ours" = "0" ] && [ "$run_official" = "0" ] && [ "$run_pytorch" = "0" ]; then
        log_error "Nothing to run: run_ours=0, run_official=0, run_pytorch=0"
        exit 1
    fi
    if ! echo "$official_dtype" | grep -Eq '^(fp16|bf16)$'; then
        log_error "Invalid --official-dtype: $official_dtype (expected fp16 or bf16)"
        exit 1
    fi
    if ! echo "$official_pytorch_dtype" | grep -Eq '^(match|fp32)$'; then
        log_error "Invalid --official-pytorch-dtype: $official_pytorch_dtype (expected match or fp32)"
        exit 1
    fi
    if ! echo "$official_batch_size" | grep -Eq '^[1-9][0-9]*$'; then
        log_error "Invalid --official-batch-size: $official_batch_size"
        exit 1
    fi
    if ! echo "$official_n_heads" | grep -Eq '^[1-9][0-9]*$'; then
        log_error "Invalid --official-n-heads: $official_n_heads"
        exit 1
    fi
    if ! echo "$official_warmups" | grep -Eq '^[0-9]+$'; then
        log_error "Invalid --official-warmups: $official_warmups"
        exit 1
    fi
    if ! echo "$official_repeats" | grep -Eq '^[1-9][0-9]*$'; then
        log_error "Invalid --official-repeats: $official_repeats"
        exit 1
    fi
    if ! echo "$official_input_mode" | grep -Eq '^(random|structured|stress|replay)$'; then
        log_error "Invalid --official-input-mode: $official_input_mode"
        exit 1
    fi
    if ! echo "$official_seed" | grep -Eq '^[0-9]+$'; then
        log_error "Invalid --official-seed: $official_seed"
        exit 1
    fi

    local bench_arg=""
    if [ "$bench_config" != "all" ]; then
        bench_arg="$bench_config"
    fi
    local ncu_arg=""
    if [ "$ncu_config" != "all" ]; then
        ncu_arg="$ncu_config"
    fi

    if [ -z "$remote_addr" ]; then
        remote_addr="$DEFAULT_REMOTE_ADDR"
        log_info "Using default remote target: $remote_addr"
    fi

    local ssh_key="$DEFAULT_SSH_KEY"
    if [ -n "$ssh_key_override" ]; then
        ssh_key="$ssh_key_override"
    fi
    local ssh_opts=("-p" "$remote_port")
    local scp_opts=("-P" "$remote_port")

    if [ ! -f "$ssh_key" ]; then
        log_error "SSH key not found: $ssh_key"
        exit 1
    fi
    ssh_opts+=("-i" "$ssh_key")
    scp_opts+=("-i" "$ssh_key")

    log_info "Remote target: $remote_addr"
    log_info "Port: $remote_port"
    log_info "Remote path: $remote_path"
    [ -n "$config_file" ] && log_info "Config file: $config_file"
    log_info "Bench config: $bench_config"
    log_info "NCU config: $ncu_config"
    log_info "NCU set: $NCU_SET"
    log_info "NCU launch skip: $NCU_LAUNCH_SKIP"
    log_info "NCU launch count: $NCU_LAUNCH_COUNT"
    log_info "Run ours profile_kernel: $run_ours"
    log_info "Run benchmark: $run_benchmark"
    log_info "Run ncu: $run_ncu"
    log_info "Run official benchmark: $run_official"
    log_info "Run pytorch benchmark: $run_pytorch"
    log_info "Strict mode: $strict_mode"
    log_info "Official dtype: $official_dtype"
    log_info "Official pytorch dtype: $official_pytorch_dtype"
    log_info "Official seq_lens: $official_seq_lens"
    log_info "Official d_heads: $official_d_heads"
    log_info "Official batch_size: $official_batch_size"
    log_info "Official n_heads: $official_n_heads"
    log_info "Official warmups: $official_warmups"
    log_info "Official repeats: $official_repeats"
    log_info "Official input_mode: $official_input_mode"
    log_info "Official seed: $official_seed"
    log_info "Official replay_dir: $official_replay_dir"
    log_info "Official replay_file: $official_replay_file"
    log_info "Using SSH key: $ssh_key"

    local remote_cc=""
    local remote_arch=""
    if [ -n "$remote_arch_override" ]; then
        if ! echo "$remote_arch_override" | grep -Eq '^[0-9]+$'; then
            log_error "--remote-arch must be integer like 80 or 90, got: $remote_arch_override"
            exit 1
        fi
        remote_arch="$remote_arch_override"
        log_info "Using overridden remote arch: $remote_arch"
    else
        remote_cc=$(ssh "${ssh_opts[@]}" "$remote_addr" \
            "env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin /usr/bin/nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1" 2>/dev/null || true)

        if echo "$remote_cc" | grep -Eq '^[0-9]+\.[0-9]+$'; then
            remote_arch=$(echo "$remote_cc" | tr -d '.')
            log_info "Remote compute capability: $remote_cc (arch=$remote_arch)"
        else
            log_warning "Failed to query compute_cap directly, got: $remote_cc"
            log_info "Trying fallback parse from GPU name"

            local remote_gpu_name
            remote_gpu_name=$(ssh "${ssh_opts[@]}" "$remote_addr" \
                "env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin /usr/bin/nvidia-smi --query-gpu=name --format=csv,noheader | head -n1" 2>/dev/null || true)

            case "$remote_gpu_name" in
                *H100*|*H200*|*GH200*) remote_arch="90" ;;
                *A100*|*A800*|*A30*|*A10*|*A40*|*RTX\ A6000*) remote_arch="80" ;;
                *)
                    log_error "Cannot determine remote arch automatically."
                    log_info "GPU name: $remote_gpu_name"
                    log_info "Use --remote-arch 80 (Ampere) or --remote-arch 90 (Hopper)."
                    log_info "Remote nvidia-smi diagnostic:"
                    ssh "${ssh_opts[@]}" "$remote_addr" "env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin /usr/bin/nvidia-smi | head -n 8 || true"
                    exit 1
                    ;;
            esac

            log_info "Remote GPU: $remote_gpu_name"
            log_info "Derived remote arch from GPU name: $remote_arch"
        fi
    fi

    print_section "Uploading and Running"
    ssh "${ssh_opts[@]}" "$remote_addr" "mkdir -p '$remote_path'"

    local src_pkg="/tmp/flash_attention_v2_src_${TIMESTAMP}.tgz"
    (
        cd "$PROJECT_ROOT"
        tar czf "$src_pkg" \
            --exclude=.git \
            --exclude=build \
            --exclude='build_*' \
            --exclude=profile_results \
            --exclude='*.ncu-rep' \
            .
    )
    scp "${scp_opts[@]}" "$src_pkg" "${remote_addr}:${remote_path}/src.tgz"
    rm -f "$src_pkg"

    ssh "${ssh_opts[@]}" "$remote_addr" "bash -lic '
set -e
cd \"$remote_path\"
echo \"Remote working directory: \$(pwd)\"

[ -f /etc/profile ] && source /etc/profile || true
[ -f ~/.bash_profile ] && source ~/.bash_profile || true
[ -f ~/.bashrc ] && source ~/.bashrc || true

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate base || true
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate base || true
fi
if [ -f ~/fa-venv/bin/activate ]; then
  source ~/fa-venv/bin/activate || true
fi

for d in /usr/local/cuda/bin /usr/local/cuda-13.0/bin /usr/local/cuda-12.4/bin /usr/local/cuda-12.3/bin /opt/conda/bin; do
  if [ -d \"\$d\" ]; then
    export PATH=\"\$d:\$PATH\"
  fi
done

if ! command -v nvcc >/dev/null 2>&1; then
  NVCC_CAND=\$(find /usr/local -maxdepth 4 -type f -name nvcc 2>/dev/null | head -n1 || true)
  if [ -n \"\$NVCC_CAND\" ]; then
    export PATH=\"\$(dirname \"\$NVCC_CAND\"):\$PATH\"
  fi
fi

if ! command -v ncu >/dev/null 2>&1; then
  NCU_CAND=\$(find /usr/local -maxdepth 5 -type f -name ncu 2>/dev/null | head -n1 || true)
  if [ -n \"\$NCU_CAND\" ]; then
    export PATH=\"\$(dirname \"\$NCU_CAND\"):\$PATH\"
  fi
fi

echo \"========================================\"
echo \"Remote System Information\"
echo \"========================================\"
env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin /usr/bin/nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || true
echo \"PATH=\$PATH\"
which nvcc || true
which ncu || true
nvcc --version | grep release || true
echo \"NCU_RUN=sudo /usr/local/cuda/bin/ncu\"
sudo /usr/local/cuda/bin/ncu --version | head -n1 || true

if command -v nvcc >/dev/null 2>&1; then
  CUDA_BIN_DIR=\$(dirname \"\$(command -v nvcc)\")
  CUDA_ROOT=\$(cd \"\$CUDA_BIN_DIR/..\" && pwd)
  export LD_LIBRARY_PATH=\"\$CUDA_ROOT/lib64:\$CUDA_ROOT/targets/x86_64-linux/lib:\${LD_LIBRARY_PATH}\"
fi

for d in /usr/local/cuda/lib64 /usr/local/cuda-12.4/lib64 /usr/local/cuda-13.0/lib64 /usr/local/cuda/targets/x86_64-linux/lib /usr/local/cuda-12.4/targets/x86_64-linux/lib /usr/local/cuda-13.0/targets/x86_64-linux/lib; do
  if [ -d \"\$d\" ]; then
    export LD_LIBRARY_PATH=\"\$d:\$LD_LIBRARY_PATH\"
  fi
done
echo \"LD_LIBRARY_PATH=\$LD_LIBRARY_PATH\"

echo \"========================================\"
if [ \"$run_ours\" = \"1\" ]; then
echo \"Building profile_kernel on remote\"
echo \"========================================\"
rm -rf src
mkdir -p src
tar xzf src.tgz -C src
cd src
cmake -S . -B build_remote \
  -DFA_BUILD_TESTS=OFF \
  -DFA_BUILD_UTILS=ON \
  -DFA_BUILD_EXAMPLES=OFF \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_CUDA_STANDARD=20 \
  -DCMAKE_CUDA_ARCHITECTURES=$remote_arch
cmake --build build_remote -j\$(nproc) --target profile_kernel

echo \"========================================\"
echo \"Running profile_kernel\"
echo \"========================================\"
if [ \"$run_benchmark\" = \"1\" ]; then
  rm -f \"$remote_bench_pid_file\" \"$remote_bench_log_file\"
  nohup ./build_remote/profile_kernel --mode benchmark $bench_arg > \"$remote_bench_log_file\" 2>&1 < /dev/null &
  echo \$! > \"$remote_bench_pid_file\"
  echo \"Benchmark started in background: pid=\$(cat \"$remote_bench_pid_file\"), log=$remote_bench_log_file\"
  while true; do
    pid=\$(cat \"$remote_bench_pid_file\" 2>/dev/null || true)
    if [ -z \"\$pid\" ] || ! kill -0 \"\$pid\" 2>/dev/null; then
      break
    fi
    echo \"Benchmark still running...\"
    sleep 20
  done
  echo \"Benchmark finished, tail log:\"
  tail -n 30 \"$remote_bench_log_file\" || true
else
  echo \"skip plain benchmark by option (--skip-benchmark or default), keep NCU run\"
fi

if [ \"$run_ncu\" = \"1\" ] && [ -x /usr/local/cuda/bin/ncu ]; then
  echo \"Running NCU command (background): sudo /usr/local/cuda/bin/ncu --set $NCU_SET --export $remote_path/profile_results_${TIMESTAMP}.ncu-rep --force-overwrite ${NCU_LAUNCH_SKIP:+--launch-skip $NCU_LAUNCH_SKIP} ${NCU_LAUNCH_COUNT:+--launch-count $NCU_LAUNCH_COUNT} ./build_remote/profile_kernel --mode ncu $ncu_arg\"
  rm -f \"$remote_ncu_pid_file\" \"$remote_ncu_log_file\"
  if [ -n \"$NCU_LAUNCH_SKIP\" ] && [ -n \"$NCU_LAUNCH_COUNT\" ]; then
    nohup sudo /usr/local/cuda/bin/ncu \
      --set \"$NCU_SET\" \
      --launch-skip \"$NCU_LAUNCH_SKIP\" \
      --launch-count \"$NCU_LAUNCH_COUNT\" \
      --export \"$remote_path/profile_results_${TIMESTAMP}.ncu-rep\" \
      --force-overwrite \
      ./build_remote/profile_kernel --mode ncu $ncu_arg > \"$remote_ncu_log_file\" 2>&1 < /dev/null &
  elif [ -n \"$NCU_LAUNCH_SKIP\" ]; then
    nohup sudo /usr/local/cuda/bin/ncu \
      --set \"$NCU_SET\" \
      --launch-skip \"$NCU_LAUNCH_SKIP\" \
      --export \"$remote_path/profile_results_${TIMESTAMP}.ncu-rep\" \
      --force-overwrite \
      ./build_remote/profile_kernel --mode ncu $ncu_arg > \"$remote_ncu_log_file\" 2>&1 < /dev/null &
  elif [ -n \"$NCU_LAUNCH_COUNT\" ]; then
    nohup sudo /usr/local/cuda/bin/ncu \
      --set \"$NCU_SET\" \
      --launch-count \"$NCU_LAUNCH_COUNT\" \
      --export \"$remote_path/profile_results_${TIMESTAMP}.ncu-rep\" \
      --force-overwrite \
      ./build_remote/profile_kernel --mode ncu $ncu_arg > \"$remote_ncu_log_file\" 2>&1 < /dev/null &
  else
    nohup sudo /usr/local/cuda/bin/ncu \
      --set \"$NCU_SET\" \
      --export \"$remote_path/profile_results_${TIMESTAMP}.ncu-rep\" \
      --force-overwrite \
      ./build_remote/profile_kernel --mode ncu $ncu_arg > \"$remote_ncu_log_file\" 2>&1 < /dev/null &
  fi
  echo \$! > \"$remote_ncu_pid_file\"
  echo \"NCU started in background: pid=\$(cat \"$remote_ncu_pid_file\"), log=$remote_ncu_log_file\"
fi
if [ \"$run_ncu\" != \"1\" ]; then
  echo \"skip ncu by option (--skip-ncu)\"
fi
else
  echo \"========================================\"
  echo \"Skip ours mode: skip profile_kernel build and NCU\"
  echo \"========================================\"
  rm -rf src
  mkdir -p src
  tar xzf src.tgz -C src
  cd src
  echo \"Remote source directory: \$(pwd)\"
fi

if [ \"$run_official\" = \"1\" ] || [ \"$run_pytorch\" = \"1\" ]; then
  echo \"========================================\"
  echo \"Running official/pytorch benchmark\"
  echo \"========================================\"
  if [ \"$run_official\" = \"0\" ] || python3 -c \"import flash_attn\" >/dev/null 2>&1; then
    kernels=both
    baseline=pytorch
    if [ \"$run_official\" = \"1\" ] && [ \"$run_pytorch\" = \"0\" ]; then
      kernels=official
      baseline=flashattn_official
    elif [ \"$run_official\" = \"0\" ] && [ \"$run_pytorch\" = \"1\" ]; then
      kernels=pytorch
      baseline=pytorch
    fi
    if ! python3 benchmark/python/benchmark_official_flashattn.py \
      --seq_lens \"$official_seq_lens\" \
      --d_heads \"$official_d_heads\" \
      --batch_size \"$official_batch_size\" \
      --n_heads \"$official_n_heads\" \
      --warmups \"$official_warmups\" \
      --repeats \"$official_repeats\" \
      --dtype \"$official_dtype\" \
      --pytorch_dtype \"$official_pytorch_dtype\" \
      --input_mode \"$official_input_mode\" \
      --seed \"$official_seed\" \
      --replay_dir \"$official_replay_dir\" \
      --replay_file \"$official_replay_file\" \
      --kernels \"\$kernels\" \
      --baseline \"\$baseline\" \
      --csv \
      --out_csv \"$remote_path/official_${TIMESTAMP}.csv\"; then
      if [ \"$strict_mode\" = \"1\" ]; then
        echo \"official benchmark failed under --strict\"
        exit 1
      fi
      echo \"official benchmark failed (non-strict mode), continue\"
    fi
  else
    if [ \"$strict_mode\" = \"1\" ]; then
      echo \"official flash-attn is not importable on remote (strict mode)\"
      exit 1
    fi
    echo \"official flash-attn is not importable on remote; skip official benchmark\"
  fi
fi
'"

    if [ "$run_ours" = "1" ] && [ "$run_ncu" = "1" ]; then
        if ssh "${ssh_opts[@]}" "$remote_addr" "test -f '$remote_ncu_pid_file'"; then
            log_info "Waiting for remote NCU process to finish..."
            while ssh "${ssh_opts[@]}" "$remote_addr" "pid=\$(cat '$remote_ncu_pid_file' 2>/dev/null || true); [ -n \"\$pid\" ] && kill -0 \"\$pid\" 2>/dev/null"; do
                log_info "Remote NCU still running..."
                sleep 20
            done
            log_info "Remote NCU finished"
            log_info "Remote NCU log tail:"
            ssh "${ssh_opts[@]}" "$remote_addr" "tail -n 30 '$remote_ncu_log_file' || true"
        fi
    fi

    if ssh "${ssh_opts[@]}" "$remote_addr" "test -f '$remote_path/profile_results_${TIMESTAMP}.ncu-rep'"; then
        scp "${scp_opts[@]}" "${remote_addr}:${remote_path}/profile_results_${TIMESTAMP}.ncu-rep" \
            "$PROFILE_DIR/remote_${TIMESTAMP}.ncu-rep"
        log_success "Remote profiling complete"
        log_info "Result: $PROFILE_DIR/remote_${TIMESTAMP}.ncu-rep"
    else
        log_warning "NCU report not found on remote host"
    fi

    if [ "$run_official" = "1" ] || [ "$run_pytorch" = "1" ]; then
        if ssh "${ssh_opts[@]}" "$remote_addr" "test -f '$remote_path/official_${TIMESTAMP}.csv'"; then
            scp "${scp_opts[@]}" "${remote_addr}:${remote_path}/official_${TIMESTAMP}.csv" \
                "$PROFILE_DIR/official_remote_${TIMESTAMP}.csv"
            log_success "Remote official/pytorch benchmark CSV synced"
            log_info "Result: $PROFILE_DIR/official_remote_${TIMESTAMP}.csv"
        else
            log_warning "Official benchmark CSV not found on remote host"
        fi
    fi
}

compare_with_official() {
    print_section "Comparing with Official Flash Attention"

    if [ -f "$SCRIPT_DIR/compare_ours_vs_official.py" ]; then
        python3 "$SCRIPT_DIR/compare_ours_vs_official.py" \
            --csv-out "$PROFILE_DIR/comparison_${TIMESTAMP}.csv"
        log_success "Comparison CSV: $PROFILE_DIR/comparison_${TIMESTAMP}.csv"
    else
        log_warning "compare_ours_vs_official.py not found"
    fi
}

main() {
    print_section "Flash Attention V2 Auto Profiler"

    local mode="$1"
    if [ -z "$mode" ]; then
        mode="local"
    fi

    case "$mode" in
        local)
            shift 1
            local local_config_file=""
            while [ $# -gt 0 ]; do
                case "$1" in
                    --config)
                        local_config_file="$2"
                        if [ ! -f "$local_config_file" ]; then
                            log_error "Config file not found: $local_config_file"
                            exit 1
                        fi
                        # shellcheck disable=SC1090
                        source "$local_config_file"
                        [ -n "${OURS_BENCH_CONFIG:-}" ] && BENCH_CONFIG="$OURS_BENCH_CONFIG"
                        [ -n "${OURS_NCU_CONFIG:-}" ] && NCU_CONFIG="$OURS_NCU_CONFIG"
                        [ -n "${RUN_BENCHMARK:-}" ] && RUN_BENCHMARK="$RUN_BENCHMARK"
                        [ -n "${RUN_NCU:-}" ] && RUN_NCU="$RUN_NCU"
                        if [ -n "${PROFILE_SHAPE:-}" ]; then
                            IFS='x' read -r ps_b ps_h ps_n ps_d <<< "$PROFILE_SHAPE"
                            if ! echo "$ps_b $ps_h $ps_n $ps_d" | grep -Eq '^[0-9]+ [0-9]+ [0-9]+ [0-9]+$'; then
                                log_error "Invalid PROFILE_SHAPE in config: $PROFILE_SHAPE (expected BxHxNxD)"
                                exit 1
                            fi
                            BENCH_CONFIG="${ps_b}x${ps_h}x${ps_n}x${ps_d}"
                            NCU_CONFIG="${ps_b}x${ps_h}x${ps_n}x${ps_d}"
                        fi
                        shift 2
                        ;;
                    --bench-config)
                        BENCH_CONFIG="$2"
                        shift 2
                        ;;
                    --run-benchmark)
                        RUN_BENCHMARK=1
                        shift 1
                        ;;
                    --skip-benchmark)
                        RUN_BENCHMARK=0
                        shift 1
                        ;;
                    --run-ncu)
                        RUN_NCU=1
                        shift 1
                        ;;
                    --skip-ncu)
                        RUN_NCU=0
                        shift 1
                        ;;
                    --ncu-config)
                        NCU_CONFIG="$2"
                        shift 2
                        ;;
                    *)
                        log_error "Unknown local argument: $1"
                        log_info "Usage: $0 local [--config path] [--bench-config tiny|small|medium|large|BxHxNxD|all] [--ncu-config tiny|small|medium|large|BxHxNxD|all] [--run-benchmark|--skip-benchmark] [--run-ncu|--skip-ncu]"
                        exit 1
                        ;;
                esac
            done
            detect_system_info
            compile_project
            run_local_profile
            compare_with_official
            ;;
        remote)
            run_remote_profile "$@"
            ;;
        compile-only)
            detect_system_info
            compile_project
            ;;
        *)
            log_error "Unknown mode: $mode"
            echo "Usage:"
            echo "  $0 local [--config path] [--bench-config tiny|small|medium|large|BxHxNxD|all] [--ncu-config tiny|small|medium|large|BxHxNxD|all] [--run-benchmark|--skip-benchmark] [--run-ncu|--skip-ncu]"
            echo "  $0 remote [user@host] [-p port] [--config path] [--ssh-key ~/.ssh/id_rsa] [--remote-path path] [--remote-arch 80|90] [--bench-config tiny|small|medium|large|BxHxNxD|all] [--ncu-config tiny|small|medium|large|BxHxNxD|all] [--run-benchmark|--skip-benchmark] [--run-ncu|--skip-ncu] [--run-official|--skip-official|--run-pytorch|--skip-pytorch|--only-ours|--only-official|--only-pytorch] [--strict] [--official-dtype fp16|bf16] [--official-pytorch-dtype match|fp32] [--official-seq-lens 4096] [--official-d-heads 128] [--official-batch-size 8] [--official-n-heads 16] [--official-warmups 10] [--official-repeats 50] [--official-input-mode random|structured|stress|replay] [--official-seed 1234] [--official-replay-dir benchmark/data/replay] [--official-replay-file replay.pt]"
            echo "    default user@host: $DEFAULT_REMOTE_ADDR"
            echo "    default ssh key:   $DEFAULT_SSH_KEY"
            echo "  $0 compile-only"
            exit 1
            ;;
    esac

    print_section "Profiling Complete"
    log_success "Results dir: $PROFILE_DIR"
}

main "$@"

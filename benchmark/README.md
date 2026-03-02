# Flash Attention Benchmark / Profile Guide

This document describes the current benchmark and profiling pipeline under `benchmark/` and `scripts/`.

## Directory

```text
benchmark/
  cpp/
    benchmark.cu                  # profile_kernel source (C++ benchmark + NCU mode)
    correctness.cu                # standalone C++ correctness helper
    include/
      perf_metrics.cuh
      test_configs.h
  python/
    benchmark_pybind.py           # pybind vs pytorch
    benchmark_official_flashattn.py # official flash-attn vs pytorch
    input_data.py                 # random/structured/stress/replay input generation
    tests/
      test_pybind_interface.py
      test_pybind_correctness.py
      test_pybind_vs_official_correctness.py
  data/replay/                    # replay input placeholder

scripts/
  run_python_benchmarks.sh
  run_local_perf_gate.sh
  run_correctness_gate.sh
  run_compare_official_correctness.sh
  smoke_python_bench.sh
  auto_profile.sh
  remote_config.example
```

## Build (C++ profile kernel)

`profile_kernel` is built from the root CMake project (recommended path).

```bash
cmake -S . -B build \
  -DFA_BUILD_UTILS=ON \
  -DFA_BUILD_TESTS=OFF \
  -DFA_BUILD_EXAMPLES=OFF && \
cmake --build build -j --target profile_kernel
```

Optional architecture pin:

```bash
cmake -S . -B build \
  -DFA_BUILD_UTILS=ON \
  -DFA_BUILD_TESTS=OFF \
  -DFA_BUILD_EXAMPLES=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=80 && \
cmake --build build -j --target profile_kernel
```

## C++ benchmark usage (`profile_kernel`)

Binary:

```bash
./build/profile_kernel --help
```

Modes:
- `--mode benchmark` (default): warmup+repeat timing
- `--mode ncu`: profiling-friendly defaults (`warmup=0`, `benchmark=1`)

Examples:

```bash
./build/profile_kernel \
  --mode benchmark \
  16x16x4096x128
```

```bash
./build/profile_kernel \
  --mode ncu \
  16x16x4096x128
```

```bash
./build/profile_kernel \
  --mode benchmark \
  --warmup-iters 10 \
  --benchmark-iters 100 \
  large
```

Accepted positional config:
- preset name: `tiny|small|medium|large|all`
- custom shape: `BxHxNxD` (for example `16x16x4096x128`)

## Python benchmark usage

Set import path first:

```bash
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
```

### 1) Our pybind vs PyTorch

```bash
python3 benchmark/python/benchmark_pybind.py \
  --seq_lens 1024,2048 \
  --d_heads 64,128 \
  --batch_size 16 \
  --n_heads 16 \
  --warmups 10 \
  --repeats 50 \
  --dtype fp16 \
  --baseline pytorch
```

Notes:
- `--dtype fp16` currently follows `forward_v3_tensor_core` path and only supports `d_head=128`.
- CSV is saved to `profile_results/pybind_<timestamp>.csv` unless `--no_save`.

### 2) Official flash-attn / PyTorch benchmark

```bash
python3 benchmark/python/benchmark_official_flashattn.py \
  --seq_lens 4096 \
  --d_heads 128 \
  --batch_size 16 \
  --n_heads 16 \
  --dtype fp16 \
  --kernels both \
  --baseline pytorch
```

Kernel selection:
- `--kernels both`
- `--kernels official`
- `--kernels pytorch`

Baseline constraints:
- `baseline=pytorch` requires `kernels=both|pytorch`
- `baseline=flashattn_official` requires `kernels=both|official`

Official package import path defaults to:
- `~/flash-attention`

## Input data modes

Both python benchmark scripts support:
- `--input_mode random`
- `--input_mode structured`
- `--input_mode stress`
- `--input_mode replay --replay_dir benchmark/data/replay [--replay_file xxx.pt]`

## Correctness gates

Pybind vs PyTorch:

```bash
./scripts/run_correctness_gate.sh \
  --dtype fp16 \
  --input_mode random \
  --seed 1234 \
  --cases "2x4x128x128,2x4x256x128,2x4x512x128"
```

Pybind vs official flash-attn:

```bash
./scripts/run_compare_official_correctness.sh \
  --dtype fp16 \
  --input_mode random \
  --seed 1234 \
  --cases "2x4x128x128,2x4x256x128,2x4x512x128" \
  --official_path "$HOME/flash-attention"
```

Smoke check:

```bash
./scripts/smoke_python_bench.sh
```

## Combined python runner

`run_python_benchmarks.sh` always runs pybind; official benchmark is optional.

Pybind only:

```bash
./scripts/run_python_benchmarks.sh \
  --seq_lens 4096 \
  --d_heads 128 \
  --batch_size 16 \
  --n_heads 16 \
  --pybind_dtype fp16 \
  --skip_official
```

Pybind + official + merged CSV:

```bash
./scripts/run_python_benchmarks.sh \
  --seq_lens 4096 \
  --d_heads 128 \
  --batch_size 16 \
  --n_heads 16 \
  --pybind_dtype fp16 \
  --official_dtype fp16 \
  --run_official
```

## Local performance gate (auto build + benchmark + optional NCU)

This script is intended for fast local iteration and logging:
- rebuilds `profile_kernel`
- runs pybind vs pytorch benchmark
- emits a summary CSV row with:
  - `time, commit, gpu, shape, variant`
  - `mean_ms/tflops/rel_perf`
  - `regs/spill`
  - `achieved_occ_pct/global_load_B_per_sector/global_store_B_per_sector/top_stall` (empty when NCU is skipped)

```bash
./scripts/run_local_perf_gate.sh \
  --seq_len 4096 \
  --d_head 128 \
  --batch_size 16 \
  --n_heads 16 \
  --dtype fp16 \
  --warmups 5 \
  --repeats 20
```

Single-command mode with config:

```bash
./scripts/run_local_perf_gate.sh \
  --config scripts/local_perf_gate.example
```

Each run creates a timestamped directory:

```text
profile_results/gate/<prefix>_<YYYYmmdd_HHMMSS>/
```

Outputs:
- `profile_results/gate/compile_<ts>.log`
- `profile_results/gate/pybind_<ts>.csv`
- `profile_results/gate/ncu_<ts>.csv` (if enabled)
- `profile_results/gate/summary_<ts>.csv`
- `profile_results/gate_history.csv` (append-only aggregate)

## Local / remote auto profiling

`scripts/auto_profile.sh` is the orchestrator for:
- local C++ benchmark run
- local NCU run
- remote compile + benchmark + NCU + result sync
- optional official/pytorch python benchmark on remote

Local examples:

```bash
./scripts/auto_profile.sh local \
  --bench-config 16x16x4096x128 \
  --ncu-config 16x16x4096x128 \
  --run-benchmark \
  --run-ncu
```

```bash
./scripts/auto_profile.sh local \
  --bench-config 16x16x4096x128 \
  --skip-benchmark \
  --run-ncu
```

Remote examples:

```bash
./scripts/auto_profile.sh remote ubuntu@<host> \
  --ssh-key ~/.ssh/id_rsa \
  --remote-arch 80 \
  --only-ours \
  --skip-benchmark \
  --run-ncu \
  --ncu-config 16x16x4096x128
```

```bash
./scripts/auto_profile.sh remote \
  --config scripts/remote_config.example
```

`scripts/remote_config.example` is the canonical config template. It supports:
- target host/path/key/arch
- run target selection (`all|ours|official|pytorch`)
- unified `PROFILE_SHAPE=BxHxNxD`
- official benchmark dtype/shape/input-mode controls
- NCU set / launch-skip / launch-count

## Output files

Common output location:
- `profile_results/`

Typical files:
- `pybind_<timestamp>.csv`
- `official_<timestamp>.csv`
- `merged_<timestamp>.csv`
- `correctness/gate_<timestamp>.json|csv`
- `correctness/pybind_vs_official_<timestamp>.json|csv`
- `ncu_local_<timestamp>.ncu-rep`
- `remote_<timestamp>.ncu-rep`
- `official_remote_<timestamp>.csv`

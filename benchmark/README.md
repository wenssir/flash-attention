# Flash Attention Benchmark Framework

## 📁 Directory Structure

```
benchmark/
├── cpp/                 # C++ benchmark programs (optional legacy path)
│   ├── benchmark.cu      # Main performance benchmark
│   ├── correctness.cu    # Correctness test
│   ├── include/
│   │   ├── perf_metrics.cuh
│   │   └── test_configs.h
│   └── CMakeLists.txt
├── python/              # Python benchmark scripts
│   ├── benchmark_pybind.py
│   ├── benchmark_official_flashattn.py
│   └── input_data.py
└── data/
    └── replay/          # placeholder for replay datasets
```

## 🚀 Quick Start

### 1. Build

```bash
cmake -S benchmark/cpp -B build_benchmark_cpp
cmake --build build_benchmark_cpp -j
```

### 2. Run Benchmark

```bash
# Run all configs
./build_benchmark_cpp/benchmark

# Run specific config
./build_benchmark_cpp/benchmark medium
```

### 3. Run Correctness Test

```bash
./build_benchmark_cpp/correctness
```

### 4. Benchmark Local Pybind Interface

```bash
PYTHONPATH=python python3 benchmark/python/benchmark_pybind.py --seq_lens 1024,2048 --d_heads 64,128
```

By default, results are also saved to `profile_results/pybind_<timestamp>.csv`.
Use `--out_csv <path>` to override, or `--no_save` to disable file output.

Benchmark official flash-attn separately:

```bash
python3 benchmark/python/benchmark_official_flashattn.py --seq_lens 1024,2048 --d_heads 64,128 --dtype fp16
```

By default, results are also saved to `profile_results/official_<timestamp>.csv`.

Choose relative-performance baseline (100% reference):

```bash
python3 benchmark/python/benchmark_pybind.py --baseline fa2_pybind
python3 benchmark/python/benchmark_official_flashattn.py --baseline flashattn_official
```

Input data strategies:

```bash
# random (default)
python3 benchmark/python/benchmark_pybind.py --input_mode random --seed 1234

# structured deterministic pattern
python3 benchmark/python/benchmark_pybind.py --input_mode structured

# stress with sparse outliers
python3 benchmark/python/benchmark_pybind.py --input_mode stress --seed 1234

# replay from benchmark/data/replay/*.pt (or pass --replay_file)
python3 benchmark/python/benchmark_pybind.py --input_mode replay --replay_dir benchmark/data/replay
```

Run one-command smoke check (syntax + interface + correctness + tiny benchmark):

```bash
./scripts/smoke_python_bench.sh
```

Run correctness gate only (non-zero exit on fail, emits JSON/CSV report):

```bash
./scripts/run_correctness_gate.sh
```

Run pybind + official benchmark and merge CSV automatically:

```bash
./scripts/run_python_benchmarks.sh --seq_lens 1024,2048 --d_heads 64,128
```

Merge existing CSV files only:

```bash
python3 scripts/merge_bench_csv.py --pybind_csv profile_results/pybind_xxx.csv --official_csv profile_results/official_xxx.csv --out_csv profile_results/merged_xxx.csv
```

### 5. Profile with Nsight Compute

Use `scripts/auto_profile.sh` for local/remote NCU workflows.

## 📊 Test Configs

| Config | B  | H  | N    | d   |
|---------|-----|-----|------|-----|
| tiny    | 2   | 4   | 512  | 64  |
| small   | 4   | 8   | 1024 | 128 |
| medium  | 8   | 16  | 2048 | 128 |
| large   | 8   | 16  | 4096 | 128 |

## 📝 Output Format

### Console Output
```
========================================
Config: medium
========================================
Shape: B=8, H=16, N=2048, d=128
Time:  5.234 ms
GFLOPS: 328.21
Memory: 4.00 GB
Bandwidth: 764.45 GB/s
```

### CSV Output
```
CSV,8,16,2048,128,5.234,328.21,65.2,4.00,764.45,49.2,52.43,52428.57
```

## 🔧 Performance Metrics

- **GFLOPS**: Billions of floating point operations per second
- **Compute Efficiency**: Actual FLOPS / A100 peak (156 TFLOPS)
- **Bandwidth**: Memory transferred per second
- **Bandwidth Utilization**: Actual bandwidth / A100 peak (1555 GB/s)
- **Throughput**: Tokens processed per second

## 🏷️  Version Management

Use git tags to track different versions:

```bash
# Tag current version
git tag -a v1.0-baseline -m "Initial baseline performance"

# Checkout specific version
git checkout v1.0-baseline

# Compare versions
git diff v1.0-baseline v2.0-optimized
```

## 📖 References

- [DAO Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [A100 Architecture](https://www.nvidia.com/content/dam/en-zz/NVIDIA/downloads/nvidia-architecture-whitepapers/a100-whitepaper.pdf)

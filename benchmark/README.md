# Flash Attention Benchmark / Profile Guide

当前 pipeline 统一为：
- `benchmark/api/*`：执行与聚合接口（核心）
- `scripts/local/run.sh`：本地编排入口
- `scripts/remote/run.sh`：远程编排入口
- `scripts/auto_profile.sh`：兼容转发入口（local/remote）

## Directory

```text
benchmark/
  api/
    runner.py                  # 统一执行入口（pybind/official/pytorch/profile_kernel/ncu）
    collect.py                 # CSV 聚合 + profile_kernel 日志转换
    schema.py                  # 统一 CSV 字段定义
  cpp/
    benchmark.cu               # profile_kernel source
    correctness.cu
    include/
      perf_metrics.cuh
      test_configs.h
  python/
    benchmark_pybind.py
    benchmark_official_flashattn.py
    input_data.py
    tests/
      test_pybind_interface.py
      test_pybind_correctness.py
      test_pybind_vs_official_correctness.py
  data/replay/

scripts/
  local/
    run.sh                     # 本地 pipeline
  remote/
    run.sh                     # 远程上传/执行/回传
  auto_profile.sh              # compatibility wrapper
  remote_config.example        # 配置模板
```

## 默认行为

默认仅跑 `pybind`。`official` / `pytorch` / `profile_kernel` / `ncu` 都是可选开关。  
无论开启哪些任务，都会汇总到同一个文件：

- `profile_results/<run_id>/results.csv`

## Local

```bash
./scripts/local/run.sh \
  --config scripts/remote_config.example
```

或通过兼容入口：

```bash
./scripts/auto_profile.sh local \
  --config scripts/remote_config.example
```

## Remote

```bash
./scripts/remote/run.sh \
  --config scripts/remote_config.example
```

或通过兼容入口：

```bash
./scripts/auto_profile.sh remote \
  --config scripts/remote_config.example
```

## Build profile_kernel only

```bash
cmake -S . -B build \
  -DFA_BUILD_UTILS=ON \
  -DFA_BUILD_TESTS=OFF \
  -DFA_BUILD_EXAMPLES=OFF && \
cmake --build build -j --target profile_kernel
```

## Output

每次 run 目录：

```text
profile_results/<run_id>/
  result.json
  results.csv
  logs/
  tables/
  artifacts/
```

`results.csv` 为统一 schema，包含：
- 时延统计（mean/median/min/max/stddev）
- 相对性能（rel_perf_pct）
- 吞吐与 FLOPs、估算带宽
- 环境元数据（gpu/torch/cuda/git_commit）

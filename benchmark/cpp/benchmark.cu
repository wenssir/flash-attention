#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>

#include "../../src/config/config.cuh"
#include "../../src/forward/forward_v5_mma.cuh"
#include "../../src/utils/util_func.cuh"
#include "include/test_configs.h"
#include "include/perf_metrics.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

using Element = __half;
using Config = config::ForwardConfig;

void print_gpu_info() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        printf("\n========================================\n");
        printf("GPU %d Information\n", i);
        printf("========================================\n");
        printf("Name: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0*1024*1024));
        printf("Shared Memory per Block: %.1f KB\n", prop.sharedMemPerBlock / 1024.0);
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("SM Count: %d\n", prop.multiProcessorCount);
        printf("L2 Cache Size: %.1f MB\n", prop.l2CacheSize / (1024.0*1024));
    }
}

struct BenchmarkResult {
    TestConfig config;
    float time_ms;
    PerformanceMetrics metrics;
};

BenchmarkResult run_benchmark(const TestConfig& cfg, const DevicePeakSpecs& peak_specs,
                              int warmup_iters = 10, int benchmark_iters = 100) {
    const int B = cfg.B, H = cfg.H, N = cfg.N, d = cfg.d;

    // Allocate device memory
    Element *d_Q, *d_K, *d_V, *d_O;
    size_t total_elements = B * H * N * d;
    size_t bytes_per_element = sizeof(Element);

    CUDA_CHECK(cudaMalloc(&d_Q, total_elements * bytes_per_element));
    CUDA_CHECK(cudaMalloc(&d_K, total_elements * bytes_per_element));
    CUDA_CHECK(cudaMalloc(&d_V, total_elements * bytes_per_element));
    CUDA_CHECK(cudaMalloc(&d_O, total_elements * bytes_per_element));

    // Initialize with random data
    std::vector<Element> h_Q(total_elements);
    std::vector<Element> h_K(total_elements);
    std::vector<Element> h_V(total_elements);

    for (size_t i = 0; i < total_elements; ++i) {
        h_Q[i] = __float2half((float)rand() / RAND_MAX);
        h_K[i] = __float2half((float)rand() / RAND_MAX);
        h_V[i] = __float2half((float)rand() / RAND_MAX);
    }

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), total_elements * bytes_per_element, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), total_elements * bytes_per_element, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), total_elements * bytes_per_element, cudaMemcpyHostToDevice));

    // Prepare kernel args
    config::ForwardKernelArgs args;
    args.Q = d_Q;
    args.K = d_K;
    args.V = d_V;
    args.O = d_O;

    args.stride_batch_q = H * N * d;
    args.stride_seq_q = d;
    args.stride_head_q = N * d;

    args.stride_batch_k = H * N * d;
    args.stride_seq_k = d;
    args.stride_head_k = N * d;

    args.stride_batch_v = H * N * d;
    args.stride_seq_v = d;
    args.stride_head_v = N * d;

    args.stride_batch_o = H * N * d;
    args.stride_seq_o = d;
    args.stride_head_o = N * d;

    args.seq_len = N;
    args.heads = H;
    args.head_dim = d;

    args.softmax_scale = 1.0f / sqrtf(d);
    args.causal = 0;

    // Compute grid dimensions
    dim3 grid((N + Config::BlockM - 1) / Config::BlockM, H, B);
    dim3 block(Config::NThreads);
    size_t smem_size = forward::forward_smem_bytes<Config>();
    CUDA_CHECK(cudaFuncSetAttribute(
        forward::flash_attention_forward_v5_mma<Config>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_size)));
    CUDA_CHECK(cudaFuncSetAttribute(
        forward::flash_attention_forward_v5_mma<Config>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100));

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        forward::flash_attention_forward_v5_mma<Config><<<grid, block, smem_size>>>(args);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < benchmark_iters; i++) {
        forward::flash_attention_forward_v5_mma<Config><<<grid, block, smem_size>>>(args);
    }
    // Catch async launch/runtime errors before consuming timing results.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_time_ms = total_ms / benchmark_iters;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Calculate metrics
    auto metrics = calculate_flash_attention_metrics(B, H, N, d, avg_time_ms, peak_specs, bytes_per_element);

    // Cleanup
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    BenchmarkResult result;
    result.config = cfg;
    result.time_ms = avg_time_ms;
    result.metrics = metrics;

    return result;
}

bool parse_shape_config(const std::string& text, TestConfig& cfg_out) {
    int b = 0, h = 0, n = 0, d = 0;
    if (sscanf(text.c_str(), "%dx%dx%dx%d", &b, &h, &n, &d) != 4) {
        return false;
    }
    if (b <= 0 || h <= 0 || n <= 0 || d <= 0) {
        return false;
    }
    cfg_out = {"custom", b, h, n, d};
    return true;
}

int main(int argc, char* argv[]) {
    printf("========================================\n");
    printf("  Flash Attention Benchmark\n");
    printf("========================================\n");

    print_gpu_info();
    int current_device = 0;
    CUDA_CHECK(cudaGetDevice(&current_device));
    cudaDeviceProp current_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&current_prop, current_device));
    DevicePeakSpecs peak_specs = infer_device_peak_specs(current_prop, current_device);

    // Parse command line
    enum class RunMode { Benchmark, Ncu };
    RunMode run_mode = RunMode::Benchmark;
    int warmup_iters = 10;
    int benchmark_iters = 100;
    bool warmup_explicit = false;
    bool benchmark_explicit = false;
    std::string config_token;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printf("Usage: %s [config_name|BxHxNxD] [--mode benchmark|ncu] [--warmup-iters N] [--benchmark-iters N]\n", argv[0]);
            printf("  mode=benchmark default: warmup=10, benchmark=100\n");
            printf("  mode=ncu default: warmup=0, benchmark=1\n");
            return 0;
        }
        if (arg == "--mode") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--mode requires value benchmark|ncu\n");
                return 1;
            }
            const std::string mode = argv[++i];
            if (mode == "benchmark") {
                run_mode = RunMode::Benchmark;
            } else if (mode == "ncu") {
                run_mode = RunMode::Ncu;
            } else {
                fprintf(stderr, "Invalid --mode: %s (expected benchmark|ncu)\n", mode.c_str());
                return 1;
            }
            continue;
        }
        if (arg == "--warmup-iters") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--warmup-iters requires integer value\n");
                return 1;
            }
            warmup_iters = std::max(0, atoi(argv[++i]));
            warmup_explicit = true;
            continue;
        }
        if (arg == "--benchmark-iters") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--benchmark-iters requires integer value\n");
                return 1;
            }
            benchmark_iters = std::max(1, atoi(argv[++i]));
            benchmark_explicit = true;
            continue;
        }
        if (arg.rfind("--", 0) == 0) {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            return 1;
        }
        if (!config_token.empty()) {
            fprintf(stderr, "Unexpected extra positional argument: %s\n", arg.c_str());
            return 1;
        }
        config_token = arg;
    }

    if (run_mode == RunMode::Ncu) {
        if (!warmup_explicit) warmup_iters = 0;
        if (!benchmark_explicit) benchmark_iters = 1;
    }

    bool run_all = true;
    std::string specific_config;
    bool run_custom_shape = false;
    TestConfig custom_cfg{};

    if (!config_token.empty()) {
        specific_config = config_token;
        if (parse_shape_config(specific_config, custom_cfg)) {
            run_custom_shape = true;
            run_all = false;
            printf("\nRunning custom shape: B=%d H=%d N=%d D=%d\n",
                   custom_cfg.B, custom_cfg.H, custom_cfg.N, custom_cfg.d);
        } else {
            run_all = false;
            printf("\nRunning specific config: %s\n", specific_config.c_str());
        }
    } else {
        printf("\nRunning all benchmark configurations...\n");
        printf("Usage: %s [config_name|BxHxNxD] [--mode benchmark|ncu] [--warmup-iters N] [--benchmark-iters N]\n", argv[0]);
    }
    printf("Run mode: %s, warmup_iters=%d, benchmark_iters=%d\n",
           (run_mode == RunMode::Ncu ? "ncu" : "benchmark"),
           warmup_iters, benchmark_iters);

    // Get configs
    auto configs = get_default_configs();
    if (run_custom_shape) {
        configs = {custom_cfg};
    }

    std::vector<BenchmarkResult> results;

    // Run benchmarks
    for (const auto& cfg : configs) {
        if (!run_all && !run_custom_shape && cfg.name != specific_config) {
            continue;
        }

        printf("\n========================================\n");
        printf("Benchmarking: %s\n", cfg.name.c_str());
        printf("========================================\n");

        try {
            auto result = run_benchmark(cfg, peak_specs, warmup_iters, benchmark_iters);
            result.metrics.print_report();
            result.metrics.print_csv();
            results.push_back(result);
        } catch (const std::exception& e) {
            fprintf(stderr, "Error benchmarking %s: %s\n", cfg.name.c_str(), e.what());
        }
    }

    if (results.empty()) {
        fprintf(stderr, "\nNo benchmark case was executed.\n");
        return 1;
    }

    // Print summary
    if (results.size() > 1) {
        printf("\n========================================\n");
        printf("Summary\n");
        printf("========================================\n");
        printf("%-15s %8s %8s %8s %8s %12s %12s %15s\n",
               "Config", "B", "H", "N", "d", "Time(ms)", "GFLOPS", "Bandwidth(GB/s)");
        printf("%-15s %8s %8s %8s %8s %12s %12s %15s\n",
               "-------", "--", "--", "--", "--", "-------", "------", "---------------");

        for (const auto& r : results) {
            printf("%-15s %8d %8d %8d %8d %12.3f %12.2f %15.2f\n",
                   r.config.name.c_str(), r.config.B, r.config.H, r.config.N, r.config.d,
                   r.time_ms, r.metrics.gflops, r.metrics.bandwidth_gb_s);
        }
    }

    printf("\n========================================\n");
    printf("Benchmark Complete!\n");
    printf("========================================\n");

    return 0;
}

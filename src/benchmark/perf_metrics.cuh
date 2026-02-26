#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================
// A100 GPU 硬件参数
// ============================================================
struct A100Specs {
    // Tensor Core 性能峰值（FP16，稠密）
    static constexpr double TENSOR_CORE_FP16_DENSE_TFLOPS = 156.0;
    static constexpr double TENSOR_CORE_FP16_DENSE_GFLOPS = 156000.0;

    // HBM 带宽峰值
    static constexpr double HBM_BANDWIDTH_TB_S = 1.555;
    static constexpr double HBM_BANDWIDTH_GB_S = 1555.0;

    // 显存容量
    static constexpr double MEMORY_GB = 40.0;
};

// ============================================================
// 性能指标计算
// ============================================================
struct PerformanceMetrics {
    // 输入配置
    int B, H, N, d;
    size_t bytes_per_element;  // FP16=2, FP32=4

    // 测量结果
    float time_ms;

    // 计算的指标
    double theoretical_flops;      // 理论 FLOPs
    double gflops;                // 实际 GFLOPS
    double compute_efficiency;     // 计算效率 (%)

    double memory_bytes;           // 理论最小内存访问量
    double memory_gb;             // 同上，单位 GB
    double bandwidth_gb_s;        // 实际带宽
    double bandwidth_utilization;  // 带宽利用率 (%)

    double throughput_tokens_ms;   // 吞吐量 (tokens/ms)
    double throughput_tokens_sec;   // 吞吐量 (tokens/s)

    // 打印报告
    void print_report() const {
        printf("\n========================================\n");
        printf("Performance Report\n");
        printf("========================================\n");
        printf("Config: B=%d, H=%d, N=%d, d=%d\n", B, H, N, d);
        printf("Data Type: %s\n", bytes_per_element == 2 ? "FP16" : "FP32");
        printf("\n");

        printf("--- Execution Time ---\n");
        printf("  Time: %.3f ms\n", time_ms);
        printf("\n");

        printf("--- Compute Performance ---\n");
        printf("  Theoretical FLOPs: %.3e\n", theoretical_flops);
        printf("  Actual GFLOPS: %.2f\n", gflops);
        printf("  Compute Efficiency: %.1f%% (%.2f / %.0f GFLOPS)\n",
               compute_efficiency, gflops, A100Specs::TENSOR_CORE_FP16_DENSE_GFLOPS);
        printf("  Status: %s\n", get_efficiency_status(compute_efficiency));
        printf("\n");

        printf("--- Memory Performance ---\n");
        printf("  Memory Accessed: %.3f GB\n", memory_gb);
        printf("  Actual Bandwidth: %.2f GB/s\n", bandwidth_gb_s);
        printf("  Bandwidth Utilization: %.1f%% (%.2f / %.0f GB/s)\n",
               bandwidth_utilization, bandwidth_gb_s, A100Specs::HBM_BANDWIDTH_GB_S);
        printf("  Status: %s\n", get_bandwidth_status(bandwidth_utilization));
        printf("\n");

        printf("--- Throughput ---\n");
        printf("  Throughput: %.2f K tokens/ms\n", throughput_tokens_ms / 1000.0);
        printf("  Throughput: %.2f M tokens/s\n", throughput_tokens_sec / 1e6);
        printf("\n");

        printf("--- Bottleneck Analysis ---\n");
        analyze_bottleneck();
        printf("========================================\n\n");
    }

    // 打印 CSV 格式（方便导入 Excel）
    void print_csv() const {
        printf("CSV,%d,%d,%d,%d,%.3f,%.2f,%.1f,%.2f,%.1f,%.2f,%.2f\n",
               B, H, N, d, time_ms, gflops, compute_efficiency,
               memory_gb, bandwidth_gb_s, bandwidth_utilization,
               throughput_tokens_ms, throughput_tokens_sec);
    }

private:
    static const char* get_efficiency_status(double efficiency) {
        if (efficiency < 30.0) return "❌ Needs Optimization";
        if (efficiency < 60.0) return "⚠️  Fair";
        if (efficiency < 80.0) return "✅ Good";
        return "🚀 Excellent";
    }

    static const char* get_bandwidth_status(double utilization) {
        if (utilization < 40.0) return "❌ Poor Memory Access";
        if (utilization < 80.0) return "⚠️  Fair";
        return "✅ Good";
    }

    void analyze_bottleneck() const {
        // 判断主要瓶颈
        if (compute_efficiency < 30.0 && bandwidth_utilization < 40.0) {
            printf("  ⚠️  Both compute and memory are low.\n");
            printf("      Possible issues:\n");
            printf("      - Register spilling\n");
            printf("      - Warp divergence\n");
            printf("      - Poor memory coalescing\n");
        } else if (compute_efficiency < 40.0) {
            printf("  🔧 Compute-bound: focus on Tensor Core usage\n");
            printf("      Check: Are you using mma.sync?\n");
        } else if (bandwidth_utilization > 80.0) {
            printf("  💾 Memory-bound: memory bandwidth saturated\n");
            printf("      Good! This is expected for Attention.\n");
        } else {
            printf("  ✅ Balanced performance\n");
        }
    }
};

// ============================================================
// 计算 Flash Attention 性能指标
// ============================================================
inline PerformanceMetrics calculate_flash_attention_metrics(
    int B, int H, int N, int d,
    float time_ms,
    size_t bytes_per_element = sizeof(__half)  // 默认 FP16
) {
    PerformanceMetrics metrics;
    metrics.B = B;
    metrics.H = H;
    metrics.N = N;
    metrics.d = d;
    metrics.bytes_per_element = bytes_per_element;
    metrics.time_ms = time_ms;

    // 1. 计算 FLOPs
    // 标准公式：4*B*H*N²*d
    // 分解：Q*K^T (2乘2加) + softmax (2) + Attn*V (2乘2加) ≈ 4
    metrics.theoretical_flops = 4.0 * B * H * N * N * d;
    metrics.gflops = metrics.theoretical_flops / (time_ms * 1e-3) / 1e9;

    // 2. 计算效率
    metrics.compute_efficiency = (metrics.gflops / A100Specs::TENSOR_CORE_FP16_DENSE_GFLOPS) * 100.0;

    // 3. 计算内存访问
    // 理论最小：Q+K+V (读) + O (写) = 4 * total_elements
    size_t total_elements = B * H * N * d;
    metrics.memory_bytes = 4.0 * total_elements * bytes_per_element;
    metrics.memory_gb = metrics.memory_bytes / (1024.0 * 1024.0 * 1024.0);

    // 4. 计算带宽
    metrics.bandwidth_gb_s = metrics.memory_gb / (time_ms * 1e-3);
    metrics.bandwidth_utilization = (metrics.bandwidth_gb_s / A100Specs::HBM_BANDWIDTH_GB_S) * 100.0;

    // 5. 计算吞吐量
    double total_tokens = double(B) * H * N;
    metrics.throughput_tokens_ms = total_tokens / time_ms;
    metrics.throughput_tokens_sec = total_tokens / (time_ms * 1e-3);

    return metrics;
}

// ============================================================
// 使用示例
// ============================================================
/*
int main() {
    // 示例：B=8, H=16, N=2048, d=128, FP16
    float time_ms = 5.234f;  // 假设你的 kernel 运行时间

    auto metrics = calculate_flash_attention_metrics(
        8, 16, 2048, 128, time_ms, sizeof(__half)
    );

    metrics.print_report();
    metrics.print_csv();

    return 0;
}
*/

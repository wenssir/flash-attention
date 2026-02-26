#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

struct DevicePeakSpecs {
    double tensor_fp16_tflops;
    double memory_bandwidth_gb_s;
    bool has_tensor_fp16_peak;
};

struct PerformanceMetrics {
    int B, H, N, d;
    size_t bytes_per_element;
    float time_ms;

    double theoretical_flops;
    double gflops;
    double compute_efficiency;

    double memory_bytes;
    double memory_gb;
    double bandwidth_gb_s;
    double bandwidth_utilization;

    double throughput_tokens_ms;
    double throughput_tokens_sec;

    double peak_tensor_fp16_tflops;
    double peak_memory_bandwidth_gb_s;
    bool has_tensor_fp16_peak;

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
        if (has_tensor_fp16_peak) {
            printf("  Compute Efficiency: %.1f%% (%.2f / %.0f GFLOPS)\n",
                   compute_efficiency, gflops, peak_tensor_fp16_tflops * 1000);
        } else {
            printf("  Compute Efficiency: N/A (unknown tensor FP16 peak)\n");
        }
        printf("\n");

        printf("--- Memory Performance ---\n");
        printf("  Memory Accessed: %.3f GB\n", memory_gb);
        printf("  Actual Bandwidth: %.2f GB/s\n", bandwidth_gb_s);
        printf("  Bandwidth Utilization: %.1f%% (%.2f / %.0f GB/s)\n",
               bandwidth_utilization, bandwidth_gb_s, peak_memory_bandwidth_gb_s);
        printf("\n");

        printf("--- Throughput ---\n");
        printf("  Throughput: %.2f K tokens/ms\n", throughput_tokens_ms / 1000.0);
        printf("  Throughput: %.2f M tokens/s\n", throughput_tokens_sec / 1e6);
        printf("========================================\n\n");
    }

    void print_csv() const {
        printf("CSV,%d,%d,%d,%d,%.3f,%.2f,%.1f,%.2f,%.2f,%.1f,%.2f,%.2f\n",
               B, H, N, d, time_ms, gflops, compute_efficiency,
               memory_gb, bandwidth_gb_s, bandwidth_utilization,
               throughput_tokens_ms, throughput_tokens_sec);
    }
};

inline DevicePeakSpecs infer_device_peak_specs(const cudaDeviceProp& prop, int device_ordinal = 0) {
    DevicePeakSpecs specs{};
    specs.tensor_fp16_tflops = 0.0;
    specs.has_tensor_fp16_peak = false;

    // Theoretical memory BW in GB/s from memory clock and bus width.
    // Use runtime attributes for CUDA-version compatibility.
    int mem_clock_khz = 0;
    int mem_bus_width_bits = 0;
    cudaError_t clk_ok = cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device_ordinal);
    cudaError_t bus_ok = cudaDeviceGetAttribute(&mem_bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, device_ordinal);
    if (clk_ok == cudaSuccess && bus_ok == cudaSuccess) {
        specs.memory_bandwidth_gb_s =
            2.0 * static_cast<double>(mem_clock_khz) * 1000.0 *
            (static_cast<double>(mem_bus_width_bits) / 8.0) / 1e9;
    } else {
        specs.memory_bandwidth_gb_s = 0.0;
    }

    if (strstr(prop.name, "H100") != nullptr || strstr(prop.name, "H200") != nullptr) {
        specs.tensor_fp16_tflops = 989.0;
        specs.has_tensor_fp16_peak = true;
    } else if (strstr(prop.name, "A100") != nullptr || strstr(prop.name, "A800") != nullptr) {
        specs.tensor_fp16_tflops = 312.0;
        specs.has_tensor_fp16_peak = true;
    } else if (strstr(prop.name, "RTX A4000") != nullptr) {
        specs.tensor_fp16_tflops = 76.7;
        specs.has_tensor_fp16_peak = true;
    } else if (strstr(prop.name, "L40") != nullptr) {
        specs.tensor_fp16_tflops = 181.0;
        specs.has_tensor_fp16_peak = true;
    } else if (strstr(prop.name, "RTX 4090") != nullptr) {
        specs.tensor_fp16_tflops = 165.0;
        specs.has_tensor_fp16_peak = true;
    } else if (prop.major >= 9) {
        specs.tensor_fp16_tflops = 989.0;
        specs.has_tensor_fp16_peak = true;
    } else if (prop.major == 8 && prop.minor == 0) {
        specs.tensor_fp16_tflops = 312.0;
        specs.has_tensor_fp16_peak = true;
    }

    return specs;
}

inline PerformanceMetrics calculate_flash_attention_metrics(
    int B, int H, int N, int d,
    float time_ms,
    const DevicePeakSpecs& peak_specs,
    size_t bytes_per_element = sizeof(__half)
) {
    PerformanceMetrics metrics;
    metrics.B = B;
    metrics.H = H;
    metrics.N = N;
    metrics.d = d;
    metrics.bytes_per_element = bytes_per_element;
    metrics.time_ms = time_ms;
    metrics.peak_tensor_fp16_tflops = peak_specs.tensor_fp16_tflops;
    metrics.peak_memory_bandwidth_gb_s = peak_specs.memory_bandwidth_gb_s;
    metrics.has_tensor_fp16_peak = peak_specs.has_tensor_fp16_peak;

    metrics.theoretical_flops = 4.0 * B * H * N * N * d;
    metrics.gflops = metrics.theoretical_flops / (time_ms * 1e-3) / 1e9;
    if (metrics.has_tensor_fp16_peak && metrics.peak_tensor_fp16_tflops > 0.0) {
        metrics.compute_efficiency =
            (metrics.gflops / (metrics.peak_tensor_fp16_tflops * 1000.0)) * 100.0;
    } else {
        metrics.compute_efficiency = 0.0;
    }

    size_t total_elements = B * H * N * d;
    metrics.memory_bytes = 4.0 * total_elements * bytes_per_element;
    metrics.memory_gb = metrics.memory_bytes / (1024.0 * 1024.0 * 1024.0);
    metrics.bandwidth_gb_s = metrics.memory_gb / (time_ms * 1e-3);
    if (metrics.peak_memory_bandwidth_gb_s > 0.0) {
        metrics.bandwidth_utilization =
            (metrics.bandwidth_gb_s / metrics.peak_memory_bandwidth_gb_s) * 100.0;
    } else {
        metrics.bandwidth_utilization = 0.0;
    }

    double total_tokens = double(B) * H * N;
    metrics.throughput_tokens_ms = total_tokens / time_ms;
    metrics.throughput_tokens_sec = total_tokens / (time_ms * 1e-3);

    return metrics;
}

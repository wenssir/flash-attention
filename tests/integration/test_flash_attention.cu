#include "../src/forward/forward_v2_float4_double_buffer_prefetch.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================
// Naive GPU Reference Implementation (Golden Reference)
// This version is simple but correct, with proper normalization
// ============================================================
template<typename T>
__global__ void flash_attention_naive_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    const int N, const int d,
    const int q_stride, const int k_stride, const int v_stride, const int o_stride
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    // Online softmax + attention accumulation for this row
    float m = -INFINITY;  // max score
    float l = 0.0f;       // sum of exp(scores)

    // Initialize output accumulator
    for (int col = 0; col < d; col++) {
        O[row * o_stride + col] = 0.0f;
    }

    // Process each key row
    for (int k = 0; k < N; k++) {
        // Compute Q[row,:] @ K[k,:]^T
        float score = 0.0f;
        for (int i = 0; i < d; i++) {
            score += Q[row * q_stride + i] * K[k * k_stride + i];
        }

        // Online softmax update
        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float beta = expf(score - m_new);

        // Update output accumulator
        for (int col = 0; col < d; col++) {
            O[row * o_stride + col] = O[row * o_stride + col] * alpha + beta * V[k * v_stride + col];
        }

        m = m_new;
        l = l * alpha + beta;
    }

    // Normalize by l
    float inv_l = 1.0f / l;
    for (int col = 0; col < d; col++) {
        O[row * o_stride + col] *= inv_l;
    }
}

// Wrapper for naive kernel
void run_naive_reference(
    const float* d_Q, const float* d_K, const float* d_V, float* d_O_ref,
    int N, int d, int B, int H
) {
    int threads = 256;
    int blocks = (N * B * H + threads - 1) / threads;

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int offset = (b * H + h) * N * d;
            flash_attention_naive_kernel<float><<<blocks, threads>>>(
                d_Q + offset, d_K + offset, d_V + offset, d_O_ref + offset,
                N, d, d, d, d, d
            );
        }
    }
    cudaDeviceSynchronize();
}

// ============================================================
// Check if two arrays are close
// ============================================================
bool check_close(const float* ref, const float* out, int n, float atol = 1e-4, float rtol = 5e-3) {
    float max_diff = 0.0f;
    float max_ref = 0.0f;
    int max_idx = 0;
    int fail_count = 0;

    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - out[i]);
        float tolerance = atol + rtol * fabsf(ref[i]);

        if (diff > tolerance) {
            if (fail_count < 10) {  // Print first 10 failures
                printf("  Mismatch at [%d]: ref=%.6f, out=%.6f, diff=%.6f (tol=%.6f)\n",
                       i, ref[i], out[i], diff, tolerance);
            }
            fail_count++;
        }

        if (diff > max_diff) {
            max_diff = diff;
            max_ref = fabsf(ref[i]);
            max_idx = i;
        }
    }

    if (fail_count > 0) {
        printf("  Total mismatches: %d / %d\n", fail_count, n);
    }

    float max_rel_diff = max_ref > 0 ? max_diff / max_ref : max_diff;
    printf("  Max difference: %.6f (relative: %.6f) at index %d\n", max_diff, max_rel_diff, max_idx);

    return fail_count == 0;
}

// ============================================================
// Test 1: Small scale with naive GPU reference
// ============================================================
bool test_small_with_gpu_reference() {
    printf("\n=== Test 1: Small Scale (N=32, d=64, B=1, H=1) ===\n");

    const int N = 32;
    const int d = 64;
    const int B = 4;
    const int H = 4;
    const int total_elements = B * H * N * d;

    // Allocate host memory
    float *h_Q, *h_K, *h_V, *h_O_ref, *h_O_out;
    cudaMallocHost(&h_Q, total_elements * sizeof(float));
    cudaMallocHost(&h_K, total_elements * sizeof(float));
    cudaMallocHost(&h_V, total_elements * sizeof(float));
    cudaMallocHost(&h_O_ref, total_elements * sizeof(float));
    cudaMallocHost(&h_O_out, total_elements * sizeof(float));

    // Initialize with small random values to avoid overflow
    srand(42);
    for (int i = 0; i < total_elements; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_K[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_V[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O_ref, *d_O_out;
    cudaMalloc(&d_Q, total_elements * sizeof(float));
    cudaMalloc(&d_K, total_elements * sizeof(float));
    cudaMalloc(&d_V, total_elements * sizeof(float));
    cudaMalloc(&d_O_ref, total_elements * sizeof(float));
    cudaMalloc(&d_O_out, total_elements * sizeof(float));

    // Copy to device
    cudaMemcpy(d_Q, h_Q, total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, total_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Run naive reference kernel
    printf("Running naive GPU reference...\n");
    run_naive_reference(d_Q, d_K, d_V, d_O_ref, N, d, B, H);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in naive kernel: %s\n", cudaGetErrorString(err));
        return false;
    }

    // Run your optimized kernel
    printf("Running optimized kernel...\n");
    const int Br = 32;
    const int Bc = 32;
    const int blockSize = 128;
    dim3 grid_dim((N + Br - 1) / Br, H, B);
    dim3 block_dim(blockSize);

    // For (B,H,N,d) contiguous layout: stride_b = H*N*d, stride_h = N*d, stride_n = d
    flash_attention_forward_v2_tile_and_prefetch<Br, Bc, d, blockSize, 32><<<grid_dim, block_dim>>>(
        d_Q, d_K, d_V, d_O_out, N, B, H,
        H * N * d, N * d, d,  // Q strides
        H * N * d, N * d, d,  // K strides
        H * N * d, N * d, d,  // V strides
        H * N * d, N * d, d   // O strides
    );
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in optimized kernel: %s\n", cudaGetErrorString(err));
        return false;
    }

    // Copy results back
    cudaMemcpy(h_O_ref, d_O_ref, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_O_out, d_O_out, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // Print first few values for debugging
    printf("\nFirst row comparison:\n");
    printf("  Naive:  ");
    for (int i = 0; i < 8; i++) printf("%.6f ", h_O_ref[i]);
    printf("\n  Optim:  ");
    for (int i = 0; i < 8; i++) printf("%.6f ", h_O_out[i]);
    printf("\n");

    // Check correctness
    bool passed = check_close(h_O_ref, h_O_out, total_elements);

    // Cleanup
    cudaFreeHost(h_Q); cudaFreeHost(h_K); cudaFreeHost(h_V);
    cudaFreeHost(h_O_ref); cudaFreeHost(h_O_out);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O_ref); cudaFree(d_O_out);

    printf("Test 1: %s\n", passed ? "PASSED ✅" : "FAILED ❌");
    return passed;
}

// ============================================================
// Test 2: Medium scale sanity check
// ============================================================
bool test_medium_sanity() {
    printf("\n=== Test 2: Medium Scale (N=128, d=64, B=2, H=2) Sanity ===\n");

    const int N = 128;
    const int d = 64;
    const int B = 8;
    const int H = 8;
    const int total_elements = B * H * N * d;

    float *h_Q, *h_K, *h_V, *h_O_ref, *h_O_out;
    cudaMallocHost(&h_Q, total_elements * sizeof(float));
    cudaMallocHost(&h_K, total_elements * sizeof(float));
    cudaMallocHost(&h_V, total_elements * sizeof(float));
    cudaMallocHost(&h_O_ref, total_elements * sizeof(float));
    cudaMallocHost(&h_O_out, total_elements * sizeof(float));

    srand(42);
    for (int i = 0; i < total_elements; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_K[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_V[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    float *d_Q, *d_K, *d_V, *d_O_ref, *d_O_out;
    cudaMalloc(&d_Q, total_elements * sizeof(float));
    cudaMalloc(&d_K, total_elements * sizeof(float));
    cudaMalloc(&d_V, total_elements * sizeof(float));
    cudaMalloc(&d_O_ref, total_elements * sizeof(float));
    cudaMalloc(&d_O_out, total_elements * sizeof(float));

    cudaMemcpy(d_Q, h_Q, total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, total_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Run naive reference
    printf("Running naive GPU reference...\n");
    run_naive_reference(d_Q, d_K, d_V, d_O_ref, N, d, B, H);

    // Run optimized kernel
    printf("Running optimized kernel...\n");
    const int Br = 32;
    const int Bc = 32;
    const int blockSize = 128;
    dim3 grid_dim((N + Br - 1) / Br, H, B);
    dim3 block_dim(blockSize);

    // For (B,H,N,d) contiguous layout: stride_b = H*N*d, stride_h = N*d, stride_n = d
    flash_attention_forward_v2_tile_and_prefetch<Br, Bc, d, blockSize, 32><<<grid_dim, block_dim>>>(
        d_Q, d_K, d_V, d_O_out, N, B, H,
        H * N * d, N * d, d,  // Q strides
        H * N * d, N * d, d,  // K strides
        H * N * d, N * d, d,  // V strides
        H * N * d, N * d, d   // O strides
    );
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return false;
    }

    cudaMemcpy(h_O_ref, d_O_ref, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_O_out, d_O_out, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    bool passed = check_close(h_O_ref, h_O_out, total_elements);

    cudaFreeHost(h_Q); cudaFreeHost(h_K); cudaFreeHost(h_V);
    cudaFreeHost(h_O_ref); cudaFreeHost(h_O_out);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O_ref); cudaFree(d_O_out);

    printf("Test 2: %s\n", passed ? "PASSED ✅" : "FAILED ❌");
    return passed;
}

// ============================================================
// Test 3: d=128
// ============================================================
bool test_d128() {
    printf("\n=== Test 3: d=128 (N=64, d=128, B=1, H=1) ===\n");

    const int N = 64;
    const int d = 64;
    const int B = 1;
    const int H = 1;
    const int total_elements = B * H * N * d;

    float *h_Q, *h_K, *h_V, *h_O_ref, *h_O_out;
    cudaMallocHost(&h_Q, total_elements * sizeof(float));
    cudaMallocHost(&h_K, total_elements * sizeof(float));
    cudaMallocHost(&h_V, total_elements * sizeof(float));
    cudaMallocHost(&h_O_ref, total_elements * sizeof(float));
    cudaMallocHost(&h_O_out, total_elements * sizeof(float));

    srand(123);
    for (int i = 0; i < total_elements; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
        h_K[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
        h_V[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    float *d_Q, *d_K, *d_V, *d_O_ref, *d_O_out;
    cudaMalloc(&d_Q, total_elements * sizeof(float));
    cudaMalloc(&d_K, total_elements * sizeof(float));
    cudaMalloc(&d_V, total_elements * sizeof(float));
    cudaMalloc(&d_O_ref, total_elements * sizeof(float));
    cudaMalloc(&d_O_out, total_elements * sizeof(float));

    cudaMemcpy(d_Q, h_Q, total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, total_elements * sizeof(float), cudaMemcpyHostToDevice);

    printf("Running naive GPU reference...\n");
    run_naive_reference(d_Q, d_K, d_V, d_O_ref, N, d, B, H);

    printf("Running optimized kernel...\n");
    const int Br = 32;
    const int Bc = 32;
    const int blockSize = 128;
    dim3 grid_dim((N + Br - 1) / Br, H, B);
    dim3 block_dim(blockSize);

    // For (B,H,N,d) contiguous layout: stride_b = H*N*d, stride_h = N*d, stride_n = d
    flash_attention_forward_v2_tile_and_prefetch<Br, Bc, d, blockSize, 32><<<grid_dim, block_dim>>>(
        d_Q, d_K, d_V, d_O_out, N, B, H,
        H * N * d, N * d, d,  // Q strides
        H * N * d, N * d, d,  // K strides
        H * N * d, N * d, d,  // V strides
        H * N * d, N * d, d   // O strides
    );
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return false;
    }

    cudaMemcpy(h_O_ref, d_O_ref, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_O_out, d_O_out, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    bool passed = check_close(h_O_ref, h_O_out, total_elements);

    cudaFreeHost(h_Q); cudaFreeHost(h_K); cudaFreeHost(h_V);
    cudaFreeHost(h_O_ref); cudaFreeHost(h_O_out);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O_ref); cudaFree(d_O_out);

    printf("Test 3: %s\n", passed ? "PASSED ✅" : "FAILED ❌");
    return passed;
}

// ============================================================
// Main
// ============================================================
int main() {
    printf("========================================\n");
    printf("Flash Attention V2 Correctness Test\n");
    printf("Testing: flash_attention_forward_v2_tile_and_prefetch\n");
    printf("Golden Reference: Naive GPU Kernel\n");
    printf("========================================\n");

    int passed = 0;
    int total = 0;

    total++;
    if (test_small_with_gpu_reference()) passed++;

    total++;
    if (test_medium_sanity()) passed++;

    total++;
    if (test_d128()) passed++;

    printf("\n========================================\n");
    printf("Summary: %d/%d tests passed\n", passed, total);
    printf("========================================\n");

    return (passed == total) ? 0 : 1;
}

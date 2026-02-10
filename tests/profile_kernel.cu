#include <iostream>
#include <cuda_runtime.h>

#include "../src/forward/forward_v2_float4_double_buffer_prefetch.h"

using IndexType = int;
using ValueType = float;

int main() {
    const int B = 8, H = 8, N = 1024, d = 64;
    const int Br = 16, Bc = 16;

    const int q_stride_b = H * N * d;
    const int q_stride_h = N * d;
    const int q_stride_n = d;

    const int k_stride_b = H * N * d;
    const int k_stride_h = N * d;
    const int k_stride_n = d;

    const int v_stride_b = H * N * d;
    const int v_stride_h = N * d;
    const int v_stride_n = d;

    const int o_stride_b = H * N * d;
    const int o_stride_h = N * d;
    const int o_stride_n = d;

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, B * H * N * d * sizeof(float));
    cudaMalloc(&d_K, B * H * N * d * sizeof(float));
    cudaMalloc(&d_V, B * H * N * d * sizeof(float));
    cudaMalloc(&d_O, B * H * N * d * sizeof(float));

    const int blockSize = 128;

    dim3 grid(N / Br, H, B);
    dim3 block(blockSize); // blockSize

    // 预热
    flash_attention_forward_v2_tile_and_prefetch<Br, Bc, d, blockSize, 32><<<grid, block>>>(d_Q, d_K, d_V, d_O, N, B, H, 
    q_stride_b, q_stride_h, q_stride_n,
    k_stride_b, k_stride_h, k_stride_n,
    v_stride_b, v_stride_h, v_stride_n,
    o_stride_b, o_stride_h, o_stride_n);
    cudaDeviceSynchronize();

    flash_attention_forward_v2_tile_and_prefetch<Br, Bc, d, blockSize, 32><<<grid, block>>>(d_Q, d_K, d_V, d_O, N, B, H, q_stride_b, q_stride_h, q_stride_n,
    k_stride_b, k_stride_h, k_stride_n,
    v_stride_b, v_stride_h, v_stride_n,
    o_stride_b, o_stride_h, o_stride_n);
    cudaDeviceSynchronize();

    return 0;
}
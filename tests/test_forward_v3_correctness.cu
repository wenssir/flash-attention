#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>

#include "../src/config/config.cuh"
#include "../src/forward/forward_v3_tensor_core.cuh"

namespace {

void cpu_reference_online_attention(const float* Q, const float* K, const float* V, float* O,
                                    int B, int H, int N, int D, float scale, bool causal) {
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const float* q_base = Q + (b * H + h) * N * D;
            const float* k_base = K + (b * H + h) * N * D;
            const float* v_base = V + (b * H + h) * N * D;
            float* o_base = O + (b * H + h) * N * D;

            for (int i = 0; i < N; ++i) {
                float m = -INFINITY;
                float l = 0.0f;
                std::vector<float> acc(D, 0.0f);

                for (int j = 0; j < N; ++j) {
                    float s = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        s += q_base[i * D + d] * k_base[j * D + d];
                    }
                    s *= scale;
                    if (causal && j > i) {
                        s = -INFINITY;
                    }

                    float m_new = fmaxf(m, s);
                    float alpha = expf(m - m_new);
                    float p = expf(s - m_new);
                    for (int d = 0; d < D; ++d) {
                        acc[d] = acc[d] * alpha + p * v_base[j * D + d];
                    }
                    l = l * alpha + p;
                    m = m_new;
                }

                float inv_l = 1.0f / l;
                for (int d = 0; d < D; ++d) {
                    o_base[i * D + d] = acc[d] * inv_l;
                }
            }
        }
    }
}

bool run_forward_v3_correctness() {
    using Cfg = config::ForwardV3Config;
    using Elem = typename Cfg::Element;

    constexpr int B = 1;
    constexpr int H = 1;
    constexpr int N = 64;
    constexpr int D = 128;
    constexpr int E = B * H * N * D;
    constexpr float kScale = 1.0f / 8.0f;

    std::vector<float> hQf(E), hKf(E), hVf(E), hO(E, 0.0f), hRef(E, 0.0f);
    std::vector<Elem> hQ(E), hK(E), hV(E);
    srand(7);
    for (int i = 0; i < E; ++i) {
        hQf[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        hKf[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        hVf[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        hQ[i] = __float2half(hQf[i]);
        hK[i] = __float2half(hKf[i]);
        hV[i] = __float2half(hVf[i]);
    }

    cpu_reference_online_attention(hQf.data(), hKf.data(), hVf.data(), hRef.data(), B, H, N, D, kScale, false);

    printf("First few reference values:\n");
    for (int i = 0; i < 10; ++i) {
        printf("  hRef[%d] = %.6f\n", i, hRef[i]);
    }

    Elem *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    if (cudaMalloc(&dQ, E * sizeof(Elem)) != cudaSuccess) return false;
    if (cudaMalloc(&dK, E * sizeof(Elem)) != cudaSuccess) return false;
    if (cudaMalloc(&dV, E * sizeof(Elem)) != cudaSuccess) return false;
    if (cudaMalloc(&dO, E * sizeof(Elem)) != cudaSuccess) return false;

    cudaMemcpy(dQ, hQ.data(), E * sizeof(Elem), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK.data(), E * sizeof(Elem), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV.data(), E * sizeof(Elem), cudaMemcpyHostToDevice);
    cudaMemset(dO, 0, E * sizeof(Elem));

    config::ForwardKernelArgs args{};
    args.Q = dQ;
    args.K = dK;
    args.V = dV;
    args.O = dO;
    args.stride_batch_q = H * N * D;
    args.stride_seq_q = D;
    args.stride_head_q = N * D;
    args.stride_batch_k = H * N * D;
    args.stride_seq_k = D;
    args.stride_head_k = N * D;
    args.stride_batch_v = H * N * D;
    args.stride_seq_v = D;
    args.stride_head_v = N * D;
    args.stride_batch_o = H * N * D;
    args.stride_seq_o = D;
    args.stride_head_o = N * D;
    args.seq_len = N;
    args.heads = H;
    args.head_dim = D;
    args.softmax_scale = kScale;
    args.causal = 0;

    dim3 grid((N + Cfg::BlockM - 1) / Cfg::BlockM, H, B);
    dim3 block(Cfg::NThreads);
    size_t smem_bytes = forward::forward_v3_smem_bytes<Cfg>();

    printf("Launching kernel: grid=(%d,%d,%d), block=(%d)\n", grid.x, grid.y, grid.z, block.x);
    printf("BlockM=%d, BlockN=%d, HeadDim=%d, NThreads=%d\n",
           Cfg::BlockM, Cfg::BlockN, Cfg::HeadDim, Cfg::NThreads);
    printf("Dynamic shared memory bytes=%zu\n", smem_bytes);

    forward::flash_attention_forward_v3_tensor_core<Cfg><<<grid, block, smem_bytes>>>(args);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return false;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA kernel execution error: %s\n", cudaGetErrorString(err));
        return false;
    }

    std::vector<Elem> hO_half(E);
    cudaMemcpy(hO_half.data(), dO, E * sizeof(Elem), cudaMemcpyDeviceToHost);
    for (int i = 0; i < E; ++i) {
        hO[i] = __half2float(hO_half[i]);
    }

    printf("First few GPU values:\n");
    for (int i = 0; i < 10; ++i) {
        printf("  hO[%d] = %.6f (diff = %.6e)\n", i, hO[i], fabsf(hO[i] - hRef[i]));
    }

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);

    float max_diff = 0.0f;
    for (int i = 0; i < E; ++i) {
        max_diff = fmaxf(max_diff, fabsf(hO[i] - hRef[i]));
    }
    printf("forward_v3 max abs diff = %.6e\n", max_diff);
    return max_diff < 2e-3f;
}

} // namespace

int main() {
    bool ok = run_forward_v3_correctness();
    printf("forward_v3 correctness: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

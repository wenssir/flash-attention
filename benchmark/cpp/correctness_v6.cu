#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../src/config/config.cuh"
#include "../../src/forward/forward_v6_mma.cuh"
#include "../../src/utils/util_func.cuh"

namespace {

bool report_cuda_error(cudaError_t err, const char* what) {
    if (err == cudaSuccess) {
        return true;
    }
    printf("%s failed: %s\n", what, cudaGetErrorString(err));
    return false;
}

void cpu_reference_attention(const float* q,
                             const float* k,
                             const float* v,
                             float* o,
                             int B,
                             int H,
                             int N,
                             int D,
                             float scale,
                             bool causal) {
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const float* q_base = q + (b * H + h) * N * D;
            const float* k_base = k + (b * H + h) * N * D;
            const float* v_base = v + (b * H + h) * N * D;
            float* o_base = o + (b * H + h) * N * D;

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

bool check_close(const float* ref,
                 const float* out,
                 int n,
                 float atol,
                 float rtol) {
    float max_diff = 0.0f;
    int max_idx = 0;
    int fail_count = 0;

    for (int i = 0; i < n; ++i) {
        float diff = fabsf(ref[i] - out[i]);
        float tol = atol + rtol * fabsf(ref[i]);
        if (diff > tol) {
            if (fail_count < 10) {
                printf("Mismatch [%d]: ref=%.6f out=%.6f diff=%.6f tol=%.6f\n",
                       i, ref[i], out[i], diff, tol);
            }
            fail_count++;
        }
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }

    printf("Max abs diff = %.6e at index %d\n", max_diff, max_idx);
    if (fail_count > 0) {
        printf("Total mismatches: %d / %d\n", fail_count, n);
    }
    return fail_count == 0;
}

bool run_correctness() {
    using Cfg = config::ForwardConfig;
    using Elem = typename Cfg::Element;

    constexpr int B = 1;
    constexpr int H = 1;
    constexpr int N = 64;
    constexpr int D = 128;
    constexpr int E = B * H * N * D;

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> hQf(E), hKf(E), hVf(E), hRef(E, 0.0f), hOut(E, 0.0f);
    std::vector<Elem> hQ(E), hK(E), hV(E), hO(E);

    srand(17);
    for (int i = 0; i < E; ++i) {
        hQf[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        hKf[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        hVf[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        hQ[i] = __float2half(hQf[i]);
        hK[i] = __float2half(hKf[i]);
        hV[i] = __float2half(hVf[i]);
    }

    cpu_reference_attention(hQf.data(), hKf.data(), hVf.data(), hRef.data(), B, H, N, D, scale, false);

    Elem *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    if (!report_cuda_error(cudaMalloc(&dQ, E * sizeof(Elem)), "cudaMalloc(dQ)")) return false;
    if (!report_cuda_error(cudaMalloc(&dK, E * sizeof(Elem)), "cudaMalloc(dK)")) return false;
    if (!report_cuda_error(cudaMalloc(&dV, E * sizeof(Elem)), "cudaMalloc(dV)")) return false;
    if (!report_cuda_error(cudaMalloc(&dO, E * sizeof(Elem)), "cudaMalloc(dO)")) return false;

    if (!report_cuda_error(cudaMemcpy(dQ, hQ.data(), E * sizeof(Elem), cudaMemcpyHostToDevice), "cudaMemcpy(Q)")) return false;
    if (!report_cuda_error(cudaMemcpy(dK, hK.data(), E * sizeof(Elem), cudaMemcpyHostToDevice), "cudaMemcpy(K)")) return false;
    if (!report_cuda_error(cudaMemcpy(dV, hV.data(), E * sizeof(Elem), cudaMemcpyHostToDevice), "cudaMemcpy(V)")) return false;
    if (!report_cuda_error(cudaMemset(dO, 0, E * sizeof(Elem)), "cudaMemset(O)")) return false;

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
    args.softmax_scale = scale;
    args.causal = 0;

    dim3 grid((N + Cfg::BlockM - 1) / Cfg::BlockM, H, B);
    dim3 block(Cfg::NThreads);
    size_t smem = static_cast<size_t>(forward::forward_smem_bytes<Cfg>());

    auto set_attr_err = cudaFuncSetAttribute(
        forward::flash_attention_forward_v6_mma<Cfg>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem));
    if (set_attr_err != cudaSuccess) {
        printf("cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(set_attr_err));
        return false;
    }
    set_attr_err = cudaFuncSetAttribute(
        forward::flash_attention_forward_v6_mma<Cfg>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    if (set_attr_err != cudaSuccess) {
        printf("cudaFuncSetAttribute(carveout) failed: %s\n", cudaGetErrorString(set_attr_err));
        return false;
    }

    forward::flash_attention_forward_v6_mma<Cfg><<<grid, block, smem>>>(args);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel launch failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("kernel execution failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    if (!report_cuda_error(cudaMemcpy(hO.data(), dO, E * sizeof(Elem), cudaMemcpyDeviceToHost), "cudaMemcpy(O)")) return false;
    for (int i = 0; i < E; ++i) {
        hOut[i] = __half2float(hO[i]);
    }

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);

    printf("First 8 reference: ");
    for (int i = 0; i < 8; ++i) printf("%.6f ", hRef[i]);
    printf("\nFirst 8 output:    ");
    for (int i = 0; i < 8; ++i) printf("%.6f ", hOut[i]);
    printf("\n");

    return check_close(hRef.data(), hOut.data(), E, 2e-2f, 8e-2f);
}

} // namespace

int main() {
    printf("v6 correctness test (non-causal, v6 assumptions)\n");
    bool ok = run_correctness();
    printf("v6 correctness: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

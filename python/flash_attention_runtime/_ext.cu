#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cmath>
#include <optional>
#include <stdexcept>
#include <tuple>

#include "../../src/config/config.cuh"
#include "../../src/forward/forward_v2_float4_double_buffer_prefetch.h"
#include "../../src/forward/forward_v3_tensor_core.cuh"

namespace py = pybind11;

namespace {

constexpr int kBlockSize = 128;
constexpr int kWarpSize = 32;

template <int HeadDim>
struct KernelTile;

template <>
struct KernelTile<64> {
    static constexpr int BlockM = 32;
    static constexpr int BlockN = 32;
};

template <>
struct KernelTile<128> {
    static constexpr int BlockM = 16;
    static constexpr int BlockN = 16;
};

inline void check_cuda_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(
        t.dtype() == torch::kFloat32 || t.dtype() == torch::kFloat16,
        name,
        " must be float32 or float16"
    );
    TORCH_CHECK(t.dim() == 4, name, " must be a 4D tensor with shape [B, H, N, D]");
}

inline void validate_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const std::optional<torch::Tensor>& o_opt
) {
    check_cuda_tensor(q, "q");
    check_cuda_tensor(k, "k");
    check_cuda_tensor(v, "v");

    TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have the same shape");
    TORCH_CHECK(q.sizes() == v.sizes(), "q and v must have the same shape");
    TORCH_CHECK(q.dtype() == k.dtype(), "q and k must have the same dtype");
    TORCH_CHECK(q.dtype() == v.dtype(), "q and v must have the same dtype");

    const auto d = q.size(3);
    TORCH_CHECK(d == 64 || d == 128, "Only head_dim 64 or 128 is supported");

    const auto n = q.size(2);
    if (q.dtype() == torch::kFloat16) {
        TORCH_CHECK(d == 128, "fp16 path currently supports d_head=128 only");
        TORCH_CHECK((n % config::ForwardV3Config::BlockM) == 0, "fp16 path requires seq_len divisible by 64");
    } else {
        if (d == 64) {
            TORCH_CHECK((n % KernelTile<64>::BlockM) == 0, "seq_len must be divisible by 32 for d=64");
        } else if (d == 128) {
            TORCH_CHECK((n % KernelTile<128>::BlockM) == 0, "seq_len must be divisible by 16 for d=128");
        }
    }

    if (o_opt.has_value()) {
        check_cuda_tensor(o_opt.value(), "o");
        TORCH_CHECK(o_opt.value().sizes() == q.sizes(), "o must have the same shape as q");
        TORCH_CHECK(o_opt.value().dtype() == q.dtype(), "o must have the same dtype as q");
    }
}

template <int HeadDim>
void launch_kernel(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    torch::Tensor& o
) {
    const int B = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int N = static_cast<int>(q.size(2));

    const int q_stride_b = static_cast<int>(q.stride(0));
    const int q_stride_h = static_cast<int>(q.stride(1));
    const int q_stride_n = static_cast<int>(q.stride(2));
    const int k_stride_b = static_cast<int>(k.stride(0));
    const int k_stride_h = static_cast<int>(k.stride(1));
    const int k_stride_n = static_cast<int>(k.stride(2));
    const int v_stride_b = static_cast<int>(v.stride(0));
    const int v_stride_h = static_cast<int>(v.stride(1));
    const int v_stride_n = static_cast<int>(v.stride(2));
    const int o_stride_b = static_cast<int>(o.stride(0));
    const int o_stride_h = static_cast<int>(o.stride(1));
    const int o_stride_n = static_cast<int>(o.stride(2));

    constexpr int kBlockM = KernelTile<HeadDim>::BlockM;
    constexpr int kBlockN = KernelTile<HeadDim>::BlockN;
    dim3 grid_dim((N + kBlockM - 1) / kBlockM, H, B);
    dim3 block_dim(kBlockSize);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    flash_attention_forward_v2_tile_and_prefetch<kBlockM, kBlockN, HeadDim, kBlockSize, kWarpSize>
        <<<grid_dim, block_dim, 0, stream>>>(
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            o.data_ptr<float>(),
            N, B, H,
            q_stride_b, q_stride_h, q_stride_n,
            k_stride_b, k_stride_h, k_stride_n,
            v_stride_b, v_stride_h, v_stride_n,
            o_stride_b, o_stride_h, o_stride_n
        );
}

void launch_v3_fp16(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    torch::Tensor& o
) {
    using Config = config::ForwardV3Config;

    TORCH_CHECK(q.dtype() == torch::kFloat16, "v3 fp16 path expects fp16 input");
    TORCH_CHECK(static_cast<int>(q.size(3)) == Config::HeadDim, "v3 fp16 path expects d_head=128");
    TORCH_CHECK((q.size(2) % Config::BlockM) == 0, "v3 fp16 path expects seq_len divisible by 64");

    config::ForwardKernelArgs args{};
    args.Q = q.data_ptr();
    args.K = k.data_ptr();
    args.V = v.data_ptr();
    args.O = o.data_ptr();

    args.stride_batch_q = static_cast<int>(q.stride(0));
    args.stride_seq_q = static_cast<int>(q.stride(2));
    args.stride_head_q = static_cast<int>(q.stride(1));
    args.stride_batch_k = static_cast<int>(k.stride(0));
    args.stride_seq_k = static_cast<int>(k.stride(2));
    args.stride_head_k = static_cast<int>(k.stride(1));
    args.stride_batch_v = static_cast<int>(v.stride(0));
    args.stride_seq_v = static_cast<int>(v.stride(2));
    args.stride_head_v = static_cast<int>(v.stride(1));
    args.stride_batch_o = static_cast<int>(o.stride(0));
    args.stride_seq_o = static_cast<int>(o.stride(2));
    args.stride_head_o = static_cast<int>(o.stride(1));

    args.seq_len = static_cast<int>(q.size(2));
    args.heads = static_cast<int>(q.size(1));
    args.head_dim = static_cast<int>(q.size(3));
    args.softmax_scale = 1.0f / sqrtf(static_cast<float>(args.head_dim));
    args.causal = 0;

    dim3 grid(
        (args.seq_len + Config::BlockM - 1) / Config::BlockM,
        static_cast<unsigned int>(q.size(1)),
        static_cast<unsigned int>(q.size(0))
    );
    dim3 block(Config::NThreads);
    size_t smem = static_cast<size_t>(forward::forward_v3_smem_bytes<Config>());

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    forward::flash_attention_forward_v3_tensor_core<Config><<<grid, block, smem, stream>>>(args);
}

std::tuple<torch::Tensor, float> fa_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    std::optional<torch::Tensor> o_opt,
    bool benchmark
) {
    validate_inputs(q, k, v, o_opt);

    auto in_dtype = q.dtype();
    torch::Tensor o = o_opt.has_value() ? o_opt.value() : torch::empty_like(q);
    float runtime_ms = 0.0f;

    cudaEvent_t start{};
    cudaEvent_t stop{};
    if (benchmark) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, at::cuda::getCurrentCUDAStream().stream());
    }

    if (in_dtype == torch::kFloat16) {
        launch_v3_fp16(q, k, v, o);
    } else {
        const int d = static_cast<int>(q.size(3));
        if (d == 64) {
            launch_kernel<64>(q, k, v, o);
        } else if (d == 128) {
            launch_kernel<128>(q, k, v, o);
        } else {
            TORCH_CHECK(false, "Unsupported head_dim: ", d);
        }
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    if (benchmark) {
        cudaEventRecord(stop, at::cuda::getCurrentCUDAStream().stream());
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runtime_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return {o, runtime_ms};
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &fa_forward,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("o") = std::optional<torch::Tensor>{},
        py::arg("benchmark") = false,
        "FlashAttentionV2 forward (CUDA)"
    );
}

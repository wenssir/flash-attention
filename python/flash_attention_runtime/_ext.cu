#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cmath>
#include <optional>
#include <stdexcept>
#include <tuple>

#include "../../src/config/config.cuh"
#include "../../src/utils/util_func.cuh"
#include "../../src/forward/forward_v6_mma.cuh"

namespace py = pybind11;

namespace {

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
    TORCH_CHECK(d == config::ForwardConfig::HeadDim, "Only head_dim 128 is supported");

    const auto n = q.size(2);
    TORCH_CHECK(q.dtype() == torch::kFloat16, "pybind path currently supports fp16 only");
    TORCH_CHECK((n % config::ForwardConfig::BlockM) == 0, "fp16 path requires seq_len divisible by BlockM");

    if (o_opt.has_value()) {
        check_cuda_tensor(o_opt.value(), "o");
        TORCH_CHECK(o_opt.value().sizes() == q.sizes(), "o must have the same shape as q");
        TORCH_CHECK(o_opt.value().dtype() == q.dtype(), "o must have the same dtype as q");
    }
}

void launch_fp16(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    torch::Tensor& o
) {
    using Config = config::ForwardConfig;

    TORCH_CHECK(q.dtype() == torch::kFloat16, "fp16 path expects fp16 input");
    TORCH_CHECK(static_cast<int>(q.size(3)) == Config::HeadDim, "fp16 path expects fixed d_head");
    TORCH_CHECK((q.size(2) % Config::BlockM) == 0, "fp16 path expects seq_len divisible by BlockM");

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
    size_t smem = static_cast<size_t>(forward::forward_smem_bytes<Config>());

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto set_attr_err = cudaFuncSetAttribute(
        forward::flash_attention_forward_v6_mma<Config>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem));
    TORCH_CHECK(
        set_attr_err == cudaSuccess,
        "Failed to set MaxDynamicSharedMemorySize for kernel: ",
        cudaGetErrorString(set_attr_err));
    auto carveout_err = cudaFuncSetAttribute(
        forward::flash_attention_forward_v6_mma<Config>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    TORCH_CHECK(
        carveout_err == cudaSuccess,
        "Failed to set PreferredSharedMemoryCarveout for kernel: ",
        cudaGetErrorString(carveout_err));
    forward::flash_attention_forward_v6_mma<Config><<<grid, block, smem, stream>>>(args);
}

std::tuple<torch::Tensor, float> fa_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    std::optional<torch::Tensor> o_opt,
    bool benchmark
) {
    validate_inputs(q, k, v, o_opt);

    torch::Tensor o = o_opt.has_value() ? o_opt.value() : torch::empty_like(q);
    float runtime_ms = 0.0f;

    cudaEvent_t start{};
    cudaEvent_t stop{};
    if (benchmark) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, at::cuda::getCurrentCUDAStream().stream());
    }

    launch_fp16(q, k, v, o);

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

import torch
from torch.utils.cpp_extension import load_inline

# ==========================================
# 1. 你的 CUDA Kernel 代码
# ==========================================
# 请把你修正后的完整 kernel 代码粘贴在下面的字符串里
cuda_source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

using ValueType = float;
using IndexType = int;

// 【请在这里粘贴你的 flash_attention_v2_naive 函数】
// 务必确保写回逻辑在 j 循环外面！
// 为了方便，我这里假设你已经把上面修正后的代码贴进来了
// 模板参数建议设为默认值或者在 launcher 里指定
// ...

template<const IndexType Br, const IndexType Bc, const IndexType d, const IndexType blockSize, const IndexType warpSize>
__global__ void flash_attention_v2_naive(
    const ValueType* __restrict__ Q, 
    const ValueType* __restrict__ K,
    const ValueType* __restrict__ V,
    ValueType* __restrict__ O,
    const IndexType N,
    const IndexType Batch,
    const IndexType Head,
    const IndexType q_stride_b, const IndexType q_stride_h, const IndexType q_stride_n,
    const IndexType k_stride_b, const IndexType k_stride_h, const IndexType k_stride_n,
    const IndexType v_stride_b, const IndexType v_stride_h, const IndexType v_stride_n,
    const IndexType o_stride_b, const IndexType o_stride_h, const IndexType o_stride_n
) {
    IndexType bid = blockIdx.z;
    IndexType hid = blockIdx.y;
    IndexType tid = blockIdx.x; // tile ID

    const IndexType q_tile_stride = Br * q_stride_n;

    IndexType q_offset = q_stride_b * bid + q_stride_h * hid + q_tile_stride * tid;
    IndexType k_offset = k_stride_b * bid + k_stride_h * hid;
    IndexType v_offset = v_stride_b * bid + v_stride_h * hid;
    IndexType o_offset = o_stride_b * bid + o_stride_h * hid;

    const ValueType* q_ptr = Q + q_offset;
    const ValueType* k_ptr = K + k_offset;
    const ValueType* v_ptr = V + v_offset;

    ValueType* o_ptr = O + o_offset + Br * q_stride_n * tid;

    __shared__ ValueType qmem[Br * d];
    __shared__ ValueType kmem[Bc * d];
    __shared__ ValueType vmem[Bc * d];

    IndexType per_thread_row_count = (d + blockSize - 1) / blockSize;

    const IndexType warp_count = blockSize / warpSize;
    constexpr int rows_per_warp = (Br + warp_count - 1) / warp_count;

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    constexpr int per_thread_row_count_warp = (d + warpSize - 1) / warpSize;

    float m[rows_per_warp];
    float l[rows_per_warp];

    float acc_o[rows_per_warp][per_thread_row_count_warp];

    for (int i = 0; i < rows_per_warp; ++i) {
        m[i] = -INFINITY; // 初始化为负无穷
        l[i] = 0.0f;
        for (int j = 0; j < per_thread_row_count_warp; ++j) {
            acc_o[i][j] = 0.0f;
        }
    }

    for (int i = 0; i < Br; ++i) {
        const ValueType* tmp_ptr = q_ptr + i * q_stride_n + threadIdx.x * per_thread_row_count;
        for (int j = 0; j < per_thread_row_count; ++j) {
            qmem[i * d + threadIdx.x * per_thread_row_count + j] = *(tmp_ptr + j);
        }
    }
    __syncthreads();

    for (int j = 0; j < N; j += Bc) {
        const ValueType* k_tile_ptr = k_ptr + k_stride_n * j;
        const ValueType* v_tile_ptr = v_ptr + v_stride_n * j;

        for (int i = 0; i < Bc; ++i) {
            const IndexType row_offset_k = i * k_stride_n + threadIdx.x * per_thread_row_count;
            const IndexType row_offset_v = i * v_stride_n + threadIdx.x * per_thread_row_count;
            for (int index = 0; index < per_thread_row_count; ++index) {
                if (threadIdx.x * per_thread_row_count + index < d) {
                    kmem[i * d + threadIdx.x * per_thread_row_count + index] = *(k_tile_ptr + row_offset_k + index);
                    vmem[i * d + threadIdx.x * per_thread_row_count + index] = *(v_tile_ptr + row_offset_v + index);
                }
            }
        }
        __syncthreads();

        for (int i = 0; i < Bc; ++i) {
            for (int j = 0; j <  rows_per_warp; ++j) { // j 代表第几行
                int t = warp_id * rows_per_warp + j;
                ValueType Sij = 0;
                if (t < Br) {
                    for (int index = lane_id * per_thread_row_count_warp; index < (lane_id + 1) * per_thread_row_count_warp; ++index) {
                        if (index < d) {
                            Sij += qmem[t * d + index] * kmem[i * d + index];
                        }
                    }

                    Sij += __shfl_xor_sync(0xffffffff, Sij, 16); // 0-15 和 16-31 交换相加
                    Sij += __shfl_xor_sync(0xffffffff, Sij, 8);  // 0-7 和 8-15 ...
                    Sij += __shfl_xor_sync(0xffffffff, Sij, 4);
                    Sij += __shfl_xor_sync(0xffffffff, Sij, 2);
                    Sij += __shfl_xor_sync(0xffffffff, Sij, 1);
                    
                    ValueType m_prev = m[j];
                    ValueType l_prev = l[j];
                    
                    ValueType m_new = max(m_prev, Sij);

                    ValueType p_val = expf(Sij - m_new);
                    ValueType alpha = expf(m_prev - m_new);

                    l[j] = l_prev * alpha + p_val;

                    m[j] = m_new;

                    for (int index = 0; index < per_thread_row_count_warp; ++index) {
                        int col_idx = lane_id * per_thread_row_count_warp + index;
                        if (col_idx < d) {
                            float v_val = vmem[i * d + col_idx];
                            acc_o[j][index] = acc_o[j][index] * alpha + p_val * v_val;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < rows_per_warp; ++i) {
        int t = warp_id * rows_per_warp + i;
        if (t < Br) {
            const int row_offset = t * o_stride_n; // 相对于 o_ptr 的行偏移
            for (int j = 0; j < per_thread_row_count_warp; ++j) {
                const int offset = lane_id * per_thread_row_count_warp + j;
                if (offset < d) {
                    // 最终结果 = 累加值 / 分母 l
                    *(o_ptr + row_offset + offset) = acc_o[i][j] / l[i];
                }
            }
        }
    }
}

// 2. C++ Host 端封装 (Launcher)
// 这就是 PyTorch 调用的入口
torch::Tensor flash_attn_forward(
    torch::Tensor Q, 
    torch::Tensor K, 
    torch::Tensor V
) {
    // 确保输入是 CUDA Tensor
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");

    // 获取维度
    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);

    // 分配输出 Tensor
    auto O = torch::empty_like(Q);

    // 设置 Kernel 参数
    const int Br = 64;
    const int Bc = 64;
    const int blockSize = 128;

    // 计算 Grid
    dim3 grid_dim(N / Br, H, B); // x: Tile ID, y: Head ID, z: Batch ID
    dim3 block_dim(blockSize);

    // 启动 Kernel (实例化模板)
    // <Br, Bc, Tc, Tr, blockSize, warpSize>
    // 注意：Tc 和 Tr 在你的代码里似乎没用到，这里随便传个值或者修正你的模板
    flash_attention_v2_naive<64, 64, 64, 128, 32><<<grid_dim, block_dim>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        N, B, H,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2)
    );
    
    // 检查是否有 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    return O;
}
'''

cpp_source = "torch::Tensor flash_attn_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);"

# ==========================================
# 3. 编译并加载
# ==========================================
print("正在编译 CUDA Kernel...")
flash_attn = load_inline(
    name='flash_attn_v2_test',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['flash_attn_forward'],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)
print("编译成功！")

# ==========================================
# 4. 运行验证
# ==========================================
def test_correctness():
    # 设置参数
    B, H, N, d = 2, 4, 1024, 64  # 小一点的规模方便调试
    torch.manual_seed(42)

    # 准备数据 (FP32)
    Q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    K = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    V = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)

    # 1. 跑 PyTorch 标准 Attention (Ground Truth)
    # scale_factor 默认为 1/sqrt(d)，这里为了简单我们可以不用 scale 或者手动乘
    # F.scaled_dot_product_attention 默认会有 scale
    # 你的 Kernel 里没有写 scale，所以我们对比时要把 ref 的 scale 去掉
    # 或者给你的 Kernel 加上 scale。
    # 这里我们对比 "无 scale" 版本：
    
    # 手动实现一个简单的 Ref (因为 F.sdpa 强制带 scale)
    print("运行 PyTorch Reference...")
    scores = torch.matmul(Q, K.transpose(-2, -1))
    attn = torch.softmax(scores, dim=-1)
    ref_out = torch.matmul(attn, V)
    print("ref_out = %d\n", ref_out)
    # 2. 跑你的 Kernel
    print("运行 Custom Kernel...")
    my_out = flash_attn.flash_attn_forward(Q, K, V)
    print("my_out = %d\n", my_out)
    # 3. 对比结果
    # 允许一定的误差 (FP32 也就 1e-4 左右)
    is_close = torch.allclose(ref_out, my_out, atol=1e-3, rtol=1e-3)
    max_diff = (ref_out - my_out).abs().max().item()

    print(f"\n测试结果: {'✅ 通过' if is_close else '❌ 失败'}")
    print(f"最大误差: {max_diff:.6f}")
    
    if not is_close:
        print("Reference (前3个值):", ref_out[0,0,0,:3])
        print("My Kernel (前3个值):", my_out[0,0,0,:3])

if __name__ == "__main__":
    test_correctness()
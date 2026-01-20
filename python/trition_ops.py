import torch
import triton
import triton.language as tl

# ==========================================
# 核心 Kernel：运行在 GPU 上的代码
# ==========================================
@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, sm_scale,  # 显存指针 (HBM)
    L, Out,             # 输出指针
    stride_qz, stride_qh, stride_qm, stride_qk, # Q 的 stride
    stride_kz, stride_kh, stride_kn, stride_kk, # K 的 stride
    stride_vz, stride_vh, stride_vn, stride_vk, # V 的 stride
    stride_oz, stride_oh, stride_om, stride_on, # Out 的 stride
    Z, H, N_CTX,        # 维度信息: Batch, Head, SeqLen
    BLOCK_M: tl.constexpr, # Q 的分块大小 (编译期常量)
    BLOCK_N: tl.constexpr, # K, V 的分块大小
    HEAD_DIM: tl.constexpr, # Head Dimension
):
    # 1. 确定当前 Program (Block) 的坐标
    # FlashAttn V2 并行策略：Grid = (Q的分块数, Batch * Head)
    start_m = tl.program_id(0) # 当前负责 Q 的哪一段 (0, 1, 2...)
    off_hz = tl.program_id(1)  # 当前负责哪个 batch 和 head
    
    # 2. 计算 Q, K, V 在 HBM 上的基础偏移量
    # 这一步是为了定位到当前 batch 和 head 的数据起始位置
    q_offset = off_hz * stride_qh 
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh
    
    # ==========================================
    # 指针魔法：make_block_ptr (Triton 最强功能)
    # 它创建了一个指向 HBM 的"2D 窗口"，自动处理边界和 stride
    # ==========================================
    
    # Q 指针：形状 [BLOCK_M, HEAD_DIM]
    # start_m * BLOCK_M 是当前块在序列维度(M)的起始位置
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    
    # K 指针：形状 [HEAD_DIM, BLOCK_N] (注意这里要转置，方便做 dot)
    # 实际上我们加载 [BLOCK_N, HEAD_DIM]，然后在 SRAM 里转置
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0), # K 从 0 开始遍历
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1)
    )

    # V 指针：形状 [BLOCK_N, HEAD_DIM]
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0), # V 从 0 开始遍历
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )

    # ==========================================
    # 核心循环：加载 Q -> 遍历 K, V
    # ==========================================
    
    # 1. 加载 Q 到 SRAM (Load once)
    # boundary_check 确保不越界，padding_option 补零
    q = tl.load(Q_block_ptr) 

    # 2. 初始化累加器 (Accumulators)
    # m_i: 记录当前行 max 值，用于数值稳定性 (防止 exp 溢出)
    # l_i: 记录分母 (Sum of exp)
    # acc: 记录分子 (Weighted Sum of V)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. 循环 K, V (Loop over N blocks)
    # range(0, N_CTX, BLOCK_N) 意味着每次处理 BLOCK_N 个 Key/Value
    for start_n in range(0, N_CTX, BLOCK_N):
        # 3.1 加载 K, V 到 SRAM
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

        # 3.2 计算 Attention Score: S = Q * K^T
        # tl.dot 会自动调用 Tensor Cores (HMMA)
        qk = tl.dot(q, k) 
        qk *= sm_scale # 缩放

        # 3.3 Online Softmax 核心逻辑
        # 找到当前块的最大值
        m_ij = tl.max(qk, 1) 
        # 更新全局最大值 (与之前的 m_i 比较)
        m_new = tl.maximum(m_i, m_ij)
        
        # 计算缩放系数
        # alpha: 旧的 acc 需要缩小多少 (因为 max 变大了)
        # beta:  当前的 qk 需要缩小多少
        # alpha = tl.exp(m_i - m_new)
        # beta = tl.exp(m_ij - m_new)
        
        # 3.4 更新 Output 累加器
        # P_ij = exp(S - m_new)
        # 这里的 None 相当于 unsqueeze，用于广播维度
        p = tl.exp(qk - m_new[:, None]) 
        
        # acc = acc * alpha + P * V
        acc *= tl.exp(m_i - m_new)[:, None] # 缩放旧值
        acc = tl.dot(p.to(v.dtype), v, acc)    # 累加新值 (FMA)

        # 3.5 更新分母 L
        l_i *= tl.exp(m_i - m_new)
        l_i = tl.sum(p, 1)
        
        # 3.6 更新 m_i
        m_i = m_new

        # 3.7 指针步进：移动到下一个 K/V 块
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # ==========================================
    # 收尾：归一化并写回 HBM
    # ==========================================
    
    # O = acc / L
    acc = acc / l_i[:, None]

    # 保存 L (Backward 需要用到)
    # L 的 shape 是 [Z, H, N_CTX]，我们需要根据 start_m 计算偏移
    l_ptrs = L + off_hz * N_CTX + start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(l_ptrs, m_i + tl.log(l_i)) # 存 LogSumExp，数值更稳

    # 保存 Output
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty))

# ==========================================
# Python Wrapper：数据准备与调用
# ==========================================
def flash_attention_v2(q, k, v, causal=False):
    # 形状检查 [Batch, Head, SeqLen, HeadDim]
    # 注意：Triton 里的 N_CTX 对应 SeqLen
    batch, n_heads, seqlen, head_dim = q.shape
    
    # 只有 V2 这里的 BLOCK_M 可以设得比较大 (64 或 128)
    # BLOCK_N 可以小一点，减少 SRAM 压力
    BLOCK_M = 32
    BLOCK_N = 64
    
    # 分配输出内存
    o = torch.empty_like(q)
    L = torch.empty((batch, n_heads, seqlen), device=q.device, dtype=torch.float32)
    
    sm_scale = 1.0 / (head_dim ** 0.5)

    # 启动 Grid
    # x 轴：并行化 SeqLen (这是 V2 的特征，V1 这里是 1)
    # y 轴：并行化 Batch * Heads
    grid = (triton.cdiv(seqlen, BLOCK_M), batch * n_heads)
    
    # 启动 Kernel
    kernel = _flash_attn_fwd_kernel[grid](
        q, k, v, sm_scale,
        L, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch, n_heads, seqlen,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=head_dim,
        num_stages=2, num_warps=16
    )

    # print(kernel.asm["ptx"])
    return o


# ==========================================
# 验证脚本 (Verify correctness)
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(0)
    # 准备数据 (FP16)
    BATCH, HEADS, SEQLEN, HEADDIM = 32, 16, 8192, 128
    q = torch.randn((BATCH, HEADS, SEQLEN, HEADDIM), device="cuda", dtype=torch.float16)
    k = torch.randn((BATCH, HEADS, SEQLEN, HEADDIM), device="cuda", dtype=torch.float16)
    v = torch.randn((BATCH, HEADS, SEQLEN, HEADDIM), device="cuda", dtype=torch.float16)
    
    # 1. 跑 Triton 实现
    print("Warming up...")
    triton_out = flash_attention_v2(q, k, v)
    
    print("Profiling...")
    flash_attention_v2(q, k, v)
    # 2. 跑 PyTorch 标准实现 (Golden)
    # PyTorch 的 scaled_dot_product_attention 已经集成了 FlashAttn
    # torch_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    
    # 3. 对比误差
    # print(f"Max Diff: {(triton_out - torch_out).abs().max().item()}")
    # if torch.allclose(triton_out, torch_out, atol=1e-2, rtol=1e-2):
    #     print("✅ Triton implementation matches PyTorch!")
    # else:
    #     print("❌ Mismatch!")
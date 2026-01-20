import numpy as np

# Q, K, V ∈ N × d， O = Softmax(QK^T)V, Br : tiling size of Q, Bc: tiling size of K, V, scale = 1 / √d
def flash_attention_naive(Q, K, V, Br, Bc):
    N, d = Q.shape
    O = np.zeros((N, d))
    scale = 1.0 / np.sqrt(d)
    for i in range(0, N, Br):
        m_prev = np.full((Br, 1), -np.inf) # temp max molecule
        l_prev = np.full((Br, 1), 0.0) #temp Denominator
        o = np.full((Br, d), 0.0) # temp output not Normalized
        q = Q[i : i + Br, :]
        for j in range(0, N, Bc):
            k = K[j : j + Bc, :].T
            v = V[j : j + Bc, :]
            s = scale * np.matmul(q, k) # (Br, Bc)
            m = np.max(s, axis=1, keepdims=True) # (Br, 1)
            p = np.exp(s - m)                    # (Br, Bc)
            l = np.sum(p, axis=1, keepdims=True) # (Br, 1)
            m_new = np.maximum(m_prev, m)
            a = np.exp(m_prev - m_new)
            b = np.exp(m - m_new)
            l = l_prev * a + l * b
            o = o * a + (p * b) @ v
            m_prev = m_new
            l_prev = l
        O[i : i + Br, :] = o / l_prev
    return O


def verify():
    N, d = 64, 32
    Br, Bc = 16, 16
    Q = np.random.randn(N, d)
    K = np.random.randn(N, d)
    V = np.random.randn(N, d)

    # Reference (Naive Attention)
    S = (Q @ K.T) * (1.0 / np.sqrt(d))
    # 为数值稳定，手动减去全局最大值
    S_max = np.max(S, axis=-1, keepdims=True)
    P = np.exp(S - S_max)
    P /= np.sum(P, axis=-1, keepdims=True)
    O_ref = P @ V

    O_flash = flash_attention_naive(Q, K, V, Br, Bc)

    diff = np.max(np.abs(O_ref - O_flash))
    print(f"Max Difference: {diff}")
    if diff < 1e-6:
        print("✅ 逻辑完美！结果与标准 Attention 完全一致。")
    else:
        print("❌ 结果有差异，请检查 alpha/beta 修正逻辑。")

verify()

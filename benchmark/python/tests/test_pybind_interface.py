import torch

import flash_attention_runtime as fa2


def main():
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    B, H, N, D = 2, 4, 128, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32).contiguous()
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32).contiguous()
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32).contiguous()

    out, ms = fa2.forward_timed(q, k, v)
    print("output shape:", tuple(out.shape))
    print("runtime ms:", ms)


if __name__ == "__main__":
    main()

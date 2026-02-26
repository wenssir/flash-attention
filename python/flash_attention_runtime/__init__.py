from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXT = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT

    this_dir = Path(__file__).resolve().parent
    src = this_dir / "_ext.cu"
    repo_root = this_dir.parent.parent

    _EXT = load(
        name="flash_attention_v2_kernels",
        sources=[str(src)],
        extra_include_paths=[str(repo_root)],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-std=c++20"],
        verbose=False,
    )
    return _EXT


def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor | None = None):
    return _load_ext().forward(q, k, v, o, False)[0]


def forward_timed(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor | None = None
):
    out, runtime_ms = _load_ext().forward(q, k, v, o, True)
    return out, runtime_ms

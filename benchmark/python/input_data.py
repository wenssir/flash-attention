#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _case_seed(seed: int, case_index: int) -> int:
    return int(seed + case_index * 100_003)


def _target_shape(batch_size: int, n_heads: int, seq_len: int, d_head: int) -> tuple[int, int, int, int]:
    return (batch_size, n_heads, seq_len, d_head)


def _generator(seed: int, device: str) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def _make_random(shape: tuple[int, int, int, int], dtype: torch.dtype, device: str, seed: int):
    gen = _generator(seed, device)
    q = torch.randn(shape, device=device, dtype=dtype, generator=gen)
    k = torch.randn(shape, device=device, dtype=dtype, generator=gen)
    v = torch.randn(shape, device=device, dtype=dtype, generator=gen)
    return q.contiguous(), k.contiguous(), v.contiguous()


def _make_structured(shape: tuple[int, int, int, int], dtype: torch.dtype, device: str):
    b, h, n, d = shape
    bf = torch.arange(b, device=device, dtype=torch.float32).view(b, 1, 1, 1)
    hf = torch.arange(h, device=device, dtype=torch.float32).view(1, h, 1, 1)
    nf = torch.arange(n, device=device, dtype=torch.float32).view(1, 1, n, 1)
    df = torch.arange(d, device=device, dtype=torch.float32).view(1, 1, 1, d)

    q = torch.sin((nf + 1.0) * (hf + 1.0) / 17.0) + 0.01 * df + 0.05 * bf
    k = torch.cos((nf + 1.0) * (hf + 1.0) / 19.0) + 0.02 * df + 0.03 * bf
    v = torch.sin((df + 1.0) * (hf + 1.0) / 23.0) + 0.001 * nf + 0.02 * bf
    return q.to(dtype=dtype).contiguous(), k.to(dtype=dtype).contiguous(), v.to(dtype=dtype).contiguous()


def _make_stress(shape: tuple[int, int, int, int], dtype: torch.dtype, device: str, seed: int):
    gen = _generator(seed, device)
    q = torch.randn(shape, device=device, dtype=torch.float32, generator=gen) * 32.0
    k = torch.randn(shape, device=device, dtype=torch.float32, generator=gen) * 32.0
    v = torch.randn(shape, device=device, dtype=torch.float32, generator=gen) * 16.0

    # Inject sparse outliers to stress softmax stability.
    outlier_rate = 0.001
    q_mask = torch.rand(shape, device=device, generator=gen) < outlier_rate
    k_mask = torch.rand(shape, device=device, generator=gen) < outlier_rate
    v_mask = torch.rand(shape, device=device, generator=gen) < outlier_rate
    q = q + q_mask.to(torch.float32) * 256.0
    k = k - k_mask.to(torch.float32) * 256.0
    v = v + v_mask.to(torch.float32) * 64.0
    return q.to(dtype=dtype).contiguous(), k.to(dtype=dtype).contiguous(), v.to(dtype=dtype).contiguous()


def _pick_replay_file(replay_dir: Path, replay_file: str) -> Path:
    if replay_file:
        candidate = Path(replay_file).expanduser()
        if not candidate.is_absolute():
            candidate = replay_dir / candidate
        if candidate.exists():
            return candidate
        raise RuntimeError(f"Replay file not found: {candidate}")

    pt_files = sorted(replay_dir.glob("*.pt"))
    if pt_files:
        return pt_files[0]
    raise RuntimeError(
        f"Replay mode selected but no .pt file found in: {replay_dir}. "
        "Place a replay tensor file and rerun."
    )


def _maybe_permute_to_bhnd(t: torch.Tensor, target_shape: tuple[int, int, int, int]) -> torch.Tensor:
    if tuple(t.shape) == target_shape:
        return t
    b, h, n, d = target_shape
    if tuple(t.shape) == (b, n, h, d):
        return t.permute(0, 2, 1, 3).contiguous()
    raise RuntimeError(f"Replay tensor shape {tuple(t.shape)} does not match target {target_shape} or [B,N,H,D]")


def _extract_qkv(obj: Any):
    if isinstance(obj, dict):
        if all(k in obj for k in ("q", "k", "v")):
            return obj["q"], obj["k"], obj["v"]
        raise RuntimeError("Replay file dict must contain keys: q, k, v")
    if isinstance(obj, (list, tuple)) and len(obj) >= 3:
        return obj[0], obj[1], obj[2]
    raise RuntimeError("Replay file must be a dict(q,k,v) or tuple/list(q,k,v)")


def _make_replay(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    device: str,
    replay_dir: Path,
    replay_file: str,
):
    replay_path = _pick_replay_file(replay_dir, replay_file)
    payload = torch.load(replay_path, map_location="cpu")
    q, k, v = _extract_qkv(payload)
    q = _maybe_permute_to_bhnd(q, shape).to(device=device, dtype=dtype).contiguous()
    k = _maybe_permute_to_bhnd(k, shape).to(device=device, dtype=dtype).contiguous()
    v = _maybe_permute_to_bhnd(v, shape).to(device=device, dtype=dtype).contiguous()
    return q, k, v, replay_path


def generate_qkv(
    *,
    batch_size: int,
    n_heads: int,
    seq_len: int,
    d_head: int,
    dtype: torch.dtype,
    device: str,
    input_mode: str,
    seed: int,
    case_index: int,
    replay_dir: Path,
    replay_file: str,
):
    shape = _target_shape(batch_size, n_heads, seq_len, d_head)
    seeded = _case_seed(seed, case_index)
    if input_mode == "random":
        q, k, v = _make_random(shape, dtype, device, seeded)
        return q, k, v, None
    if input_mode == "structured":
        q, k, v = _make_structured(shape, dtype, device)
        return q, k, v, None
    if input_mode == "stress":
        q, k, v = _make_stress(shape, dtype, device, seeded)
        return q, k, v, None
    if input_mode == "replay":
        q, k, v, replay_path = _make_replay(shape, dtype, device, replay_dir, replay_file)
        return q, k, v, replay_path
    raise RuntimeError(f"Unsupported input_mode: {input_mode}")

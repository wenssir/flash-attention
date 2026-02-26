#!/usr/bin/env python3

import argparse
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from input_data import generate_qkv


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
sys.path.insert(0, str(PYTHON_DIR))

import flash_attention_runtime as fa2  # noqa: E402

BATCH_SIZE_FOR_SEQ_LEN = {
    512: 16,
    1024: 16,
    2048: 16,
    4096: 16,
    8192: 8,
    16384: 4,
}

CSV_HEADER = (
    "mode,kernel,dtype,d_head,seq_len,batch_size,n_heads,warmups,repeats,stabilize,"
    "baseline,input_mode,mean_ms,median_ms,min_ms,max_ms,stddev_ms,rel_perf_pct,attn_tflops,"
    "git_commit,gpu_name,torch_version,cuda_version"
)


@dataclass
class BenchmarkStats:
    mean: float
    median: float
    min: float
    max: float
    stddev: float
    attn_tflops: float

    def relative_perf(self, baseline_mean: float) -> float:
        return 100.0 * baseline_mean / self.mean


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            )
            .strip()
        )
    except Exception:
        return "unknown"


def calc_self_attn_flop(batch: int, heads: int, seq_len: int, d_head: int) -> int:
    return batch * heads * (4 * seq_len * seq_len * d_head + 6 * seq_len * seq_len)


def summarize(samples_ms: list[float], attn_flops: int) -> BenchmarkStats:
    mean = statistics.mean(samples_ms)
    median = statistics.median(samples_ms)
    min_v = min(samples_ms)
    max_v = max(samples_ms)
    stddev = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    attn_tflops = attn_flops / (mean * 1e6) / 1e3
    return BenchmarkStats(
        mean=mean,
        median=median,
        min=min_v,
        max=max_v,
        stddev=stddev,
        attn_tflops=attn_tflops,
    )


def pytorch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


@torch.inference_mode()
def benchmark_kernel(
    fn,
    warmups: int,
    repeats: int,
    flush_l2: bool,
    cache_buf: torch.Tensor | None,
) -> list[float]:
    for _ in range(warmups):
        _ = fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        if flush_l2 and cache_buf is not None:
            cache_buf.zero_()
            torch.cuda._sleep(1_000_000)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()

        if isinstance(out, tuple):
            samples.append(float(out[1]))
        else:
            samples.append(float(start.elapsed_time(end)))
    return samples


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark FlashAttention pybind interface")
    p.add_argument("--d_heads", type=str, default="128", help="comma separated head dims")
    p.add_argument("--seq_lens", type=str, default="4096", help="comma separated seq lens")
    p.add_argument("--batch_size", type=int, default=8, help="override batch size for all seq lens")
    p.add_argument("--n_heads", type=int, default=16, help="number of attention heads")
    p.add_argument("--warmups", type=int, default=10, help="warmup iterations")
    p.add_argument("--repeats", type=int, default=50, help="repeat iterations")
    p.add_argument("--no_stabilize", action="store_true", help="disable L2 flush and sleep")
    p.add_argument("--csv", action="store_true", help="print CSV format")
    p.add_argument("--out_csv", type=str, default="", help="csv output path (default: profile_results/pybind_<timestamp>.csv)")
    p.add_argument("--no_save", action="store_true", help="do not save csv file to disk")
    p.add_argument(
        "--baseline",
        type=str,
        default="pytorch",
        choices=["pytorch", "fa2_pybind"],
        help="kernel used as 100%% for rel_perf",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="input dtype for this script (fp16 currently uses forward_v3_tensor_core path, d_head=128 only)",
    )
    p.add_argument(
        "--input_mode",
        type=str,
        default="random",
        choices=["random", "structured", "stress", "replay"],
        help="input generation strategy",
    )
    p.add_argument("--seed", type=int, default=1234, help="base seed for random/stress input")
    p.add_argument(
        "--replay_dir",
        type=str,
        default="benchmark/data/replay",
        help="directory containing replay .pt files",
    )
    p.add_argument(
        "--replay_file",
        type=str,
        default="",
        help="specific replay file path (absolute or under replay_dir)",
    )
    return p.parse_args()


def default_out_csv() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "profile_results" / f"pybind_{ts}.csv"


def print_table(title: str, rows: list[list[str]]):
    print(title)
    header = [
        "Kernel",
        "d_head",
        "seq_len",
        "Mean(ms)",
        "Median",
        "Min",
        "Max",
        "StdDev",
        "RelPerf",
        "Attn TFLOP/s",
    ]
    widths = [max(len(str(x)) for x in [h] + [r[i] for r in rows]) for i, h in enumerate(header)]
    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    sep = "-+-".join("-" * w for w in widths)
    print(line)
    print(sep)
    for r in rows:
        print(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(header))))
    print()


def csv_line(
    kernel: str,
    stats: BenchmarkStats,
    baseline_mean: float,
    args,
    d_head: int,
    seq_len: int,
    batch_size: int,
    meta: dict,
) -> str:
    return (
        f"pybind,{kernel},{args.dtype},{d_head},{seq_len},{batch_size},{args.n_heads},"
        f"{args.warmups},{args.repeats},{int(not args.no_stabilize)},{args.baseline},{args.input_mode},"
        f"{stats.mean:.6f},{stats.median:.6f},{stats.min:.6f},{stats.max:.6f},{stats.stddev:.6f},"
        f"{stats.relative_perf(baseline_mean):.2f},{stats.attn_tflops:.6f},"
        f"{meta['git_commit']},{meta['gpu_name']},{meta['torch_version']},{meta['cuda_version']}"
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    d_heads = [int(x) for x in args.d_heads.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    meta = {
        "git_commit": git_commit(),
        "gpu_name": torch.cuda.get_device_name(0).replace(",", " "),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "unknown",
    }

    print(f"GPU: {meta['gpu_name']}")
    print(f"torch={meta['torch_version']} cuda={meta['cuda_version']} commit={meta['git_commit'][:12]}")
    print(f"warmups={args.warmups}, repeats={args.repeats}, stabilize={not args.no_stabilize}")
    print(f"baseline={args.baseline}, dtype={args.dtype}, input_mode={args.input_mode}, seed={args.seed}")
    if args.dtype == "fp16":
        print("note: fa2_pybind fp16 currently uses forward_v3_tensor_core path; only d_head=128")
    print()

    cache_buf = None
    if not args.no_stabilize:
        cache_buf = torch.empty(int(100 * (1024**2)), dtype=torch.int8, device="cuda")

    first_csv = True
    csv_rows: list[str] = []
    case_index = 0
    replay_dir = Path(args.replay_dir).expanduser()
    for d_head in d_heads:
        for seq_len in seq_lens:
            if d_head not in (64, 128):
                print(f"skip d_head={d_head}: pybind kernel currently supports only 64/128")
                continue
            if args.dtype == "fp16" and d_head != 128:
                print(f"skip d_head={d_head}: fp16 path supports d_head=128 only")
                continue

            batch_size = args.batch_size if args.batch_size > 0 else BATCH_SIZE_FOR_SEQ_LEN.get(seq_len, 4)
            input_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
            q, k, v, replay_path = generate_qkv(
                batch_size=batch_size,
                n_heads=args.n_heads,
                seq_len=seq_len,
                d_head=d_head,
                dtype=input_dtype,
                device="cuda",
                input_mode=args.input_mode,
                seed=args.seed,
                case_index=case_index,
                replay_dir=replay_dir,
                replay_file=args.replay_file,
            )
            case_index += 1
            if replay_path is not None:
                print(f"replay input: {replay_path}")

            pt_fn = lambda: pytorch_attention(q, k, v)
            fa_fn = lambda: fa2.forward_timed(q, k, v)

            pt_samples = benchmark_kernel(pt_fn, args.warmups, args.repeats, not args.no_stabilize, cache_buf)
            fa_samples = benchmark_kernel(fa_fn, args.warmups, args.repeats, not args.no_stabilize, cache_buf)

            attn_flops = calc_self_attn_flop(batch_size, args.n_heads, seq_len, d_head)
            stats_map = {
                "pytorch": summarize(pt_samples, attn_flops),
                "fa2_pybind": summarize(fa_samples, attn_flops),
            }
            baseline_mean = stats_map[args.baseline].mean

            if args.csv:
                if first_csv:
                    print(CSV_HEADER)
                    first_csv = False
                row_pt = csv_line("pytorch", stats_map["pytorch"], baseline_mean, args, d_head, seq_len, batch_size, meta)
                row_fa = csv_line("fa2_pybind", stats_map["fa2_pybind"], baseline_mean, args, d_head, seq_len, batch_size, meta)
                print(row_pt)
                print(row_fa)
                csv_rows.extend([row_pt, row_fa])
                continue

            row_pt = csv_line("pytorch", stats_map["pytorch"], baseline_mean, args, d_head, seq_len, batch_size, meta)
            row_fa = csv_line("fa2_pybind", stats_map["fa2_pybind"], baseline_mean, args, d_head, seq_len, batch_size, meta)
            csv_rows.extend([row_pt, row_fa])

            rows = [
                [
                    "pytorch",
                    str(d_head),
                    str(seq_len),
                    f"{stats_map['pytorch'].mean:.4f}",
                    f"{stats_map['pytorch'].median:.4f}",
                    f"{stats_map['pytorch'].min:.4f}",
                    f"{stats_map['pytorch'].max:.4f}",
                    f"{stats_map['pytorch'].stddev:.4f}",
                    f"{stats_map['pytorch'].relative_perf(baseline_mean):.2f}%",
                    f"{stats_map['pytorch'].attn_tflops:.4f}",
                ],
                [
                    "fa2_pybind",
                    str(d_head),
                    str(seq_len),
                    f"{stats_map['fa2_pybind'].mean:.4f}",
                    f"{stats_map['fa2_pybind'].median:.4f}",
                    f"{stats_map['fa2_pybind'].min:.4f}",
                    f"{stats_map['fa2_pybind'].max:.4f}",
                    f"{stats_map['fa2_pybind'].stddev:.4f}",
                    f"{stats_map['fa2_pybind'].relative_perf(baseline_mean):.2f}%",
                    f"{stats_map['fa2_pybind'].attn_tflops:.4f}",
                ],
            ]
            print_table(
                f"--- d_head={d_head}, seq_len={seq_len}, batch={batch_size}, heads={args.n_heads} ---",
                rows,
            )

    if not args.no_save:
        out_csv = Path(args.out_csv) if args.out_csv else default_out_csv()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8") as f:
            f.write(CSV_HEADER + "\n")
            for row in csv_rows:
                f.write(row + "\n")
        print(f"saved csv: {out_csv}")


if __name__ == "__main__":
    main()

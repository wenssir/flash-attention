#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
PY_BENCH_DIR = THIS_DIR.parent
sys.path.insert(0, str(PY_BENCH_DIR))

from input_data import generate_qkv  # noqa: E402

import flash_attention_runtime as fa2  # noqa: E402


def pytorch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def parse_cases(text: str) -> list[tuple[int, int, int, int]]:
    out = []
    for item in text.split(","):
        item = item.strip().lower().replace("*", "x")
        if not item:
            continue
        parts = item.split("x")
        if len(parts) != 4:
            raise ValueError(f"invalid case '{item}', expected BxHxNxD")
        b, h, n, d = map(int, parts)
        out.append((b, h, n, d))
    if not out:
        raise ValueError("no valid cases parsed")
    return out


def default_tolerances(dtype: str) -> tuple[float, float]:
    if dtype == "fp16":
        return 2e-2, 2e-2
    return 1e-3, 1e-3


def parse_args():
    p = argparse.ArgumentParser(description="Correctness gate for flash_attention_runtime pybind kernel")
    p.add_argument(
        "--cases",
        type=str,
        default="2x4x128x64,2x4x256x64,2x4x128x128,2x4x256x128",
        help="comma-separated BxHxNxD cases",
    )
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"], help="input dtype")
    p.add_argument("--atol", type=float, default=-1.0, help="absolute tolerance; default depends on dtype")
    p.add_argument("--rtol", type=float, default=-1.0, help="relative tolerance; default depends on dtype")
    p.add_argument(
        "--input_mode",
        type=str,
        default="random",
        choices=["random", "structured", "stress", "replay"],
        help="input generation strategy",
    )
    p.add_argument("--seed", type=int, default=1234, help="base seed for random/stress input")
    p.add_argument("--replay_dir", type=str, default="benchmark/data/replay", help="replay data directory")
    p.add_argument("--replay_file", type=str, default="", help="specific replay file")
    p.add_argument("--fail_fast", action="store_true", help="stop at first failed case")
    p.add_argument("--out_json", type=str, default="", help="optional json report path")
    p.add_argument("--out_csv", type=str, default="", help="optional csv report path")
    p.add_argument(
        "--print_first_row",
        action="store_true",
        help="print ref/out/diff/rel of first row (B0,H0,N0) for each executed case",
    )
    return p.parse_args()


def run_one_case(
    *,
    batch: int,
    heads: int,
    seq_len: int,
    d_head: int,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    input_mode: str,
    seed: int,
    case_index: int,
    replay_dir: Path,
    replay_file: str,
    print_first_row: bool,
):
    q, k, v, replay_path = generate_qkv(
        batch_size=batch,
        n_heads=heads,
        seq_len=seq_len,
        d_head=d_head,
        dtype=dtype,
        device="cuda",
        input_mode=input_mode,
        seed=seed,
        case_index=case_index,
        replay_dir=replay_dir,
        replay_file=replay_file,
    )

    out_ref = pytorch_attention(q, k, v)
    out = fa2.forward(q, k, v)
    diff = (out_ref - out).abs()
    denom = out_ref.abs().clamp_min(1e-6)
    rel = diff / denom
    ok = torch.allclose(out_ref, out, atol=atol, rtol=rtol)

    row = {
        "B": batch,
        "H": heads,
        "N": seq_len,
        "D": d_head,
        "status": "PASS" if ok else "FAIL",
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "max_rel_diff": float(rel.max().item()),
        "replay_path": str(replay_path) if replay_path is not None else "",
    }

    if print_first_row:
        n_show = min(8, d_head)
        ref_row = out_ref[0, 0, 0, :n_show].detach().float().cpu().tolist()
        out_row = out[0, 0, 0, :n_show].detach().float().cpu().tolist()
        diff_row = diff[0, 0, 0, :n_show].detach().float().cpu().tolist()
        rel_row = rel[0, 0, 0, :n_show].detach().float().cpu().tolist()
        print(f"  first_row ref[:{n_show}]={ref_row}")
        print(f"  first_row out[:{n_show}]={out_row}")
        print(f"  first_row diff[:{n_show}]={diff_row}")
        print(f"  first_row rel[:{n_show}]={rel_row}")

    return row


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    atol, rtol = default_tolerances(args.dtype)
    if args.atol >= 0:
        atol = args.atol
    if args.rtol >= 0:
        rtol = args.rtol

    input_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    replay_dir = Path(args.replay_dir).expanduser()
    cases = parse_cases(args.cases)

    report_rows = []
    failed = 0
    skipped = 0
    for i, (b, h, n, d) in enumerate(cases):
        if args.dtype == "fp16" and d != 128:
            row = {"B": b, "H": h, "N": n, "D": d, "status": "SKIP", "max_abs_diff": 0.0, "mean_abs_diff": 0.0, "max_rel_diff": 0.0, "replay_path": ""}
            report_rows.append(row)
            skipped += 1
            print(f"case B={b} H={h} N={n} D={d}: SKIP (fp16 path requires D=128)")
            continue

        row = run_one_case(
            batch=b,
            heads=h,
            seq_len=n,
            d_head=d,
            dtype=input_dtype,
            atol=atol,
            rtol=rtol,
            input_mode=args.input_mode,
            seed=args.seed,
            case_index=i,
            replay_dir=replay_dir,
            replay_file=args.replay_file,
            print_first_row=args.print_first_row,
        )
        report_rows.append(row)
        print(
            f"case B={b} H={h} N={n} D={d}: {row['status']}, "
            f"max_abs={row['max_abs_diff']:.6e}, mean_abs={row['mean_abs_diff']:.6e}, "
            f"max_rel={row['max_rel_diff']:.6e}"
        )
        if row["status"] != "PASS":
            failed += 1
            if args.fail_fast:
                break

    passed = sum(1 for x in report_rows if x["status"] == "PASS")
    print(
        f"\nsummary: pass={passed}, fail={failed}, skip={skipped}, "
        f"atol={atol:.3e}, rtol={rtol:.3e}, dtype={args.dtype}, input_mode={args.input_mode}"
    )

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(
                {
                    "dtype": args.dtype,
                    "input_mode": args.input_mode,
                    "atol": atol,
                    "rtol": rtol,
                    "cases": cases,
                    "rows": report_rows,
                    "pass": passed,
                    "fail": failed,
                    "skip": skipped,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved json: {out_json}")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["B", "H", "N", "D", "status", "max_abs_diff", "mean_abs_diff", "max_rel_diff", "replay_path"],
            )
            writer.writeheader()
            for row in report_rows:
                writer.writerow(row)
        print(f"saved csv: {out_csv}")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

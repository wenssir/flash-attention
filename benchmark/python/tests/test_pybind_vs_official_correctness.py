#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
PY_BENCH_DIR = THIS_DIR.parent
REPO_ROOT = PY_BENCH_DIR.parents[1]
sys.path.insert(0, str(PY_BENCH_DIR))

from input_data import generate_qkv  # noqa: E402

import flash_attention_runtime as fa2  # noqa: E402


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
    if dtype == "bf16":
        return 3e-2, 3e-2
    return 2e-2, 2e-2


def parse_args():
    p = argparse.ArgumentParser(description="Correctness compare: pybind vs official flash-attn")
    p.add_argument(
        "--cases",
        type=str,
        default="2x4x128x128,2x4x256x128,2x4x512x128",
        help="comma-separated BxHxNxD cases",
    )
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"], help="input dtype")
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
    p.add_argument("--official_path", type=str, default="~/flash-attention", help="official flash-attention repo path")
    p.add_argument("--print_first_row", action="store_true", help="print first row values for each case")
    p.add_argument("--out_json", type=str, default="", help="optional json report path")
    p.add_argument("--out_csv", type=str, default="", help="optional csv report path")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    official_path = Path(args.official_path).expanduser()
    sys.path.insert(0, str(official_path))
    try:
        from flash_attn import flash_attn_func  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import official flash-attn. "
            f"Make sure it is installed and importable from {official_path}. "
            f"Original error: {e}"
        ) from e

    atol, rtol = default_tolerances(args.dtype)
    if args.atol >= 0:
        atol = args.atol
    if args.rtol >= 0:
        rtol = args.rtol

    in_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    replay_dir = Path(args.replay_dir).expanduser()
    cases = parse_cases(args.cases)

    report_rows = []
    failed = 0
    skipped = 0
    for i, (b, h, n, d) in enumerate(cases):
        # Current pybind fp16/bf16 path is v3 and requires D=128.
        if d != 128:
            report_rows.append(
                {
                    "B": b,
                    "H": h,
                    "N": n,
                    "D": d,
                    "status": "SKIP",
                    "max_abs_diff": 0.0,
                    "mean_abs_diff": 0.0,
                    "max_rel_diff": 0.0,
                    "replay_path": "",
                }
            )
            skipped += 1
            print(f"case B={b} H={h} N={n} D={d}: SKIP (pybind non-fp32 path requires D=128)")
            continue

        q, k, v, replay_path = generate_qkv(
            batch_size=b,
            n_heads=h,
            seq_len=n,
            d_head=d,
            dtype=in_dtype,
            device="cuda",
            input_mode=args.input_mode,
            seed=args.seed,
            case_index=i,
            replay_dir=replay_dir,
            replay_file=args.replay_file,
        )

        out_pybind = fa2.forward(q, k, v)

        q_off = q.permute(0, 2, 1, 3).contiguous()
        k_off = k.permute(0, 2, 1, 3).contiguous()
        v_off = v.permute(0, 2, 1, 3).contiguous()
        out_off = flash_attn_func(q_off, k_off, v_off, 0.0, None, False).permute(0, 2, 1, 3).contiguous()

        diff = (out_off - out_pybind).abs()
        denom = out_off.abs().clamp_min(1e-6)
        rel = diff / denom
        ok = torch.allclose(out_off, out_pybind, atol=atol, rtol=rtol)

        row = {
            "B": b,
            "H": h,
            "N": n,
            "D": d,
            "status": "PASS" if ok else "FAIL",
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
            "max_rel_diff": float(rel.max().item()),
            "replay_path": str(replay_path) if replay_path is not None else "",
        }
        report_rows.append(row)

        if args.print_first_row:
            n_show = min(8, d)
            off_row = out_off[0, 0, 0, :n_show].detach().float().cpu().tolist()
            py_row = out_pybind[0, 0, 0, :n_show].detach().float().cpu().tolist()
            diff_row = diff[0, 0, 0, :n_show].detach().float().cpu().tolist()
            rel_row = rel[0, 0, 0, :n_show].detach().float().cpu().tolist()
            print(f"  first_row official[:{n_show}]={off_row}")
            print(f"  first_row pybind[:{n_show}]={py_row}")
            print(f"  first_row diff[:{n_show}]={diff_row}")
            print(f"  first_row rel[:{n_show}]={rel_row}")

        print(
            f"case B={b} H={h} N={n} D={d}: {row['status']}, "
            f"max_abs={row['max_abs_diff']:.6e}, mean_abs={row['mean_abs_diff']:.6e}, "
            f"max_rel={row['max_rel_diff']:.6e}"
        )
        if row["status"] != "PASS":
            failed += 1

    passed = sum(1 for x in report_rows if x["status"] == "PASS")
    print(
        f"\nsummary: pass={passed}, fail={failed}, skip={skipped}, "
        f"atol={atol:.3e}, rtol={rtol:.3e}, dtype={args.dtype}, input_mode={args.input_mode}, "
        f"official_path={official_path}"
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
                    "official_path": str(official_path),
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


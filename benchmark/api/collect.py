from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from .schema import CSV_COLUMNS


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        return "unknown"


def _calc_self_attn_flop(batch: int, heads: int, seq_len: int, d_head: int) -> int:
    return batch * heads * (4 * seq_len * seq_len * d_head + 6 * seq_len * seq_len)


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(rows: list[dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for row in rows:
            safe_row = {k: row.get(k, "") for k in CSV_COLUMNS}
            w.writerow(safe_row)


def merge_csvs(csv_files: list[Path], out_csv: Path, run_id: str) -> int:
    rows: list[dict[str, str]] = []
    for path in csv_files:
        if not path.exists():
            continue
        for row in read_rows(path):
            row["run_id"] = run_id
            rows.append(row)
    write_rows(rows, out_csv)
    return len(rows)


def recompute_rel_perf_pct(rows: list[dict[str, str]], baseline: str) -> list[dict[str, str]]:
    baseline_kernel = "pytorch" if baseline == "pytorch" else "flashattn_official"
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row.get("dtype", ""),
            row.get("d_head", ""),
            row.get("seq_len", ""),
            row.get("batch_size", ""),
            row.get("n_heads", ""),
            row.get("input_mode", ""),
        )
        groups.setdefault(key, []).append(row)

    for _, grows in groups.items():
        base_mean = None
        for r in grows:
            if r.get("kernel") == baseline_kernel:
                try:
                    base_mean = float(r.get("mean_ms", ""))
                except Exception:
                    base_mean = None
                break
        # If configured baseline row doesn't exist in this group, keep existing rel_perf_pct.
        if base_mean is None or base_mean <= 0:
            continue
        for r in grows:
            try:
                mean_ms = float(r.get("mean_ms", ""))
                if mean_ms > 0:
                    r["rel_perf_pct"] = f"{100.0 * base_mean / mean_ms:.2f}"
            except Exception:
                continue
    return rows


def convert_profile_kernel_log_to_rows(
    *,
    log_path: Path,
    repo_root: Path,
    mode: str,
    kernel: str,
    dtype: str,
    baseline: str,
    input_mode: str,
    warmups: int,
    repeats: int,
    run_id: str,
) -> list[dict[str, str]]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    csv_lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("CSV,")]
    if not csv_lines:
        return []

    gpu_name = "unknown"
    for ln in text.splitlines():
        if ln.strip().startswith("Name:"):
            gpu_name = ln.split(":", 1)[1].strip().replace(",", " ")
            break

    commit = _git_commit(repo_root)
    out_rows: list[dict[str, str]] = []

    for line in csv_lines:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 13:
            continue

        batch = int(parts[1])
        n_heads = int(parts[2])
        seq_len = int(parts[3])
        d_head = int(parts[4])
        mean_ms = float(parts[5])
        # CSV,B,H,N,d,time_ms,gflops,compute_eff,memory,bandwidth,bw_util,tok_ms,tok_s
        memory_gb = float(parts[8])
        bw_gbps = float(parts[9])
        tok_ms = float(parts[11])
        tok_s = float(parts[12])

        theoretical = _calc_self_attn_flop(batch, n_heads, seq_len, d_head)
        achieved_gflops = theoretical / (mean_ms * 1e6)
        attn_tflops = theoretical / (mean_ms * 1e6) / 1e3
        io_bytes_est = int(memory_gb * 1e9)

        row = {
            "mode": mode,
            "kernel": kernel,
            "dtype": dtype,
            "d_head": str(d_head),
            "seq_len": str(seq_len),
            "batch_size": str(batch),
            "n_heads": str(n_heads),
            "warmups": str(warmups),
            "repeats": str(repeats),
            "stabilize": "0",
            "baseline": baseline,
            "input_mode": input_mode,
            "mean_ms": f"{mean_ms:.6f}",
            "median_ms": f"{mean_ms:.6f}",
            "min_ms": f"{mean_ms:.6f}",
            "max_ms": f"{mean_ms:.6f}",
            "stddev_ms": "0.000000",
            "rel_perf_pct": "100.00",
            "attn_tflops": f"{attn_tflops:.6f}",
            "theoretical_flops": str(theoretical),
            "achieved_gflops": f"{achieved_gflops:.6f}",
            "throughput_tokens_per_ms": f"{tok_ms:.6f}",
            "throughput_tokens_per_s": f"{tok_s:.6f}",
            "io_bytes_est": str(io_bytes_est),
            "est_bw_gbps": f"{bw_gbps:.6f}",
            "git_commit": commit,
            "gpu_name": gpu_name,
            "torch_version": "na",
            "cuda_version": "na",
            "run_id": run_id,
        }
        out_rows.append(row)

    return out_rows

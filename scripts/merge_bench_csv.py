#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest_file(pattern: str) -> Path | None:
    files = sorted(Path("profile_results").glob(pattern))
    return files[-1] if files else None


def to_float(v: str) -> float:
    return float(v) if v else 0.0


def key_of(row: dict) -> tuple:
    return (
        row.get("input_mode", "random"),
        row["dtype"],
        row["d_head"],
        row["seq_len"],
        row["batch_size"],
        row["n_heads"],
    )


def index_rows(rows: list[dict], kernel_name: str) -> dict:
    out = {}
    for r in rows:
        if r.get("kernel") == kernel_name:
            out[key_of(r)] = r
    return out


def main():
    parser = argparse.ArgumentParser(description="Merge pybind and official benchmark CSV files")
    parser.add_argument("--pybind_csv", type=str, default="", help="path to pybind csv")
    parser.add_argument("--official_csv", type=str, default="", help="path to official csv")
    parser.add_argument("--out_csv", type=str, default="", help="output path")
    args = parser.parse_args()

    pybind_csv = Path(args.pybind_csv) if args.pybind_csv else latest_file("pybind_*.csv")
    official_csv = Path(args.official_csv) if args.official_csv else latest_file("official_*.csv")
    if pybind_csv is None or official_csv is None:
        raise SystemExit("Could not find pybind/official csv in profile_results. Provide --pybind_csv and --official_csv.")

    pybind_rows = read_csv_rows(pybind_csv)
    official_rows = read_csv_rows(official_csv)

    pybind_map = index_rows(pybind_rows, "fa2_pybind")
    official_map = index_rows(official_rows, "flashattn_official")
    pybind_pt_map = index_rows(pybind_rows, "pytorch")
    official_pt_map = index_rows(official_rows, "pytorch")

    keys = sorted(set(pybind_map.keys()) & set(official_map.keys()))
    if not keys:
        raise SystemExit("No overlapping (input_mode, dtype, d_head, seq_len, batch_size, n_heads) keys found between CSV files.")

    out_rows = []
    for k in keys:
        p = pybind_map[k]
        o = official_map[k]
        p_pt = pybind_pt_map.get(k)
        o_pt = official_pt_map.get(k)

        pybind_ms = to_float(p["mean_ms"])
        official_ms = to_float(o["mean_ms"])
        pybind_tflops = to_float(p["attn_tflops"])
        official_tflops = to_float(o["attn_tflops"])
        pybind_pt_ms = to_float(p_pt["mean_ms"]) if p_pt else 0.0
        official_pt_ms = to_float(o_pt["mean_ms"]) if o_pt else 0.0

        out_rows.append(
            {
                "input_mode": k[0],
                "dtype": k[1],
                "d_head": k[2],
                "seq_len": k[3],
                "batch_size": k[4],
                "n_heads": k[5],
                "pybind_mean_ms": f"{pybind_ms:.6f}",
                "official_mean_ms": f"{official_ms:.6f}",
                "pytorch_mean_ms_pybind_run": f"{pybind_pt_ms:.6f}",
                "pytorch_mean_ms_official_run": f"{official_pt_ms:.6f}",
                "pybind_attn_tflops": f"{pybind_tflops:.6f}",
                "official_attn_tflops": f"{official_tflops:.6f}",
                "pybind_vs_official_speedup": f"{(official_ms / pybind_ms):.6f}",
                "official_vs_pybind_speedup": f"{(pybind_ms / official_ms):.6f}",
                "pybind_csv": str(pybind_csv),
                "official_csv": str(official_csv),
            }
        )

    out_csv = Path(args.out_csv) if args.out_csv else Path("profile_results") / "merged_latest.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"merged rows: {len(out_rows)}")
    print(f"output: {out_csv}")


if __name__ == "__main__":
    main()

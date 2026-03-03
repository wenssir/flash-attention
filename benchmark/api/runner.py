#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .collect import (
    convert_profile_kernel_log_to_rows,
    merge_csvs,
    read_rows,
    recompute_rel_perf_pct,
    write_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n")
        lf.flush()
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            lf.write(line)
        p.wait()
        return p.returncode


def _parse_shape(shape: str) -> tuple[int, int, int, int]:
    parts = shape.lower().replace("*", "x").split("x")
    if len(parts) != 4:
        raise ValueError(f"invalid shape: {shape}, expected BxHxNxD")
    b, h, n, d = map(int, parts)
    return b, h, n, d


def _ensure_profile_kernel(args, env, run_dir: Path, strict: bool) -> bool:
    build_dir = (REPO_ROOT / args.build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake_cfg = [
        "cmake",
        "-S",
        str(REPO_ROOT),
        "-B",
        str(build_dir),
        "-DFA_BUILD_TESTS=OFF",
        "-DFA_BUILD_UTILS=ON",
        "-DFA_BUILD_EXAMPLES=OFF",
    ]
    if args.cuda_arch:
        cmake_cfg.append(f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}")
    rc = _run(cmake_cfg, cwd=REPO_ROOT, env=env, log_path=run_dir / "logs" / "cmake_configure.log")
    if rc != 0:
        if strict:
            raise RuntimeError("cmake configure failed")
        return False

    cmake_build = ["cmake", "--build", str(build_dir), "-j", "--target", "profile_kernel"]
    rc = _run(cmake_build, cwd=REPO_ROOT, env=env, log_path=run_dir / "logs" / "cmake_build.log")
    if rc != 0:
        if strict:
            raise RuntimeError("cmake build failed")
        return False

    exe = build_dir / "profile_kernel"
    return exe.exists()


def _append_csv(dst: Path, src: Path) -> None:
    if not src.exists():
        return
    rows = []
    with src.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    exists = dst.exists()
    with dst.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified benchmark runner")
    p.add_argument("--run-name", default="bench")
    p.add_argument("--output-dir", default="profile_results")
    p.add_argument("--run-id", default="")
    p.add_argument("--run-info", default="")

    p.add_argument("--seq-lens", default="4096")
    p.add_argument("--d-heads", default="128")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--profile-shape", default="16x16x4096x128")

    p.add_argument("--warmups", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--input-mode", default="random")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--replay-dir", default="benchmark/data/replay")
    p.add_argument("--replay-file", default="")
    p.add_argument("--baseline", default="pytorch", choices=["pytorch", "official"])

    p.add_argument("--run-pybind", type=int, default=1)
    p.add_argument("--run-official", type=int, default=0)
    p.add_argument("--run-pytorch", type=int, default=0)
    p.add_argument("--run-profile-benchmark", type=int, default=1)
    p.add_argument("--run-profile-kernel", type=int, default=0)
    p.add_argument("--run-ncu", type=int, default=0)
    p.add_argument("--strict", type=int, default=0)

    p.add_argument("--official-path", default=str(Path.home() / "flash-attention"))
    p.add_argument("--official-dtype", default="fp16")
    p.add_argument("--official-pytorch-dtype", default="match")

    p.add_argument("--build-dir", default="build")
    p.add_argument("--cuda-arch", default="")
    p.add_argument("--ncu-cmd", default="ncu")
    p.add_argument("--ncu-set", default="full")
    p.add_argument("--ncu-launch-skip", default="")
    p.add_argument("--ncu-launch-count", default="")
    p.add_argument("--ncu-use-sudo", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # backward compatibility: old flag --run-profile-kernel maps to benchmark stage
    if args.run_profile_kernel == 1:
        args.run_profile_benchmark = 1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"{args.run_name}_{ts}"

    run_dir = (REPO_ROOT / args.output_dir / run_id).resolve()
    logs_dir = run_dir / "logs"
    tables_dir = run_dir / "tables"
    artifacts_dir = run_dir / "artifacts"
    for d in (run_dir, logs_dir, tables_dir, artifacts_dir):
        d.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'python'}:{env.get('PYTHONPATH', '')}"

    csv_parts: list[Path] = []
    strict = args.strict == 1
    summary: dict[str, object] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "tasks": {},
    }

    # 1) pybind
    if args.run_pybind == 1:
        pybind_csv = tables_dir / "pybind.csv"
        pybind_baseline = args.baseline
        if pybind_baseline == "official":
            pybind_baseline = "official"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "benchmark/python/benchmark_pybind.py"),
            "--seq_lens",
            args.seq_lens,
            "--d_heads",
            args.d_heads,
            "--batch_size",
            str(args.batch_size),
            "--n_heads",
            str(args.n_heads),
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
            "--dtype",
            args.dtype,
            "--input_mode",
            args.input_mode,
            "--seed",
            str(args.seed),
            "--replay_dir",
            args.replay_dir,
            "--replay_file",
            args.replay_file,
            "--baseline",
            pybind_baseline,
            "--csv",
            "--out_csv",
            str(pybind_csv),
        ]
        rc = _run(cmd, cwd=REPO_ROOT, env=env, log_path=logs_dir / "pybind.log")
        summary["tasks"]["pybind"] = {"rc": rc, "csv": str(pybind_csv)}
        if rc != 0 and strict:
            raise RuntimeError("pybind benchmark failed")
        if rc == 0:
            csv_parts.append(pybind_csv)

    # 2) official/pytorch
    if args.run_official == 1 or args.run_pytorch == 1:
        official_csv = tables_dir / "official_pytorch.csv"
        if args.run_official == 1 and args.run_pytorch == 1:
            kernels = "both"
        elif args.run_official == 1:
            kernels = "official"
        else:
            kernels = "pytorch"

        baseline = args.baseline
        if baseline == "official":
            baseline = "flashattn_official"
        if kernels == "pytorch" and baseline == "flashattn_official":
            baseline = "pytorch"

        cmd = [
            sys.executable,
            str(REPO_ROOT / "benchmark/python/benchmark_official_flashattn.py"),
            "--seq_lens",
            args.seq_lens,
            "--d_heads",
            args.d_heads,
            "--batch_size",
            str(args.batch_size),
            "--n_heads",
            str(args.n_heads),
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
            "--dtype",
            args.official_dtype,
            "--pytorch_dtype",
            args.official_pytorch_dtype,
            "--input_mode",
            args.input_mode,
            "--seed",
            str(args.seed),
            "--replay_dir",
            args.replay_dir,
            "--replay_file",
            args.replay_file,
            "--baseline",
            baseline,
            "--kernels",
            kernels,
            "--csv",
            "--out_csv",
            str(official_csv),
        ]
        # Script imports official from $HOME/flash-attention. Preserve compatibility.
        env_off = env.copy()
        env_off["HOME"] = str(Path.home())
        rc = _run(cmd, cwd=REPO_ROOT, env=env_off, log_path=logs_dir / "official_pytorch.log")
        summary["tasks"]["official_pytorch"] = {"rc": rc, "csv": str(official_csv), "kernels": kernels}
        if rc != 0 and strict:
            raise RuntimeError("official/pytorch benchmark failed")
        if rc == 0:
            csv_parts.append(official_csv)

    # 3) profile_kernel benchmark
    profile_kernel_bench_ok = True
    if args.run_profile_benchmark == 1:
        ok = _ensure_profile_kernel(args, env, run_dir, strict)
        summary["tasks"]["profile_kernel_build"] = {"ok": ok}
        if ok:
            b, h, n, d = _parse_shape(args.profile_shape)
            shape = f"{b}x{h}x{n}x{d}"
            exe = (REPO_ROOT / args.build_dir / "profile_kernel").resolve()
            bench_log = logs_dir / "profile_kernel_benchmark.log"
            cmd = [str(exe), "--mode", "benchmark", shape]
            rc = _run(cmd, cwd=exe.parent, env=env, log_path=bench_log)
            summary["tasks"]["profile_kernel_benchmark"] = {"rc": rc, "log": str(bench_log)}
            if rc != 0 and strict:
                raise RuntimeError("profile_kernel benchmark failed")
            if rc != 0:
                profile_kernel_bench_ok = False

            if rc == 0:
                rows = convert_profile_kernel_log_to_rows(
                    log_path=bench_log,
                    repo_root=REPO_ROOT,
                    mode="ours_cpp",
                    kernel="fa2_profile_kernel",
                    dtype=args.dtype,
                    baseline="na",
                    input_mode=args.input_mode,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    run_id=run_id,
                )
                ours_csv = tables_dir / "ours_cpp.csv"
                write_rows(rows, ours_csv)
                summary["tasks"]["profile_kernel_benchmark"]["csv"] = str(ours_csv)
                if rows:
                    csv_parts.append(ours_csv)

    # 4) ncu
    if args.run_ncu == 1:
        if args.run_profile_benchmark == 1 and not profile_kernel_bench_ok:
            summary["tasks"]["ncu"] = {"skipped": True, "reason": "profile_kernel benchmark failed"}
            # Skip NCU to avoid hanging on invalid runtime state/device errors.
            args.run_ncu = 0
    if args.run_ncu == 1:
        exe = (REPO_ROOT / args.build_dir / "profile_kernel").resolve()
        if not exe.exists():
            ok = _ensure_profile_kernel(args, env, run_dir, strict)
            if not ok and strict:
                raise RuntimeError("profile_kernel build required for ncu failed")
        if exe.exists():
            ncu_rep = artifacts_dir / f"{run_id}.ncu-rep"
            b, h, n, d = _parse_shape(args.profile_shape)
            shape = f"{b}x{h}x{n}x{d}"
            ncu_cmd: list[str] = []
            if args.ncu_use_sudo == 1:
                ncu_cmd.append("sudo")
            ncu_cmd.extend([args.ncu_cmd, "--set", args.ncu_set, "--export", str(ncu_rep), "--force-overwrite"])
            if args.ncu_launch_skip:
                ncu_cmd.extend(["--launch-skip", args.ncu_launch_skip])
            if args.ncu_launch_count:
                ncu_cmd.extend(["--launch-count", args.ncu_launch_count])
            ncu_cmd.extend([str(exe), "--mode", "ncu", shape])
            rc = _run(ncu_cmd, cwd=exe.parent, env=env, log_path=logs_dir / "ncu.log")
            summary["tasks"]["ncu"] = {"rc": rc, "report": str(ncu_rep)}
            if rc != 0 and strict:
                raise RuntimeError("ncu failed")

    # Final merge
    results_csv = run_dir / "results.csv"
    n_rows = merge_csvs(csv_parts, results_csv, run_id=run_id)
    if n_rows > 0:
        rows = read_rows(results_csv)
        rows = recompute_rel_perf_pct(rows, baseline=args.baseline)
        write_rows(rows, results_csv)
    summary["results_csv"] = str(results_csv)
    summary["rows"] = n_rows

    result_json = run_dir / "result.json"
    result_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.run_info:
        Path(args.run_info).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"RUN_ID={run_id}")
    print(f"RUN_DIR={run_dir}")
    print(f"RESULTS_CSV={results_csv}")
    print(f"ROWS={n_rows}")


if __name__ == "__main__":
    main()

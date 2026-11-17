#!/usr/bin/env python3
"""Run identical Kyon problems sequentially and check backend parity.

The binary runs one warmup before the requested measured solves. Raw JSON,
iteration traces, trajectories, logs, and a compact summary are preserved.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


def trajectory(path):
    with path.open() as stream:
        rows = list(csv.reader(stream))[1:]
    return [float(value) for row in rows for value in row]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--cpu-binary", type=Path,
                        help="Optional separately compiled best CPU baseline")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--nodes", nargs="+", type=int, default=[30, 256, 1024])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--self-collision", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"cases": [], "parity": [], "gpu_before": None, "gpu_after": None}

    def gpu_state():
        try:
            return subprocess.check_output([
                "nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used",
                "--format=csv"], text=True).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None

    summary["gpu_before"] = gpu_state()
    for nodes in args.nodes:
        reference = None
        cases = [("scalar", args.binary), ("kokkos", args.binary), ("device", args.binary)]
        if args.cpu_binary:
            cases.append(("cpu_reference", args.cpu_binary))
        for label, binary in cases:
            prefix = args.output / f"n{nodes}_{label}"
            backend = "scalar" if label == "cpu_reference" else label
            command = [str(binary.resolve()), "--nodes", str(nodes), "--samples", str(args.samples),
                       "--iterations", str(args.iterations), "--threads", str(args.threads),
                       "--backend", backend, "--self-collision", str(args.self_collision),
                       "--output", str(prefix.resolve())]
            print(f"Running N={nodes}, {label}", flush=True)
            with prefix.with_suffix(".log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                               env={**os.environ, "OMP_PROC_BIND": "false"}, check=True)
            report = json.loads(prefix.with_suffix(".json").read_text())
            samples = report["samples"]
            if not report["valid"] or not all(s["solver_converged"] for s in samples):
                raise RuntimeError(f"Nonconverged or invalid case: {prefix}")
            times = sorted(s["solve_ms"] for s in samples)
            case = {"nodes": nodes, "backend": label, "samples": len(times),
                    "median_ms": statistics.median(times),
                    "p95_ms": times[math.ceil(.95 * len(times)) - 1],
                    "objective": samples[0]["objective"],
                    "iterations": sorted({s["iterations"] for s in samples})}
            summary["cases"].append(case)
            values = trajectory(prefix.with_suffix(".csv"))
            if not all(math.isfinite(v) for v in values):
                raise RuntimeError(f"Nonfinite trajectory: {prefix}")
            if reference is None:
                reference = (values, samples[0]["objective"])
            else:
                if len(values) != len(reference[0]):
                    raise RuntimeError("Backend trajectory shape mismatch")
                error = max(abs(a - b) for a, b in zip(values, reference[0]))
                objective_error = max(abs(s["objective"] - reference[1]) for s in samples)
                summary["parity"].append({"nodes": nodes, "backend": label,
                                          "trajectory_max_abs": error,
                                          "objective_max_abs": objective_error})
                if error > 1e-6 or objective_error > 1e-10 + 1e-8 * abs(reference[1]):
                    raise RuntimeError(f"Backend parity failed: {prefix}")
            print(json.dumps(case), flush=True)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["gpu_after"] = gpu_state()
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Fixed-feet Kyon iLQR comparison, profiling dashboard, and optional replay.

Run ``python -m horizon.examples.kyon_collision --help``. The native executable
owns optimization; Python handles reporting and post-solve visualization.
"""
import argparse
import html
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import platform

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COLORS = {"on": "#008577", "off": "#d97924"}


def find_binary(args):
    candidates = [args.binary] if args.binary else []
    if args.build_dir:
        candidates += [args.build_dir / "kyon_collision_example",
                       args.build_dir / "tests/benchmarkKyonTrajectory"]
    found = shutil.which("kyon_collision_example")
    if found:
        candidates.append(Path(found))
    candidates.append(ROOT / "build/kyon_collision_example")
    for path in candidates:
        if path and path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    raise ValueError("Provide --binary or --build-dir; build the kyon_collision_example target first.")


def sphere_model():
    path = ROOT / "horizon/cpp/tests/data/kyon.spheres.json"
    data = json.loads(path.read_text())
    return [(link["name"], np.asarray(s["center"]), s["radius"])
            for link in data["links"] for s in link["spheres"]]


def prepare_replay(case, urdf):
    """Post-solve FK and exact sampled-SDF diagnostics; outside all solve timers."""
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(urdf)); data = model.createData()
    raw = np.genfromtxt(case / "trajectory.csv", delimiter=",", names=True)
    q = np.column_stack([raw[f"q{i}"] for i in range(model.nq)])
    info = json.loads((case / "profile.json").read_text())
    geometry = sphere_model(); positions = np.empty((len(q), len(geometry), 3))
    tool = np.empty((len(q), 3)); base = np.empty((len(q), 7)); feet = None
    for k, state in enumerate(q):
        pin.forwardKinematics(model, data, state); pin.updateFramePlacements(model, data)
        for i, (name, center, _) in enumerate(geometry):
            positions[k, i] = data.oMf[model.getFrameId(name)].act(center)
        tool[k] = data.oMf[model.getFrameId("dagana_1_base")].translation
        pose = data.oMf[model.getFrameId("pelvis")]
        base[k, :3] = pose.translation; base[k, 3:] = pin.Quaternion(pose.rotation).coeffs()
        if feet is None:
            feet = np.array([data.oMf[model.getFrameId(f"contact_{i}")].translation.copy() for i in range(1, 5)])
    origin = np.asarray(info["sdf_origin"]); voxel = info["voxel_size"]
    cell = (positions - origin) / voxel
    if np.any(cell < 0) or np.any(cell > np.asarray(info["sdf_dimensions"]) - 1):
        raise ValueError("Exported trajectory leaves the SDF domain")
    low = np.minimum(np.floor(cell).astype(int), np.asarray(info["sdf_dimensions"]) - 2)
    fraction = cell - low; distance = np.zeros(positions.shape[:-1])
    obstacle = np.asarray(info["obstacle_center"])
    for z in range(2):
        for y in range(2):
            for x in range(2):
                corner = origin + voxel * (low + np.array([x, y, z]))
                # Match the native SDF's float sample storage and double interpolation.
                value = (np.linalg.norm(corner - obstacle, axis=-1) - info["obstacle_radius"]).astype(np.float32)
                w = np.where(np.array([x, y, z]), fraction, 1 - fraction).prod(-1)
                distance += value * w
    radii = np.array([s[2] for s in geometry])
    env_clearance = (distance - radii).min(axis=1)
    pairs = [(i, j) for i, a in enumerate(geometry) for j, b in enumerate(geometry)
             if a[0].startswith("dagana_1") and b[0].startswith("dagana_2")]
    self_clearance = np.array([min(np.linalg.norm(p[i] - p[j]) - radii[i] - radii[j]
                                   for i, j in pairs) for p in positions])
    np.savez_compressed(case / "replay.npz", time=raw["time"], q=q, centers=positions,
                        tool=tool, base=base, feet=feet, radii=radii,
                        environment_clearance=env_clearance, self_clearance=self_clearance)


def run_case(args, binary, mode):
    directory = args.output / mode; directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / "trajectory"
    prefix.with_suffix(".json").unlink(missing_ok=True)
    for stale in directory.glob("trajectory_iterations_*.csv"):
        stale.unlink()
    command = [str(binary), "--urdf", str(args.urdf), "--nodes", str(args.nodes),
               "--scenario", args.scenario, "--backend", args.backend,
               "--collision", str(int(mode == "on")), "--self-collision", str(int(args.self_collision)),
               "--samples", str(args.samples), "--warmup", str(args.warmup),
               "--iterations", str(args.iterations), "--threads", str(args.threads),
               "--damping", str(args.damping),
               "--kokkos-threads", str(args.kokkos_threads), "--output", str(prefix)]
    if args.collision_weight is not None:
        command += ["--collision-weight",str(args.collision_weight)]
    wrapped = command
    if args.profiler in ("callgrind", "memcheck"):
        valgrind = shutil.which("valgrind")
        if not valgrind:
            raise ValueError("Install Valgrind to use --profiler " + args.profiler)
        options = (["--tool=callgrind", f"--callgrind-out-file={directory}/callgrind.out.%p"]
                   if args.profiler == "callgrind" else
                   ["--tool=memcheck", "--leak-check=full", "--errors-for-leak-kinds=definite"])
        wrapped = [valgrind, *options, "--error-exitcode=99", f"--log-file={directory}/instrumentation.log", *command]
    elif args.profiler == "cuda-memcheck":
        sanitizer = args.sanitizer or shutil.which("compute-sanitizer")
        if not sanitizer:
            raise ValueError("Select a toolkit containing compute-sanitizer, or pass --sanitizer")
        wrapped = [str(sanitizer), "--tool", "memcheck", "--error-exitcode", "99", *command]
    gnu_time = Path("/usr/bin/time")
    if gnu_time.exists() and os.name == "posix":
        wrapped = [str(gnu_time), "-v", "-o", str(directory / "process_counters.txt"), *wrapped]
    print(f"Solving collision {mode}: N={args.nodes}, {args.backend}", flush=True)
    # Keep inner Eigen/BLAS parallelism explicit; --threads controls solver workers.
    environment = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment.setdefault(key, "1")
    environment.setdefault("OMP_PROC_BIND", "false")
    begin = time.perf_counter()
    with (directory / "run.log").open("w") as log:
        result = subprocess.run(wrapped, stdout=log, stderr=subprocess.STDOUT,
                                env=environment)
    elapsed = time.perf_counter() - begin
    native = prefix.with_suffix(".json")
    if not native.exists():
        raise RuntimeError(f"Native run failed ({result.returncode}); see {directory / 'run.log'}")
    info = json.loads(native.read_text())
    if not info.get("profiling_enabled"):
        raise ValueError("Build the native example with -DHORIZON_PROFILING=ON for complete profiling")
    native.rename(directory / "profile.json")
    prepare_replay(directory, args.urdf)
    samples = info["samples"]; times = sorted(s["solve_ms"] for s in samples)
    return {"mode": mode, "returncode": result.returncode, "valid": info["valid"],
            "process_wall_s": elapsed, "median_ms": statistics.median(times),
            "p95_ms": times[int(np.ceil(.95 * len(times))) - 1], "command": command,
            "native_environment": {key: environment[key] for key in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_PROC_BIND")},
            "profile": info, "directory": mode}


def write_dashboard(output, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    cases = manifest["cases"]
    phase_names = ["linearize_quadratize", "backward_pass", "line_search"]
    phase_colors = ["#2878a1", "#7256a4", "#e5ac42", "#bcc7cf"]
    cards = []; details = []
    for i, case in enumerate(cases):
        mode = case["mode"]; color = COLORS[mode]; profile = case["profile"]; samples = profile["samples"]
        values = [s["solve_ms"] for s in samples]
        axes[0, 0].bar(i, case["median_ms"], color=color, width=.55)
        axes[0, 0].scatter(np.full(len(values), i), values, color="#173245", s=16, zorder=3)
        offset = 0
        phases = [statistics.median(s.get("phase_us", {}).get(name, 0) / 1000 for s in samples) for name in phase_names]
        phases.append(max(0, case["median_ms"] - sum(phases)))
        for j, value in enumerate(phases):
            axes[0, 1].bar(i, value, bottom=offset, color=phase_colors[j], width=.55,
                           label=["Linearize / quadratize", "Backward", "Line search", "Other"][j] if i == 0 else None)
            offset += value
        traces = sorted((output / mode).glob("trajectory_iterations_*.csv"))
        if traces:
            trace = np.genfromtxt(traces[0], delimiter=",", names=True)
            trace = np.atleast_1d(trace)
            trace = trace[trace["accepted"] > 0]
            axes[1, 0].plot(trace["iteration"], trace["cost"], color=color, label=f"Collision {mode}")
        with np.load(output / mode / "replay.npz") as replay:
            axes[1, 1].plot(replay["time"], 1000 * replay["environment_clearance"], color=color, label=f"Collision {mode}")
        safe = "pass" if case["valid"] and case["returncode"] == 0 else "fail"
        cards.append(f'<section class="card"><span class="badge {safe}">{safe.upper()}</span><h2>Collision {mode}</h2>'
                     f'<div class="number">{case["median_ms"]:.1f}<small> ms / solve</small></div>'
                     f'<p>p95 {case["p95_ms"]:.1f} ms · {len(samples)} measured solves</p>'
                     f'<p>Target error {max(s["target_error"] for s in samples)*1000:.3f} mm · '
                     f'minimum environment clearance {min(s["min_sdf_clearance"] for s in samples)*1000:.2f} mm</p>'
                     f'<a href="{mode}/profile.json">Complete profile JSON</a> · <a href="{mode}/trajectory.csv">Trajectory CSV</a> · '
                     f'<a href="{mode}/run.log">Run log</a></section>')
        rows = []
        names = sorted(set().union(*(s.get("phase_statistics", {}) for s in samples)))
        for name in names:
            stats = [s["phase_statistics"][name] for s in samples if name in s.get("phase_statistics", {})]
            calls = sum(s["calls"] for s in stats)
            rows.append((statistics.median(s["total_us"] for s in stats),
                         f'<tr><td>{html.escape(name)}</td><td>{statistics.median(s["total_us"] for s in stats)/1000:.3f}</td>'
                         f'<td>{statistics.median(s["calls"] for s in stats):g}</td><td>{sum(s["total_us"] for s in stats)/calls:.3f}</td>'
                         f'<td>{statistics.median(s["p95_us"] for s in stats):.3f}</td></tr>'))
        rows.sort(reverse=True)
        device = {k: statistics.median(s[k] for s in samples) for k in samples[0]
                  if k.startswith(("device_", "batch_")) and isinstance(samples[0][k], (int, float))}
        links = ''.join(f'<li><a href="{mode}/{html.escape(p.name)}">{html.escape(p.name)}</a></li>'
                        for p in sorted((output / mode).iterdir()) if p.is_file())
        details.append(f'<details><summary>Collision {mode}: all timers and artifacts</summary><p>Nested timers overlap; '
                       'worker durations are aggregated CPU work. Do not sum these rows into wall time. '
                       'p95/call is the median of per-solve p95 values.</p><div class="table"><table><thead><tr>'
                       '<th>Timer</th><th>Median ms/solve</th><th>Calls/solve</th><th>Mean µs/call</th><th>p95 µs/call</th>'
                       '</tr></thead><tbody>'+''.join(r[1] for r in rows)+'</tbody></table></div>'
                       '<h3>Batch / device counters and stage totals</h3><pre>'+html.escape(json.dumps(device, indent=2))+
                       '</pre><ul>'+links+'</ul></details>')
    labels = [f'Collision {c["mode"]}' for c in cases]
    for ax in axes[0]:
        ax.set_xticks(range(len(cases)), labels); ax.set_ylabel("Milliseconds"); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    axes[0, 0].set_title("Measured solve time · warmups excluded")
    axes[0, 1].set_title("Top-level solver phases"); axes[0, 1].legend(fontsize=8)
    axes[1, 0].set_title("Accepted costs · first measured solve"); axes[1, 0].set_xlabel("Iteration"); axes[1, 0].set_ylabel("Objective")
    axes[1, 0].set_yscale("log"); axes[1, 0].legend(); axes[1, 0].grid(alpha=.2)
    axes[1, 1].set_title("Environment clearance along trajectory"); axes[1, 1].axhline(0, color="#c23c3c", linestyle="--", linewidth=1)
    axes[1, 1].set_xlabel("Time [s]"); axes[1, 1].set_ylabel("Minimum sphere clearance [mm]"); axes[1, 1].legend(); axes[1, 1].grid(alpha=.2)
    fig.savefig(output / "profile.png", dpi=170); fig.savefig(output / "profile.svg"); plt.close(fig)
    warning = ("Instrumentation is enabled; these timings include profiler overhead." if manifest["profiler"] != "none"
               else "Solve timers exclude setup, validation, plotting, and visualization.")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kyon · collision-aware trajectory optimization</title><style>
body{margin:0;background:#edf2f5;color:#173245;font:16px/1.55 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:36px 24px}
h1{font-size:36px;line-height:1.15;margin:12px 0}h2{font-size:21px}.eyebrow{color:#008577;font-weight:700;letter-spacing:.12em;font-size:12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}.card,details,.plot{background:white;border-radius:14px;padding:22px;margin:20px 0;box-shadow:0 2px 8px #1732450b}
.number{font-size:40px;font-weight:700}.number small{font-size:15px;font-weight:400}.badge{float:right;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700}.pass{background:#dcf3ea;color:#08694f}.fail{background:#ffe2df;color:#a12020}
a{color:#087b83}img{width:100%;height:auto}summary{cursor:pointer;font-weight:650}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:9px;border-bottom:1px solid #e3eaf0}td:not(:first-child){text-align:right;font-variant-numeric:tabular-nums}pre{overflow:auto;background:#f3f6f8;padding:16px}footer{font-size:13px;color:#5c7381}</style><main>
<div class="eyebrow">HORIZON / KYON / iLQR</div><h1>Collision-aware motion,<br>measured end to end.</h1>'''
    page += f'<p>N={manifest["nodes"]} · {html.escape(manifest["backend"])} · fixed feet · target: dagana_1_base · {html.escape(manifest["scenario"])} scene.</p><p>{warning}</p>'
    page += '<div class="cards">'+''.join(cards)+'</div><div class="plot"><img src="profile.svg" alt="Solve timing, solver phases, cost history, and environment clearance"></div>'
    page += ''.join(details)+'<footer>Kinematic shooting with generalized-velocity controls. Collision penalties are soft and evaluated at knots. '
    page += 'Objectives differ when collision costs are disabled. Process counters include warmups and export; worker timer sums are not wall time. '
    page += '<a href="results.json">Experiment manifest</a></footer></main></html>'
    (output / "report.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", "--nodes", dest="nodes", type=int, default=30)
    parser.add_argument("--collision", choices=["both", "on", "off"], default="both")
    parser.add_argument("--backend", choices=["scalar", "kokkos", "device"], default="scalar")
    parser.add_argument("--scenario", choices=["avoidance", "benchmark"], default="avoidance")
    parser.add_argument("--binary", type=Path); parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/tmp/kyon_collision_example"))
    parser.add_argument("--samples", type=int, default=3); parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=120); parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--kokkos-threads", type=int, default=1)
    parser.add_argument("--damping",type=float,default=.001)
    parser.add_argument("--collision-weight",type=float,help="Override the scene's squared residual weight")
    parser.add_argument("--self-collision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profiler", choices=["none", "callgrind", "memcheck", "cuda-memcheck"], default="none")
    parser.add_argument("--sanitizer", type=Path)
    parser.add_argument("--visualizer", choices=["none", "viser", "ros2"], default="none")
    parser.add_argument("--rviz", action="store_true"); parser.add_argument("--show-spheres", action="store_true")
    parser.add_argument("--port", type=int, default=8080); parser.add_argument("--speed", type=float, default=.5)
    parser.add_argument("--visualization-seconds", type=float, default=0, help="Stop replay after this wall time; 0 waits for Ctrl-C")
    parser.add_argument("--replay", type=Path, help="Replay an existing output directory without optimizing")
    args = parser.parse_args()
    if args.nodes < 4 or args.samples < 1 or args.iterations < 1 or args.warmup < 0 or args.threads < 0 or args.kokkos_threads < 1 or not np.isfinite(args.speed) or args.speed <= 0 or not np.isfinite(args.visualization_seconds) or args.visualization_seconds < 0:
        parser.error("Invalid dimensions, counts, or playback speed")
    if args.rviz and args.visualizer != "ros2":
        parser.error("--rviz requires --visualizer ros2")
    if not np.isfinite(args.damping) or args.damping<0 or (args.collision_weight is not None and (not np.isfinite(args.collision_weight) or args.collision_weight<=0)):
        parser.error("Damping must be finite and nonnegative; collision weight must be finite and positive")
    try:
        if args.replay:
            args.output = args.replay.resolve(); manifest = json.loads((args.output / "results.json").read_text())
            args.urdf = (args.urdf or Path(manifest["urdf"])).resolve()
        else:
            args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
            args.urdf = (args.urdf or Path(os.environ.get("KYON_URDF", str(ROOT.parent / "iit-kyon-ros-pkg/kyon_urdf/urdf/kyon.urdf")))).resolve()
            if not args.urdf.is_file():
                raise ValueError("Provide --urdf /path/to/kyon.urdf")
            binary = find_binary(args)
            manifest = {"nodes": args.nodes, "backend": args.backend, "scenario": args.scenario,
                        "format": "horizon.kyon_collision_example.v1", "profiler": args.profiler,
                        "urdf": str(args.urdf), "platform": platform.platform(), "cpu_count": os.cpu_count(),
                        "python": platform.python_version(), "cases": []}
            if shutil.which("nvidia-smi"):
                probe = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used", "--format=csv"],
                                       capture_output=True, text=True)
                manifest["gpu_snapshot"] = probe.stdout.strip()
            for mode in (["off", "on"] if args.collision == "both" else [args.collision]):
                manifest["cases"].append(run_case(args, binary, mode))
                (args.output / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")
            write_dashboard(args.output, manifest)
        print(f"Report: {args.output / 'report.html'}", flush=True)
        if args.visualizer != "none":
            from horizon.examples.kyon_collision_viewer import visualize
            visualize(args, manifest)
        return 0 if all(c["valid"] and c["returncode"] == 0 for c in manifest["cases"]) else 1
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

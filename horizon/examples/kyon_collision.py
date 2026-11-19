"""Fixed-feet Kyon collision comparison, reports, and optional replay."""
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

# Keep historical helpers importable for callers and reporting tests.
from horizon.examples.kyon_collision_support import (
    ROOT, COLORS, find_binary, sphere_model, prepare_replay, run_case,
    write_dashboard, parse_args)


def main():
    args, parser = parse_args()
    try:
        if args.replay:
            args.output = args.replay.resolve(); manifest = json.loads((args.output / "results.json").read_text())
            args.urdf = (args.urdf or Path(manifest["urdf"])).resolve()
            args.collision_policy = (args.collision_policy or Path(manifest["collision_policy"])).resolve()
        else:
            args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
            args.urdf = (args.urdf or Path(os.environ.get("KYON_URDF", str(ROOT / "horizon/cpp/tests/data/kyon_full_spheres.urdf")))).resolve()
            if args.collision_policy is None:
                if args.urdf != (ROOT / "horizon/cpp/tests/data/kyon_full_spheres.urdf").resolve():
                    raise ValueError("Provide --collision-policy for a custom sphere URDF")
                args.collision_policy = ROOT / "horizon/cpp/tests/data/kyon_full_disabled_collisions.json"
            args.collision_policy = args.collision_policy.resolve()
            if not args.urdf.is_file():
                raise ValueError("Provide --urdf /path/to/kyon.urdf")
            binary = find_binary(args)
            manifest = {"nodes": args.nodes, "backend": args.backend, "scenario": args.scenario,
                        "format": "horizon.kyon_collision_example.v1", "profiler": args.profiler,
                        "urdf": str(args.urdf), "collision_policy": str(args.collision_policy), "platform": platform.platform(), "cpu_count": os.cpu_count(),
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

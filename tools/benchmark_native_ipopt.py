"""Compare the existing native Kyon example with iLQR and CasADi/IPOPT.

Run from the repository root with the Horizon Python environment:
python -m tools.benchmark_native_ipopt --urdf /path/to/kyon.urdf --output /tmp/comparison.json
"""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time

import casadi as cs
import numpy as np

from horizon.examples.ilqr_native_pinocchio import make_problem


def compare(urdf, nodes, samples, threads):
    results = []
    for N in nodes:
        for backend in ("ilqr", "ipopt"):
            runs = []
            for _ in range(samples):
                started = time.perf_counter()
                solver, robot, positions = make_problem(urdf, N, threads, backend)
                setup = time.perf_counter()-started
                started = time.perf_counter()
                success = bool(solver.solve())
                elapsed = time.perf_counter()-started
                q = solver.x_opt; v = solver.u_opt
                points = np.array([positions.evaluate(state).reshape(3, 5).T for state in q.T])
                target = solver.prb.getParameters("tool_target").getValues()[:, -1]
                error = float(np.linalg.norm(points[-1, 4]-target))
                drift = float(np.linalg.norm(points[:, :4]-points[0, :4], axis=-1).max())
                step = robot.step(2./N)
                defect = max(float(np.linalg.norm(step.evaluate(np.r_[q[:, k], v[:, k]])-q[:, k+1]))
                             for k in range(N))
                quaternion = float(np.abs(np.linalg.norm(q[3:7], axis=0)-1).max())
                objective = float(.5*(.05**2*np.sum(v*v)/N+100**2*error**2))
                if not (success and error < .005 and drift < 1e-7 and defect < 2e-7
                        and quaternion < 1e-8 and np.abs(v).max() < 1+1e-7):
                    raise RuntimeError(f"Invalid {backend} solution at N={N}")
                runs.append(dict(setup_s=setup, solve_s=elapsed, objective=objective,
                                 target_error_m=error, foot_drift_m=drift,
                                 ambient_dynamics_defect=defect, quaternion_error=quaternion))
            result = dict(solver=backend, N=N, runs=runs,
                          median_solve_s=statistics.median(run["solve_s"] for run in runs))
            results.append(result)
            print(f"N={N}, {backend}: median {result['median_solve_s']:.6f} s, objective {objective:.12g}")
        if abs(results[-1]["runs"][0]["objective"]-results[-2]["runs"][0]["objective"]) > 1e-9:
            raise RuntimeError("Solver objectives disagree")
    return dict(python=platform.python_version(), casadi=cs.__version__,
                platform=platform.platform(), threads=threads, samples=samples, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--nodes", type=int, nargs="+", default=[12, 30])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.urdf.is_file() or args.samples < 1 or min(args.nodes) < 4 or args.threads < 0:
        parser.error("Require an existing URDF, samples >= 1, nodes >= 4, threads >= 0")
    result = compare(args.urdf, args.nodes, args.samples, args.threads)
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()

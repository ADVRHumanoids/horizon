"""Profile native Kyon IPOPT kernels, map threading, and barrier iterations.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. Kernel timings use the same
initial point in every case; solve timings exclude construction and diagnostics.
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import casadi as cs
import numpy as np

from horizon.examples.ilqr_native_pinocchio import make_problem


def profile(urdf, nodes, samples, kernel_samples):
    cases = []
    for N in nodes:
        for map_threads, strategy in ((10, "monotone"), (1, "monotone"), (1, "adaptive")):
            runs = []
            for _ in range(samples):
                solver, robot, positions = make_problem(urdf, N, 2, "ipopt", map_threads,
                                                        {"ipopt.mu_strategy": strategy})
                started = time.perf_counter()
                success = bool(solver.solve())
                elapsed = time.perf_counter()-started
                stats = solver.solver.stats()
                timings = {key: value for key, value in stats.items()
                           if key.startswith(("t_wall_", "t_proc_", "n_call_"))}
                q = solver.x_opt; v = solver.u_opt
                points = np.array([positions.evaluate(state).reshape(3, 5).T for state in q.T])
                target = solver.prb.getParameters("tool_target").getValues()[:, -1]
                error = float(np.linalg.norm(points[-1, 4]-target))
                drift = float(np.linalg.norm(points[:, :4]-points[0, :4], axis=-1).max())
                step = robot.step(2./N)
                defect = max(float(np.linalg.norm(step.evaluate(np.r_[q[:, k], v[:, k]])-q[:, k+1]))
                             for k in range(N))
                quaternion = float(np.abs(np.linalg.norm(q[3:7], axis=0)-1).max())
                objective = float(.5*(.05**2*np.sum(v*v)/N + 100**2*error**2))
                if not (success and error < .005 and drift < 1e-7 and defect < 2e-7
                        and quaternion < 1e-8 and np.abs(v).max() < 1+1e-7):
                    raise RuntimeError("Invalid solution in profile")
                runs.append(dict(solve_s=elapsed, iterations=stats["iter_count"], timings=timings,
                                 objective=objective, target_error_m=error, foot_drift_m=drift,
                                 dynamics_defect=defect, quaternion_error=quaternion))
            kernels = {}
            for name in ("nlp_f", "nlp_g", "nlp_grad_f", "nlp_jac_g", "nlp_hess_l"):
                function = solver.solver.get_function(name)
                inputs = [solver._getVarList("ig"), solver._getParList()]
                if name == "nlp_hess_l":
                    inputs += [1, cs.DM.zeros(solver.prob_dict["g"].numel())]
                function(*inputs)
                durations = []
                for _ in range(kernel_samples):
                    started = time.perf_counter(); function(*inputs)
                    durations.append(time.perf_counter()-started)
                kernels[name] = dict(median_s=statistics.median(durations),
                                     work_doubles=function.sz_w(),
                                     output_nonzeros=[function.nnz_out(i) for i in range(function.n_out())])
            median = statistics.median(run["solve_s"] for run in runs)
            cases.append(dict(N=N, map_threads=map_threads, barrier=strategy,
                              median_solve_s=median, runs=runs, kernels=kernels))
            print(f"N={N}, map_threads={map_threads}, barrier={strategy}: {median:.6f} s, "
                  f"{runs[-1]['iterations']} iterations")
        objectives = [case["runs"][0]["objective"] for case in cases[-3:]]
        if max(objectives)-min(objectives) > 1e-9:
            raise RuntimeError("Profiling cases have different objective values")
    return dict(python=platform.python_version(), casadi=cs.__version__,
                platform=platform.platform(), samples=samples, kernel_samples=kernel_samples,
                cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--nodes", type=int, nargs="+", default=[12, 30])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--kernel-samples", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.urdf.is_file() or min(args.nodes) < 4 or min(args.samples, args.kernel_samples) < 1:
        parser.error("Require an existing URDF, nodes >= 4, and positive sample counts")
    result = profile(args.urdf, args.nodes, args.samples, args.kernel_samples)
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()

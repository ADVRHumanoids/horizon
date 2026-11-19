#!/usr/bin/env python3
"""Compare globalization strategies in isolated native-example processes.

Example: python tools/benchmark_globalization.py --extension /path/to/pyilqr.so \
    --lessons 11 --nodes 30 --seeds 0 --output /tmp/globalization.json
Seeds 0-9 include an unperturbed run (0); others perturb the state guess in
configuration tangent coordinates and controls, preserving fixed initial bounds.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LESSONS = {"07": "07_whole_body_ik", "09": "09_contact_forward",
           "10": "10_contact_inverse", "11": "11_accelerated_collision"}
VARIANTS = ("legacy", "legacy_damped", "lm_merit", "filter_sqp")


def worker(args):
    import casadi  # Extension requires CasADi's libraries first.
    import numpy as np
    if args.extension:
        name = "horizon.solvers.pyilqr"
        spec = importlib.util.spec_from_file_location(name, args.extension)
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
        sys.modules[name] = extension
    from horizon.solvers import Solver
    from horizon.solvers.ilqr import SolverILQR
    make, solve = Solver.make_solver, SolverILQR.solve
    record = dict(lesson=args.lesson, variant=args.variant, nodes=args.node,
                  seed=args.seed, solver_converged=False, physical_pass=False)
    def make_solver(*pos, **kwargs):
        opts = kwargs.setdefault("opts", {})
        # Match objectives, AL handling, stopping tolerances and fixed control
        # regularization across variants; vary only globalization/state damping.
        opts.update({"ilqr.globalization": "legacy" if args.variant.startswith("legacy") else args.variant,
                     "ilqr.hxx_reg": 10. if args.variant == "legacy_damped" else 0.,
                     "ilqr.hxx_reg_growth_factor": 10.,
                     "ilqr.enable_auglag": True, "ilqr.rho_base": 100.,
                     "ilqr.max_iter": args.iterations, "ilqr.merit_der_threshold": 1e-12,
                     "ilqr.stationarity_threshold": 1e-6,
                     "ilqr.constraint_violation_threshold": 1e-7,
                     "ilqr.defect_norm_threshold": 1e-7,
                     "ilqr.use_kkt_solver": int(args.kkt)})
        solver = make(*pos, **kwargs)
        return solver
    Solver.make_solver = staticmethod(make_solver)
    def solve_record(s):
        if args.seed:
            rng = np.random.default_rng(args.seed)
            qstate = s.prb.getState().getInitialGuess().copy()
            # Native examples attach their configuration StateSpace during lowering.
            space = s.prb._discrete_state_space if hasattr(s.prb, "_discrete_state_space") else None
            if space is None and args.lesson in ("07", "11"):
                from horizon.robot import NativeRobot
                robot = NativeRobot(ROOT / "horizon/cpp/tests/data/kyon_full_spheres.urdf")
                space = robot.configuration_space
            for k in range(1, qstate.shape[1]):
                if space is not None:
                    qstate[:, k] = space.integrate(qstate[:, k], rng.normal(0, .01, space.tangentSize()))
                else:
                    # Contact states contain q followed by tangent velocities.
                    from horizon.robot import NativeRobot
                    robot = NativeRobot(ROOT / "horizon/cpp/tests/data/kyon_full_spheres.urdf")
                    nq, nv = robot.nq, robot.nv
                    qstate[:nq, k] = robot.configuration_space.integrate(qstate[:nq, k], rng.normal(0, .01, nv))
                    qstate[nq:, k] += rng.normal(0, .01, qstate.shape[0]-nq)
            offset = 0
            for var in s.prb.getState().var_list:
                var.setInitialGuess(qstate[offset:offset+var.numel()]); offset += var.numel()
            for var in s.prb.getInput().var_list:
                var.setInitialGuess(var.getInitialGuess()+rng.normal(0, .01, var.getInitialGuess().shape))
        trace = []
        def callback(r):
            trace.append({name: getattr(r, name) for name in (
                "iter", "accepted", "alpha", "rho", "cost", "constraint_violation", "defect_norm",
                "bound_violation", "stationarity", "damping", "predicted_reduction",
                "actual_reduction", "model_agreement", "trial_count", "direction_solves",
                "derivative_evaluations", "value_evaluations", "used_correction",
                "used_restoration", "acceptance_reason")})
            return True
        s.set_iteration_callback(callback)
        start = time.perf_counter()
        record["solver_converged"] = bool(solve(s))
        record["solve_seconds"] = time.perf_counter()-start
        record["trace"] = trace
        latest = s.ilqr.getIterationResult()
        record["final"] = {name: getattr(latest, name) for name in trace[-1]} if trace else {}
        record["iterations"] = latest.iter+1
        # Independent bounds audit, including legacy's otherwise-disabled bounds.
        x, u = s.x_opt, s.u_opt
        bounds = [(x, s.prb.getState().getLowerBounds(None), s.prb.getState().getUpperBounds(None)),
                  (u, s.prb.getInput().getLowerBounds(None), s.prb.getInput().getUpperBounds(None))]
        record["max_bound_violation"] = max(0., *(float(np.max(lo-z)) for z,lo,hi in bounds),
                                            *(float(np.max(z-hi)) for z,lo,hi in bounds))
        record["objective"] = sum(float(np.nansum(np.asarray(v)[:-1])/s.N+np.nansum(np.asarray(v)[-1:]))
                                  for v in s.ilqr.getCostsValues().values())
        return record["solver_converged"]
    def safe_solve(s):
        try:
            return solve_record(s)
        except Exception as error:
            import traceback
            traceback.print_exc()
            record["error"] = repr(error)
            return False
    SolverILQR.solve = safe_solve
    sys.argv = [args.lesson, "--nodes", str(args.node), "--max-iterations", str(args.iterations), "--quiet"]
    try:
        runpy.run_module("horizon.examples.native."+LESSONS[args.lesson], run_name="__main__")
        record["physical_pass"] = True
    except SystemExit as error:
        record["physical_pass"] = error.code in (None, 0)
    except Exception as error:
        record["error"] = repr(error)
    record["success"] = (record["physical_pass"] and record["solver_converged"]
                         and record.get("max_bound_violation", float("inf")) <= 1e-7)
    # JSON null for quantities unavailable before stationarity evaluation.
    def finite(value):
        if isinstance(value, dict): return {k: finite(v) for k,v in value.items()}
        if isinstance(value, list): return [finite(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value): return None
        return value
    Path(args.result).write_text(json.dumps(finite(record), indent=2)+"\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--lessons", nargs="+", default=list(LESSONS))
    parser.add_argument("--nodes", nargs="+", type=int, default=[12, 30, 60])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--kkt", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/horizon-globalization.json"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--lesson", choices=LESSONS, help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--node", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker: return worker(args)
    lessons = [str(v).zfill(2) for v in args.lessons]
    if any(v not in LESSONS for v in lessons): parser.error("lessons must be 07, 09, 10, or 11")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = args.output.with_suffix("")
    artifacts.mkdir(exist_ok=True)
    records = []
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env["PYTHONPATH"] = str(ROOT)+os.pathsep+env.get("PYTHONPATH", "")
    for lesson in lessons:
        for nodes in args.nodes:
            for seed in args.seeds:
                for variant in args.variants:
                    stem = f"{lesson}-n{nodes}-s{seed}-{variant}"
                    result = artifacts/(stem+".json")
                    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--lesson", lesson,
                           "--variant", variant, "--node", str(nodes), "--seed", str(seed),
                           "--result", str(result), "--iterations", str(args.iterations)]
                    if args.extension: cmd += ["--extension", str(args.extension.resolve())]
                    if args.kkt: cmd += ["--kkt"]
                    try:
                        with (artifacts/(stem+".log")).open("w") as log:
                            completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                       timeout=args.timeout)
                        record = json.loads(result.read_text()) if completed.returncode == 0 and result.exists() else dict(error="worker_failed")
                    except subprocess.TimeoutExpired:
                        record = dict(error="timeout")
                    log_text=(artifacts/(stem+".log")).read_text()
                    motion=re.search(r"Tool error: ([^;\s]+); foot drift: ([^;\s]+)(?:; self clearance: ([^;\s]+))?", log_text)
                    if motion:
                        record["tool_error"]=float(motion[1]);record["foot_drift"]=float(motion[2])
                        if motion[3]: record["minimum_clearance"]=float(motion[3])
                    backend=re.search(r"Collision execution space: (\S+)",log_text)
                    if backend: record["execution_space"]=backend[1]
                    record.update(lesson=lesson, nodes=nodes, seed=seed, variant=variant)
                    records.append(record)
                    args.output.write_text(json.dumps(dict(extension=str(args.extension), kkt=args.kkt, runs=records), indent=2)+"\n")
                    print(stem, "PASS" if record.get("success") else "FAIL", record.get("error", ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

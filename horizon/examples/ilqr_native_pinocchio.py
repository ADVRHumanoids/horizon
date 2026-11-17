"""Horizon Python variables/costs with direct C++ Pinocchio FK and dynamics.

Build pyilqr with HORIZON_WITH_PINOCCHIO=ON, then run:
python -m horizon.examples.ilqr_native_pinocchio --urdf /path/to/kyon.urdf --N 30
"""
import argparse
import os
from pathlib import Path

import casadi as cs
import numpy as np

from horizon.problem import Problem
from horizon.solvers import Solver, pyilqr
from horizon.solvers.function_manager import SymbolicFunctionManager


def make_problem(urdf, N=30, threads=2, solver_type="ilqr", map_threads=1, ipopt_options=None):
    """Return an iLQR or IPOPT solver with native robot values/Jacobians."""
    import pinocchio as pin
    robot = pyilqr.PinocchioModel(str(urdf))
    # Python Pinocchio is used only to prepare the initial configuration.
    model = pin.buildModelFromUrdf(str(urdf))
    q0 = robot.configuration_space.neutral()
    for leg in range(1, 5):
        sign = 1 if leg % 2 else -1
        for joint, value in ((f"hip_pitch_{leg}", -.7*sign),
                             (f"knee_pitch_{leg}", 1.4*sign),
                             (f"hip_roll_{leg}", .15 if leg in (1, 4) else -.15)):
            q0[model.joints[model.getJointId(joint)].idx_q] = value
    for arm in (1, 2):
        sign = 1 if arm == 1 else -1
        for joint, value in ((f"shoulder_pitch_{arm}", 1.3*sign),
                             (f"elbow_pitch_{arm}", 2.5*sign),
                             (f"wrist_pitch_{arm}", .45*sign)):
            q0[model.joints[model.getJointId(joint)].idx_q] = value

    problem = Problem(N, abstract_casadi_type=cs.MX, thread_map_num=map_threads)
    q = problem.createStateVariable("q", robot.nq)
    v = problem.createInputVariable("v", robot.nv)
    problem.setDt(2./N)
    q.setBounds(robot.lower_position_limit, robot.upper_position_limit)
    v.setBounds(-np.ones(robot.nv), np.ones(robot.nv))
    problem.setInitialState(q0)
    q.setInitialGuess(q0)
    v.setInitialGuess(np.zeros(robot.nv))
    effort_scale = 1/np.sqrt(2*N) if solver_type == "ipopt" else 1
    problem.createIntermediateResidual("effort", effort_scale*.05*v)
    target = problem.createParameter("tool_target", 3)

    functions = SymbolicFunctionManager()
    dt = cs.MX.sym("dt")
    step = functions.addNative("integrate", robot.step(1.))
    functions.addExpression("dynamics", [q, v, dt],
                            step(cs.vertcat(q, dt*v)), exact_hessian=False)
    frames = [f"contact_{i}" for i in range(1, 5)] + ["dagana_1_base"]
    positions = robot.framePositions(frames)
    initial = positions.evaluate(q0).reshape(3, len(frames)).T
    fk = functions.addNative("robot_points", positions)
    points = cs.reshape(fk(q), len(frames), 3).T
    functions.addExpression("feet", [q, v],
                            cs.vertcat(*[points[:, i] - initial[i] for i in range(4)]),
                            exact_hessian=False)
    functions.addResidual("target", [q, v, target], 100*(points[:, 4] - target))
    target.assign(initial[4] + [.04, .02, .01])

    if solver_type == "ipopt":
        # Match iLQR's half squared residuals and running 1/N normalization.
        feet = functions.asCasadi("feet", robot.nq, robot.nv)
        residual = functions.asCasadi("target", robot.nq, robot.nv)
        # Initial feet are already fixed by the initial configuration bounds.
        problem.createConstraint("native_feet", feet(q, cs.MX.zeros(robot.nv)), nodes=range(1, N+1))
        problem.createFinalResidual("native_target",
            residual(q, cs.MX.zeros(robot.nv), target)/np.sqrt(2))
        options = {"ipopt.max_iter": 200, "ipopt.tol": 1e-11, "ipopt.print_level": 0,
                   "ipopt.mu_strategy": "adaptive", "ipopt.mu_min": 1e-13, "print_time": False}
        options.update(ipopt_options or {})
        solver = Solver.make_solver("ipopt", problem, options, gauss_newton=True,
            function_manager=functions, dynamics="dynamics")
    elif solver_type == "ilqr":
        solver = Solver.make_solver("ilqr", problem,
                                   {"ilqr.max_iter": 80, "ilqr.n_threads": threads,
                                    "ilqr.enable_gn": True, "ilqr.merit_der_threshold": 1e-12, "ilqr.huu_reg": .001/N,
                                    "ilqr.constr_decomp_type": "svd",
                                    "ilqr.constraint_violation_threshold": 1e-7,
                                    "ilqr.defect_norm_threshold": 1e-7},
                                   function_manager=functions, dynamics="dynamics",
                                   state_space=robot.configuration_space)
        solver.ilqr.setConstraint(list(range(N+1)), functions, "feet")
        solver.ilqr.setResidual([N], functions, "target")
        solver.set_iteration_callback(lambda result: True)
    else:
        raise ValueError("solver_type must be ilqr or ipopt")
    return solver, robot, positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=os.environ.get("KYON_URDF"))
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--solver", choices=("ilqr", "ipopt"), default="ilqr")
    parser.add_argument("--map-threads", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not args.urdf or not args.urdf.is_file() or args.N < 4 or args.threads < 0 or args.map_threads < 1:
        parser.error("Provide an existing --urdf (or KYON_URDF), N >= 4, and threads >= 0")
    if not hasattr(pyilqr, "PinocchioModel"):
        parser.error("Rebuild pyilqr with HORIZON_WITH_PINOCCHIO=ON")
    solver, robot, positions = make_problem(args.urdf, args.N, args.threads, args.solver, args.map_threads)
    converged = solver.solve()
    solution = solver.getSolutionDict()
    points = np.array([positions.evaluate(q).reshape(3, 5).T for q in solution["q"].T])
    drift = np.linalg.norm(points[:, :4] - points[0, :4], axis=-1).max()
    target = solver.prb.getParameters("tool_target").getValues()[:, -1]
    error = np.linalg.norm(points[-1, 4] - target)
    print(f"Converged: {converged}; target error: {error:.3g} m; foot drift: {drift:.3g} m")
    if args.solver == "ilqr":
        solver.print_timings()
    return 0 if converged and error < .005 and drift < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())

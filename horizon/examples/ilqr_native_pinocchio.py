"""Horizon Python variables/costs with direct C++ Pinocchio FK and dynamics.

Build pyilqr with HORIZON_WITH_PINOCCHIO=ON, then run:
python -m horizon.examples.ilqr_native_pinocchio --urdf /path/to/kyon.urdf --N 30
"""
import casadi as cs
import numpy as np

from horizon.problem import Problem
from horizon.solvers import Solver
from horizon.robot import NativeRobot
from horizon.examples.native_pinocchio_support import (
    initial_configuration, parse_args, solver_options, validate_motion)


def make_problem(urdf, N=30, threads=2, solver_type="ilqr", map_threads=1, ipopt_options=None):
    """Return an iLQR or IPOPT solver with native robot values/Jacobians."""
    robot = NativeRobot(urdf)
    q0 = initial_configuration(urdf, robot)

    problem = Problem(N, abstract_casadi_type=cs.MX, thread_map_num=map_threads)
    q = problem.createStateVariable("q", robot.nq)
    v = problem.createInputVariable("v", robot.nv)
    problem.setDt(2./N)
    q.setBounds(robot.lower_position_limit, robot.upper_position_limit)
    v.setBounds(-np.ones(robot.nv), np.ones(robot.nv))
    problem.setInitialState(q0)
    q.setInitialGuess(q0)
    v.setInitialGuess(np.zeros(robot.nv))
    target = problem.createParameter("tool_target", 3)
    problem.setDiscreteDynamics(robot.integrate(q, problem.getDt()*v))
    # Problem residuals are squared norms summed over their active nodes.
    problem.createIntermediateResidual("effort", .05*v/np.sqrt(2*N))
    frames = [f"contact_{i}" for i in range(1, 5)] + ["dagana_1_base"]
    initial = robot.framePositions(frames, q0).T
    points = robot.framePositions(frames, q)
    problem.createConstraint("feet",
        cs.vertcat(*[points[:, i] - initial[i] for i in range(4)]), nodes=range(1, N+1))
    problem.createFinalResidual("target", 100*(points[:, 4] - target)/np.sqrt(2))
    target.assign(initial[4] + [.04, .02, .01])
    solver = Solver.make_solver(solver_type, problem,
        solver_options(solver_type, N, threads, ipopt_options))
    positions = robot.raw.framePositions(frames)
    return solver, robot.raw, positions


def main():
    args = parse_args()
    solver, robot, positions = make_problem(args.urdf, args.N, args.threads, args.solver, args.map_threads)
    converged = solver.solve()
    error, drift = validate_motion(solver, positions)
    print(f"Converged: {converged}; target error: {error:.3g} m; foot drift: {drift:.3g} m")
    if args.solver == "ilqr":
        solver.print_timings()
    return 0 if converged and error < .005 and drift < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())

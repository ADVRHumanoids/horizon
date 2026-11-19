"""Replace symbolic kinematics with a native Pinocchio frame position.

Native frame evaluation and configuration integration can live inside an
MX-based problem.
"""

import casadi as cs
import numpy as np

from horizon.examples.native.support import parse_options, configure_logging
from horizon.problem import Problem
from horizon.robot import NativeRobot
from horizon.solvers import Solver


def main():
    # Choose the horizon, solver settings and optional asset paths.
    options = parse_options(nodes=30, iterations=50, dt=.1, arm=True)
    urdf = options.urdf
    robot = NativeRobot(urdf)
    n = options.nodes

    # Define the states, controls and discrete dynamics.
    problem = Problem(N=n, abstract_casadi_type=cs.MX)
    problem.setDt(options.dt)
    q = problem.createStateVariable("q", robot.nq)
    u = problem.createInputVariable("u", robot.nv)
    problem.setDiscreteDynamics(robot.step(q, u, problem.getDt()))
    problem.setInitialState(np.zeros(robot.nq))
    target = problem.createParameter("target", 3)

    # Use a known reachable joint configuration to generate the Cartesian target.
    destination = robot.framePositions(["tool"], np.array([0.5, 1.0])).reshape(3)
    target.assign(destination)

    # Add the running objective and task-specific terms.
    problem.createIntermediateResidual("effort", 0.1 * u)
    problem.createFinalResidual("tracking", 10 * (robot.framePositions(["tool"], q)[:, 0] - target))

    # Configure iLQR and its numerical tolerances.
    solver = Solver.make_solver(type="ilqr", prb=problem, opts={"ilqr.max_iter": options.max_iterations, "ilqr.n_threads": options.threads})

    # Solve; iLQR prints iteration progress by default.
    configure_logging(solver, options)
    solver.solve()

    # Check that the computed motion achieved the task.
    final = solver.getSolutionDict()["q"][:, -1]
    error = np.linalg.norm(robot.framePositions(["tool"], final).reshape(3) - destination)
    print(f"Tool error: {error:g}")
    if options.visualize:
        from horizon.examples.native.visualization import visualize_robot
        visualize_robot(robot, urdf, solver.getSolutionDict()["q"], options.dt, options, target=destination)
    return 0 if error < 0.01 else 1


if __name__ == "__main__":
    raise SystemExit(main())

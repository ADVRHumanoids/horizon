"""Add a self-collision clearance penalty to the arm reaching problem.

Sphere-pair clearances enter as a soft residual rather than a hard safety
constraint.
"""

import casadi as cs
import numpy as np

from horizon.collision import CollisionSpheres, self_collision
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
    destination = robot.framePositions(["tool"], np.array([0.5, 0.5])).reshape(3)
    target.assign(destination)

    # Add the running objective and task-specific terms.
    problem.createIntermediateResidual("effort", 0.1 * u)

    # Describe the robot with link-local collision spheres.
    geometry = CollisionSpheres([("base", [0, 0, 0], 0.05), ("tool", [0, 0, 0], 0.05)])
    clearance = self_collision(robot, q, geometry)

    # A soft hinge cost penalizes clearances below the margin.
    problem.createResidual("collision", 100 * cs.fmax(1.9 - clearance, 0))
    problem.createFinalResidual("tracking", 10 * (robot.framePositions(["tool"], q)[:, 0] - target))

    # Configure iLQR and its numerical tolerances.
    solver = Solver.make_solver(type="ilqr", prb=problem, opts={"ilqr.max_iter": options.max_iterations, "ilqr.n_threads": options.threads})

    # Solve; iLQR prints iteration progress by default.
    configure_logging(solver, options)
    solver.solve()

    # Check that the computed motion achieved the task.
    final = solver.getSolutionDict()["q"][:, -1]
    error = np.linalg.norm(robot.framePositions(["tool"], final).reshape(3) - destination)
    distance = min(self_collision(robot, state, geometry)[0]
                   for state in solver.getSolutionDict()["q"].T)
    print(f"Tool error: {error:g}; minimum knot clearance: {distance:g}")
    if options.visualize:
        from horizon.examples.native.visualization import visualize_robot
        visualize_robot(robot, urdf, solver.getSolutionDict()["q"], options.dt, options, target=destination, collision_spheres=geometry)
    return 0 if distance > 1.895 and error < .1 else 1


if __name__ == "__main__":
    raise SystemExit(main())

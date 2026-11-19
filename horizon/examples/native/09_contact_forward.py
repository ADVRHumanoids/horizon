"""Use joint torques as controls with native constrained forward dynamics.

Fixed contacts start at static equilibrium while joint torques drive the
motion.
"""

import casadi as cs
import numpy as np

from horizon.examples.native.support import (
    parse_options,
    configure_logging,
    standing_configuration,
    contact_equilibrium,
)
from horizon.problem import Problem
from horizon.robot import NativeRobot
from horizon.solvers import Solver


def main():
    # Choose the horizon, solver settings and optional asset paths.
    options = parse_options(12, 20, .05, arm=False)
    urdf = options.urdf
    robot = NativeRobot(urdf)
    q0 = standing_configuration(robot, collision_free=True)
    contacts = [f"contact_{i}" for i in range(1, 5)]

    # The first six generalized velocities describe an unactuated floating base.
    selection = np.eye(robot.nv)[:, 6:]
    equilibrium = contact_equilibrium(robot, q0, contacts, selection)
    tau0 = equilibrium[:selection.shape[1]]
    n, dt = options.nodes, options.dt

    # Define the states, controls and discrete dynamics.
    problem = Problem(n, abstract_casadi_type=cs.MX)
    problem.setDt(dt)
    q = problem.createStateVariable("q", robot.nq)
    v = problem.createStateVariable("v", robot.nv)
    tau = problem.createInputVariable("tau", selection.shape[1])
    problem.setInitialState(np.r_[q0, np.zeros(robot.nv)])
    q.setInitialGuess(q0)
    v.setInitialGuess(np.zeros(robot.nv))
    tau.setInitialGuess(tau0)
    acceleration, forces = robot.contactDynamics(q, v, tau, contacts, actuation_matrix=selection)

    # Integrate configuration on its manifold and velocity in tangent space.
    problem.setDiscreteDynamics(cs.vertcat(robot.integrate(q, dt*v + .5*dt**2*acceleration),
                                          v + dt*acceleration))

    # Add the running objective and task-specific terms.
    problem.createIntermediateResidual("velocity", v)
    problem.createIntermediateResidual("torque", .01*(tau-tau0))
    joint = robot.joint_iq("shoulder_pitch_1")
    problem.createFinalResidual("terminal_position", 50*(q[joint] - q0[joint] - .02))
    problem.createFinalResidual("terminal_velocity", 100*v)

    # Configure iLQR and its numerical tolerances.
    solver = Solver.make_solver("ilqr", problem, {"ilqr.max_iter": options.max_iterations, "ilqr.n_threads": options.threads,
        "ilqr.enable_gn": True, "ilqr.huu_reg": 1e-4})

    # Solve; iLQR prints iteration progress by default.
    configure_logging(solver, options)
    solver.solve()

    # Check the physical residuals and resulting motion.
    solution = solver.getSolutionDict()
    q1, v1, tau1 = solution["q"][:, -1], solution["v"][:, -1], solution["tau"][:, -1]
    acceleration1, forces1 = robot.contactDynamics(q1, v1, tau1, contacts, actuation_matrix=selection)
    generalized = robot.inverseDynamics(q1, v1, acceleration1, forces1, contacts)
    residual = np.linalg.norm(generalized - selection@tau1)
    displacement = q1[joint] - q0[joint]
    initial_feet = robot.framePositions(contacts, q0)
    drift = max(np.linalg.norm(robot.framePositions(contacts, state)-initial_feet, axis=0).max()
                for state in solution["q"].T)
    print(f"Foot drift: {drift:g}")
    print(f"Joint displacement: {displacement:g}; terminal speed: {np.linalg.norm(v1):g}; dynamics residual: {residual:g}")
    if options.visualize:
        from horizon.examples.native.visualization import visualize_robot
        visualize_robot(robot, urdf, solver.getSolutionDict()["q"], options.dt, options)
    return 0 if displacement > .015 and np.linalg.norm(v1) < 1e-3 and residual < 1e-7 and drift < .001 else 1


if __name__ == "__main__":
    raise SystemExit(main())

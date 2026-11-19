"""Use acceleration and contact forces as controls, constraining floating-base dynamics.

Underactuation and zero contact acceleration are enforced as forces and motion
are optimized.
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
    options = parse_options(nodes=12, iterations=20, dt=.05, arm=False)
    urdf = options.urdf
    robot = NativeRobot(urdf)
    q0 = standing_configuration(robot, collision_free=True)
    contacts = [f"contact_{i}" for i in range(1, 5)]

    # The first six generalized velocities describe an unactuated floating base.
    selection = np.eye(robot.nv)[:, 6:]
    equilibrium = contact_equilibrium(robot=robot, q=q0, contacts=contacts,
                                      selection=selection)
    force0 = equilibrium[selection.shape[1]:]
    n, dt = options.nodes, options.dt

    # Define the states, controls and discrete dynamics.
    problem = Problem(N=n, abstract_casadi_type=cs.MX)
    problem.setDt(dt)
    q = problem.createStateVariable("q", robot.nq)
    v = problem.createStateVariable("v", robot.nv)
    acceleration = problem.createInputVariable("acceleration", robot.nv)
    forces = problem.createInputVariable("forces", len(contacts)*3)
    problem.setInitialState(np.r_[q0, np.zeros(robot.nv)])
    q.setInitialGuess(q0)
    v.setInitialGuess(np.zeros(robot.nv))
    acceleration.setInitialGuess(np.zeros(robot.nv))
    forces.setInitialGuess(force0)
    generalized = robot.inverseDynamics(q, v, acceleration, forces, contacts)
    contact_acceleration = cs.vertcat(*[robot.frameAcceleration(frame, q, v, acceleration)[:3]
                                       for frame in contacts])

    # A floating base has no actuators: its first six generalized forces vanish.
    problem.createConstraint("underactuation", generalized[:6], nodes=range(n))
    problem.createConstraint("contact_acceleration", contact_acceleration, nodes=range(n))

    # Integrate configuration on its manifold and velocity in tangent space.
    problem.setDiscreteDynamics(cs.vertcat(robot.integrate(q, dt*v + .5*dt**2*acceleration),
                                          v + dt*acceleration))

    # Add the running objective and task-specific terms.
    problem.createIntermediateResidual("velocity", v)
    problem.createIntermediateResidual("acceleration", .01*acceleration)
    problem.createIntermediateResidual("forces", .001*(forces-force0))
    joint = robot.joint_iq("shoulder_pitch_1")
    problem.createFinalResidual("terminal_position", 50*(q[joint] - q0[joint] - .02))
    problem.createFinalResidual("terminal_velocity", 100*v)

    # Configure iLQR and its numerical tolerances.
    solver = Solver.make_solver(type="ilqr", prb=problem, opts={"ilqr.max_iter": options.max_iterations, "ilqr.n_threads": options.threads,
        "ilqr.enable_gn": True, "ilqr.huu_reg": 1e-4, "ilqr.constr_decomp_type": "svd",
        "ilqr.constraint_violation_threshold": 1e-7})

    # Solve; iLQR prints iteration progress by default.
    configure_logging(solver, options)
    solver.solve()

    # Check the physical residuals and resulting motion.
    solution = solver.getSolutionDict()
    q1, v1 = solution["q"][:, -1], solution["v"][:, -1]
    residual = max(np.linalg.norm(robot.inverseDynamics(state, velocity, a, f, contacts)[:6])
                   for state, velocity, a, f in zip(solution["q"].T, solution["v"].T,
                                                    solution["acceleration"].T, solution["forces"].T))
    contact_error = max(np.linalg.norm(robot.frameAcceleration(frame, state, velocity, a)[:3])
                        for state, velocity, a in zip(solution["q"].T, solution["v"].T,
                                                       solution["acceleration"].T)
                        for frame in contacts)
    displacement = q1[joint] - q0[joint]
    initial_feet = robot.framePositions(contacts, q0)
    drift = max(np.linalg.norm(robot.framePositions(contacts, state)-initial_feet, axis=0).max()
                for state in solution["q"].T)
    print(f"Foot drift: {drift:g}; contact acceleration: {contact_error:g}")
    print(f"Joint displacement: {displacement:g}; terminal speed: {np.linalg.norm(v1):g}; dynamics residual: {residual:g}")
    if options.visualize:
        from horizon.examples.native.visualization import visualize_robot
        visualize_robot(robot, urdf, solver.getSolutionDict()["q"], options.dt, options)
    return 0 if displacement > .015 and np.linalg.norm(v1) < 1e-3 and residual < 1e-6 and drift < .001 and contact_error < 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())

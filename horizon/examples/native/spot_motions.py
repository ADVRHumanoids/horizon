"""Native floating-base contact planning for the Spot quadruped."""

import casadi as cs
import numpy as np
from pathlib import Path

from horizon.examples.native.support import parse_options, configure_logging
from horizon.examples.native.spot_actions import ACTIONS
from horizon.problem import Problem
from horizon.robot import NativeRobot
from horizon.solvers import Solver


CONTACTS = ("lf_foot", "rf_foot", "lh_foot", "rh_foot")


def _initial_configuration(robot):
    q = robot.configuration_space.neutral()
    for leg in ("lf", "rf", "lh", "rh"):
        q[robot.joint_iq(f"{leg}_haa_joint")] = 0.0
        q[robot.joint_iq(f"{leg}_hfe_joint")] = 0.9
        q[robot.joint_iq(f"{leg}_kfe_joint")] = -1.52
    q[2] = 0.5
    feet = robot.framePositions(CONTACTS, q)
    q[2] -= float(np.min(feet[2]))
    return q


def _barrier(value):
    return cs.if_else(value > 0, 0, value)


def main():
    urdf = Path(__file__).resolve().parents[1] / "urdf" / "spot.urdf"
    options = parse_options(nodes=50, iterations=200, dt=.05, default_urdf=urdf,
                           action_choices=ACTIONS, default_action="jump_up")
    robot = NativeRobot(options.urdf)
    q0 = _initial_configuration(robot)
    n, dt = options.nodes, options.dt

    print('Executing action:', options.action)

    displacement = np.zeros(3)
    quaternion = q0[3:7].copy()
    if options.action in ("jump_forward", "leap"):
        displacement[0] = 2.0
    if options.action == "jump_twist":
        quaternion = np.array([0.0, 0.0, 0.8509035, 0.525322])

    if options.action == "wheelie":
        swing_ranges = [(20, n)]
    elif options.action == "leap":
        swing_ranges = [(15, 35), (30, 45)]
    else:
        swing_ranges = [(20, 30)]
    swing_nodes = [list(range(*nodes)) for nodes in swing_ranges]
    swing_front = swing_nodes[0]
    swing_hind = swing_nodes[-1] if options.action == "leap" else swing_front
    lifted_legs = list(CONTACTS[:2])
    if options.action not in ("wheelie", "jump_on_wall"):
        lifted_legs.extend(CONTACTS[2:])

    problem = Problem(N=n, abstract_casadi_type=cs.MX)
    problem.setDt(dt)
    q = problem.createStateVariable("q", robot.nq)
    velocity = problem.createStateVariable("velocity", robot.nv)
    acceleration = problem.createInputVariable("acceleration", robot.nv)
    force_vars = [problem.createInputVariable(f"force_{frame}", 3) for frame in CONTACTS]
    forces = cs.vertcat(*force_vars)
    problem.setInitialState(np.r_[q0, np.zeros(robot.nv)])
    q.setInitialGuess(q0)
    velocity.setInitialGuess(np.zeros(robot.nv))
    acceleration.setInitialGuess(np.zeros(robot.nv))
    for force in force_vars:
        force.setInitialGuess([0.0, 0.0, 55.0])

    generalized_force = robot.inverseDynamics(q, velocity, acceleration, forces, CONTACTS)
    problem.createConstraint("floating_base_dynamics", generalized_force[:6], nodes=range(n))
    problem.createFinalConstraint("zero_final_velocity", velocity)
    problem.setDiscreteDynamics(cs.vertcat(
        robot.integrate(q, dt*velocity + .5*dt**2*acceleration),
        velocity + dt*acceleration))

    q_final = q0.copy()
    q_final[:3] += displacement
    q_final[3:7] = quaternion
    if options.action not in ("wheelie", "jump_on_wall"):
        terminal_error = robot.difference(q_final, q)
        problem.createFinalConstraint("final_base_pose", terminal_error[:6])
        problem.createFinalResidual("final_joint_pose", 1e3*(q[7:] - q_final[7:]))

    for index, frame in enumerate(CONTACTS):
        force = force_vars[index]
        # Compute the velocity and acceleration of the foot
        foot_velocity = robot.frameVelocity(frame, q, velocity)[:3]
        foot_acceleration = robot.frameAcceleration(frame, q, velocity, acceleration)[:3]
        is_front = frame in CONTACTS[:2]
        nodes_swing = swing_front if is_front else swing_hind
        swing_control_nodes = [node - 1 for node in nodes_swing]
        if frame in lifted_legs:
            state_stance = [node for node in range(1, n + 1) if node not in nodes_swing]
            control_stance = [node for node in range(n) if node not in swing_control_nodes]
        else:
            state_stance = range(1, n + 1)
            control_stance = range(n)
        problem.createConstraint(f"{frame}_contact_velocity", foot_velocity, nodes=state_stance)
        # problem.createConstraint(f"{frame}_contact_acceleration", foot_acceleration, nodes=control_stance)
        problem.createIntermediateResidual(f"{frame}_normal_force", _barrier(force[2] - 25.0))
        if options.action == "leap":
            start_height = robot.framePositions([frame], q0)[2, 0]
            foot_position = robot.framePositions([frame], q)[2, 0]
            problem.createConstraint(f"{frame}_ground_clearance", foot_position, nodes=nodes_swing,
                                     bounds={"lb": [start_height], "ub": [np.inf]})
        if frame in lifted_legs:
            force.setBounds(np.zeros(3), np.zeros(3), nodes=swing_control_nodes)

        if frame in CONTACTS[:2] and options.action == "jump_on_wall":
            wall_start = swing_ranges[0][1]
            angle = -np.pi / 2.0
            rotation = np.array([[np.cos(angle), 0.0, np.sin(angle)],
                                 [0.0, 1.0, 0.0],
                                 [-np.sin(angle), 0.0, np.cos(angle)]])
            friction = np.zeros((5, 3))
            friction[0, 0], friction[0, 2] = 1.0, -np.sqrt(2.0)
            friction[1, 0], friction[1, 2] = -1.0, -np.sqrt(2.0)
            friction[2, 1], friction[2, 2] = 1.0, -np.sqrt(2.0)
            friction[3, 1], friction[3, 2] = -1.0, -np.sqrt(2.0)
            friction[4, 2] = -1.0
            cone = cs.DM(friction @ rotation.T) @ force
            problem.createConstraint(f"{frame}_wall_friction", cone, nodes=range(wall_start - 1, n),
                                     bounds={"lb": [-np.inf]*5, "ub": [0.0]*5})
            goal = robot.framePositions([frame], q0)[:, 0] + np.array([.3, 0.0, .8])
            problem.createFinalConstraint(f"{frame}_wall_target",
                                          robot.framePositions([frame], q)[:, 0] - goal)

    for force in force_vars:
        problem.createIntermediateResidual(f"min_{force.getName()}", np.sqrt(3e-3)*force)
    problem.createIntermediateResidual("min_velocity", velocity)
    problem.createIntermediateResidual("min_acceleration", 1e-3*acceleration)

    solver = Solver.make_solver("ilqr", problem, {
        "ilqr.hxx_reg": 10.0 if options.globalization == "legacy" else 0.0,
        "ilqr.max_iter": options.max_iterations,
        "ilqr.n_threads": options.threads,
        "ilqr.constraint_violation_threshold": 1e-6,
        "ilqr.alpha_min": 0.3
    })

    configure_logging(solver, options)
    solver.solve()
    solution = solver.getSolutionDict()
    final = solution["q"][:, -1]
    print(f"Base displacement: {final[:3] - q0[:3]}")

    solver.print_timings()

    if options.visualize:
        from horizon.examples.native.visualization import visualize_robot
        visualize_robot(robot, options.urdf, solution["q"], dt, options,
                        contact_frames=CONTACTS, tool_frame=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
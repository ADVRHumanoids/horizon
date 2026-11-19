"""Problem-only contact, full-pose target and collision-aware motion model.

Scene construction is separate: use horizon.collision.build_sdf on primitives
and watertight meshes, then pass the immutable SDF to build_problem below.
"""
import casadi as cs
import numpy as np

from horizon.problem import Problem
from horizon.robot import NativeRobot
from horizon.collision import self_collision, environment_collision


def build_problem(urdf, q0, target_pose, geometry, sdf, *, disabled_pairs=(),
                  contacts=("contact_1", "contact_2", "contact_3", "contact_4"),
                  tool="dagana_1_base", N=30, duration=2., margin=.02, minimum_fz=1.):
    """Leading six-DOF floating base, point contacts and joint-torque controls.

    q0 must be a feasible stance. Contact dynamics enforces acceleration-level
    constraints; validate integration drift and between-knot clearance afterward.
    Use IPOPT for the hard force and collision inequalities in this problem.
    """
    robot = NativeRobot(urdf)
    S = np.eye(robot.nv)[:, 6:]
    problem = Problem(N, abstract_casadi_type=cs.MX)
    q = problem.createStateVariable("q", robot.nq)
    v = problem.createStateVariable("v", robot.nv)
    tau = problem.createInputVariable("tau", S.shape[1])
    target = problem.createParameter("tool_target", 7)
    target.assign(target_pose)
    problem.setDt(duration/N)
    q.setBounds(robot.lower_position_limit, robot.upper_position_limit)
    v.setBounds(-robot.velocity_limits, robot.velocity_limits)
    tau.setBounds(-robot.effort_limits[6:], robot.effort_limits[6:])
    problem.setInitialState(np.r_[q0, np.zeros(robot.nv)])
    v.setBounds(np.zeros(robot.nv), np.zeros(robot.nv), nodes=N)
    q.setInitialGuess(q0)
    v.setInitialGuess(np.zeros(robot.nv))
    J = np.vstack([robot.jacobian(frame, q0)[:3] for frame in contacts])
    equilibrium = np.linalg.lstsq(np.column_stack((S, J.T)), robot.gravity(q0), rcond=None)[0]
    tau.setInitialGuess(equilibrium[:S.shape[1]])

    acceleration, force = robot.contactDynamics(q, v, tau, contacts, actuation_matrix=S)
    dt = problem.getDt()
    problem.setDiscreteDynamics(cs.vertcat(
        robot.integrate(q, dt*v + .5*dt**2*acceleration), v + dt*acceleration))
    problem.createIntermediateResidual("effort", .01*tau)
    problem.createIntermediateResidual("velocity", .01*v)
    problem.createFinalResidual("tool_pose", cs.diag(cs.DM([100.,100.,100.,30.,30.,30.]))
                                @ robot.framePoseError(tool, q, target))

    problem.createConstraint("positive_fz", force[2::3], nodes=range(N),
        bounds={"lb": np.full(len(contacts), minimum_fz), "ub": np.full(len(contacts), np.inf)})
    self_clearance = self_collision(robot, q, geometry, disabled_pairs=disabled_pairs)
    if self_clearance.numel():
        problem.createConstraint("self_collision", self_clearance,
            bounds={"lb": np.full(self_clearance.numel(), margin), "ub": np.full(self_clearance.numel(), np.inf)})
    clearance = environment_collision(robot, q, geometry, sdf)
    problem.createConstraint("environment_collision", clearance,
        bounds={"lb": np.full(clearance.numel(), margin), "ub": np.full(clearance.numel(), np.inf)})
    return problem

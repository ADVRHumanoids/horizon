"""Add self-collision clearances to fixed-foot whole-body inverse kinematics.

The robot sphere model and disabled-pair policy define a soft clearance cost.
"""

import casadi as cs
import numpy as np

from horizon.examples.native.support import (
    parse_options,
    configure_logging,
    standing_configuration,
    motion_error,
)
from horizon.collision import CollisionSpheres, self_collision
from horizon.problem import Problem
from horizon.robot import NativeRobot
from horizon.solvers import Solver


def main():
    # Choose the horizon, solver settings and optional asset paths.
    options = parse_options(nodes=12, iterations=80, dt=None, arm=False)
    urdf = options.urdf
    robot = NativeRobot(urdf)
    q0 = standing_configuration(robot, collision_free=True)

    # Load the full robot sphere model and its pair exclusions.
    geometry = CollisionSpheres.from_urdf(urdf)
    policy = options.policy
    n = options.nodes

    # Define the states, controls and discrete dynamics.
    problem = Problem(N=n, abstract_casadi_type=cs.MX)
    problem.setDt(options.dt)
    q = problem.createStateVariable("q", robot.nq)
    v = problem.createInputVariable("v", robot.nv)
    q.setBounds(robot.lower_position_limit, robot.upper_position_limit)
    v.setBounds(-np.ones(robot.nv), np.ones(robot.nv))
    problem.setInitialState(q0)
    q.setInitialGuess(q0)
    problem.setDiscreteDynamics(robot.integrate(q, problem.getDt()*v))
    frames = [f"contact_{i}" for i in range(1, 5)] + ["dagana_1_base"]
    initial = robot.framePositions(frames, q0).T

    # Fix the first four frame positions and track the fifth, the tool.
    points = robot.framePositions(frames, q)
    tool_target = initial[4] + [.04, .02, .01]
    target = problem.createParameter("tool_target", 3)
    target.assign(tool_target)
    problem.createConstraint("feet", cs.vertcat(*[points[:, i]-initial[i] for i in range(4)]),
                             nodes=range(1, n+1))

    # Add the running objective and task-specific terms.
    problem.createIntermediateResidual("effort", .05*v/np.sqrt(2*n))
    clearance = self_collision(robot, q, geometry, disabled_pairs=policy)
    problem.createResidual("self_collision", 100*cs.fmax(.01-clearance, 0))
    problem.createFinalResidual("target", 100*(points[:, 4]-target)/np.sqrt(2))

    # Configure iLQR and its numerical tolerances.
    solver = Solver.make_solver(type="ilqr", prb=problem, opts={"ilqr.max_iter": options.max_iterations, "ilqr.n_threads": options.threads,
        "ilqr.enable_gn": True, "ilqr.merit_der_threshold": 1e-12, "ilqr.huu_reg": .001/n,
        "ilqr.constr_decomp_type": "svd", "ilqr.constraint_violation_threshold": 1e-7,
        "ilqr.defect_norm_threshold": 1e-7})

    # Solve; iLQR prints iteration progress by default.
    configure_logging(solver, options)
    solver.solve()

    # Check tool tracking, foot drift and, when enabled, clearance.
    error, drift = motion_error(robot=robot, frames=frames,
                                trajectory=solver.getSolutionDict()["q"],
                                initial=initial,
                                target=tool_target)
    minimum = min(np.min(self_collision(robot, state, geometry, disabled_pairs=policy))
                  for state in solver.getSolutionDict()["q"].T)
    print(f"Tool error: {error:g}; foot drift: {drift:g}; self clearance: {minimum:g}")
    if options.visualize:
        from horizon.examples.native.visualization import visualize_robot
        visualize_robot(robot, urdf, solver.getSolutionDict()["q"], options.dt, options, collision_spheres=geometry, target=tool_target)
    return 0 if error < .005 and drift < 1e-4 and minimum > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

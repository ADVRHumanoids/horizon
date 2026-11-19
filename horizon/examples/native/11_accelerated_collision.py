"""Replace self-collision residuals with a Kokkos batched device cost.

Collision values and derivatives are batched across shooting nodes on Kokkos's
selected execution space.
"""

import casadi as cs
import numpy as np

from horizon.examples.native.support import (
    parse_options,
    configure_logging,
    standing_configuration,
    motion_error,
)
from horizon.collision import CollisionSpheres, self_collision, self_collision_pairs
from horizon.problem import Problem
from horizon.robot import NativeRobot
from horizon.solvers import Solver, pyilqr


def run(options):
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
    tool_target = initial[4] + [.4, .0, -0.5]
    target = problem.createParameter("tool_target", 3)
    target.assign(tool_target)
    problem.createConstraint("feet", cs.vertcat(*[points[:, i]-initial[i] for i in range(4)]),
                             nodes=range(1, n+1))

    # Add the running objective and task-specific terms.
    problem.createIntermediateResidual("effort", .05*v/np.sqrt(2*n))
    problem.createFinalResidual("target", 100*(points[:, 4]-target)/np.sqrt(2))

    # Configure iLQR and its numerical tolerances.
    solver = Solver.make_solver(type="ilqr", prb=problem, opts={"ilqr.max_iter": options.max_iterations, "ilqr.n_threads": options.threads,
        "ilqr.enable_gn": True, "ilqr.merit_der_threshold": 1e-12, "ilqr.huu_reg": .001/n,
        # The large tool displacement needs damping at the standing pose.
        # Reduce it as steps are accepted so final refinement stays fast.
        "ilqr.hxx_reg": 10.0 if options.globalization == "legacy" else 0.0,
        "ilqr.globalization": options.globalization,
        "ilqr.hxx_reg_growth_factor": 10.0,
        "ilqr.constr_decomp_type": "svd", "ilqr.constraint_violation_threshold": 1e-7,
        "ilqr.defect_norm_threshold": 1e-7,
        "ilqr.enable_auglag": True,
        "ilqr.rho_base": 100.0})
    pairs = self_collision_pairs(geometry, disabled_pairs=policy)

    # This cost batches collision values and derivatives on the Kokkos backend.
    collision = robot.raw.deviceCollisionCost(spheres=geometry.native, pairs=pairs,
                                              weight=10000., margin=.01, smoothing=.001)
    solver.ilqr.setBatchedCost(list(range(n+1)), "self_collision", collision)
    print(f"Collision execution space: {collision.execution_space()}")

    # Solve; iLQR prints iteration progress by default.
    configure_logging(solver, options)
    converged = solver.solve()

    # Check tool tracking, foot drift and, when enabled, clearance.
    error, drift = motion_error(robot, frames, solver.getSolutionDict()["q"], initial, tool_target)
    minimum = min(np.min(self_collision(robot, state, geometry, disabled_pairs=policy))
                  for state in solver.getSolutionDict()["q"].T)
    print(f"Tool error: {error:g}; foot drift: {drift:g}; self clearance: {minimum:g}")
    _, derivative_batches = collision.batch_counts()
    print(f"Collision derivative batches: {derivative_batches}")
    if options.visualize:
        from horizon.examples.native.visualization import visualize_robot
        visualize_robot(robot, urdf, solver.getSolutionDict()["q"], options.dt, options, collision_spheres=geometry, target=tool_target)
    return 0 if converged and error < .005 and drift < 1e-4 and minimum > 0 and derivative_batches > 0 else 1


def main():
    # Keep Kokkos alive until run() has destroyed the solver and its cost clones.
    options = parse_options(nodes=30, iterations=80, dt=None)
    runtime = pyilqr.KokkosRuntime()
    try:
        return run(options)
    finally:
        del runtime


if __name__ == "__main__":
    raise SystemExit(main())

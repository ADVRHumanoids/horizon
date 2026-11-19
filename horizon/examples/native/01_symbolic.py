"""Drive a symbolic differential-drive model to a point.

The model separates discrete dynamics, running effort, and terminal target
costs.
"""
import casadi as cs
import numpy as np

from horizon.examples.native.support import parse_options, configure_logging
from horizon.problem import Problem
from horizon.solvers import Solver


def main():
    # Choose the horizon, solver settings and optional asset paths.
    options = parse_options(nodes=100, iterations=30, dt=.01, arm=False)
    n = options.nodes

    # Define the states, controls and discrete dynamics.
    problem = Problem(N=n, abstract_casadi_type=cs.MX)
    problem.setDt(options.dt)
    x = problem.createStateVariable("x", 3)
    u = problem.createInputVariable("u", 2)

    # State: x, y, heading. Controls: forward speed and yaw rate.
    problem.setDiscreteDynamics(x + options.dt * cs.vertcat(
        u[0] * cs.cos(x[2]), u[0] * cs.sin(x[2]), u[1]))
    problem.setInitialState(np.zeros(3))

    # Linear interpolation of initial and final states as a guess for the solver.
    final_state = np.array([0.1, 1, 0])
    initial_state = np.array([0., 0., 0.001])
    state_guess = np.linspace(initial_state, final_state, n + 1).T
    x.setInitialGuess(state_guess)
    x.setBounds(lb=initial_state, ub=initial_state, nodes=[0])

    # Add the running objective and task-specific terms.
    problem.createIntermediateResidual("effort", 0.1 * u)
    problem.createFinalConstraint("target", 1 * (x - final_state))

    # Configure iLQR and its numerical tolerances.
    solver = Solver.make_solver(
        type="ilqr",
        prb=problem,
        opts={"ilqr.max_iter": options.max_iterations,
              "ilqr.n_threads": options.threads},
    )

    # Solve; iLQR prints iteration progress by default.
    configure_logging(solver, options)
    solver.solve()

    # Check that the computed motion achieved the task.
    final = solver.getSolutionDict()["x"][:, -1]
    print(f"Final pose: {final}")
    if options.visualize:
        from horizon.examples.native.visualization import visualize_drive
        visualize_drive(solver.getSolutionDict()["x"], options.dt, options, target=final_state)
    return 0 if np.linalg.norm(final - final_state) < 0.01 else 1


if __name__ == "__main__":
    raise SystemExit(main())

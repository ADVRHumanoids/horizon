"""Move a tool around a primitive sphere placed on its direct route.

A native sphere distance field supplies the environment clearance.
"""
import casadi as cs
import numpy as np

from horizon.collision import CollisionSpheres, environment_collision
from horizon.examples.native.support import parse_options, configure_logging
from horizon.problem import Problem
from horizon.robot import NativeRobot
from horizon.solvers import Solver, pyilqr


def main():
    # Set up the arm and its discrete joint-velocity dynamics.
    options = parse_options(nodes=30, iterations=50, dt=.1, arm=True)
    robot = NativeRobot(options.urdf)
    n = options.nodes
    problem = Problem(N=n, abstract_casadi_type=cs.MX)
    problem.setDt(options.dt)
    q = problem.createStateVariable("q", robot.nq)
    u = problem.createInputVariable("u", robot.nv)
    problem.setDiscreteDynamics(robot.step(q, u, options.dt))
    problem.setInitialState(np.zeros(robot.nq))
    u.setBounds(-np.ones(robot.nv), np.ones(robot.nv))

    # Seed a feasible bend around the obstacle, avoiding a symmetric collision.
    waypoint = np.array([-.15, 1.15])
    goal = np.array([.5, .5])
    guess = np.column_stack([
        2*t*waypoint if t <= .5 else waypoint + (2*t-1)*(goal-waypoint)
        for t in np.linspace(0., 1., n+1)
    ])
    q.setInitialGuess(guess)
    u.setInitialGuess(np.diff(guess, axis=1) / options.dt)

    # The obstacle lies near the midpoint of the direct joint-space route.
    tool = CollisionSpheres([("tool", [0, 0, 0], .05)])
    obstacle_center = np.array([1.89, .72, 0.])
    obstacle_radius = .10
    obstacle = pyilqr.SphereField(obstacle_center, obstacle_radius)
    clearance = environment_collision(robot, q, tool, obstacle)
    destination = robot.framePositions(["tool"], goal).reshape(3)
    target = problem.createParameter("target", 3)
    target.assign(destination)
    problem.createIntermediateResidual("effort", .1*u)
    problem.createFinalResidual("tracking", 10*(robot.framePositions(["tool"], q)[:, 0] - target))

    # Penalize low clearance at knots and at three points within each interval.
    # These are soft costs; the denser check below is still only sampled.
    problem.createResidual("collision", 100*cs.fmax(.03 - clearance, 0))
    between = cs.vertcat(*[
        environment_collision(robot, robot.integrate(q, options.dt*fraction*u), tool, obstacle)
        for fraction in (.25, .5, .75)
    ])
    problem.createIntermediateResidual("between_knots", 100*cs.fmax(.03 - between, 0))

    # Solve and report the tool's progress and sampled clearance.
    solver = Solver.make_solver(type="ilqr", prb=problem,
        opts={"ilqr.max_iter": options.max_iterations, "ilqr.n_threads": options.threads})
    configure_logging(solver, options)
    solver.solve()
    states = solver.getSolutionDict()["q"]
    final = robot.framePositions(["tool"], states[:, -1]).reshape(3)
    error = np.linalg.norm(final - destination)
    sampled = []
    defects = []
    controls = solver.getSolutionDict()["u"]
    for start, end, velocity in zip(states.T[:-1], states.T[1:], controls.T):
        prediction = robot.integrate(start, options.dt*velocity)
        defects.append(np.linalg.norm(robot.difference(prediction, end)))
        delta = robot.difference(start, end)
        for fraction in np.linspace(0., 1., 11):
            sample = robot.integrate(start, fraction*delta)
            sampled.append(environment_collision(robot, sample, tool, obstacle)[0])
    minimum = min(sampled)
    start_clearance = environment_collision(robot, np.zeros(robot.nq), tool, obstacle)[0]
    goal_clearance = environment_collision(robot, goal, tool, obstacle)[0]
    direct_midpoint = environment_collision(robot, .5*goal, tool, obstacle)[0]
    print(f"Tool error: {error:g}; minimum sampled clearance: {minimum:g}; "
          f"direct-route midpoint clearance: {direct_midpoint:g}")
    defect = max(defects)
    print(f"Start clearance: {start_clearance:g}; target clearance: {goal_clearance:g}; "
          f"maximum dynamics defect: {defect:g}")

    if options.visualize:
        from horizon.examples.native.visualization import visualize_robot
        visualize_robot(robot, options.urdf, states, options.dt, options,
                        target=destination, collision_spheres=tool,
                        obstacle=(obstacle_center, obstacle_radius))
    return 0 if (error < .01 and minimum > .02 and direct_midpoint < 0
                 and min(start_clearance, goal_clearance) > .2 and defect < 1e-5) else 1


if __name__ == "__main__":
    raise SystemExit(main())

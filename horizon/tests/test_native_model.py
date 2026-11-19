"""NativeModel against an independent dense equality-constrained QP."""
import gc
import unittest
from unittest.mock import patch

import casadi as cs
import numpy as np

from horizon.problem import Problem
from horizon.solvers import pyilqr
from horizon.solvers.native_model import NativeModel


def qp_reference(N, dt, target, waypoint):
    size = 2 * N + 1
    H = np.zeros((size, size))
    g = np.zeros(size)
    for k in range(N):
        H[N + 1 + k, N + 1 + k] += .7 ** 2 / N
    for k in (1, 3):
        H[k, k] += .2 ** 2 / N
        g[k] -= .2 ** 2 * target / N
    H[N, N] += .8 ** 2
    g[N] -= .8 ** 2 * target
    A = np.zeros((N + 2, size))
    b = np.zeros(N + 2)
    A[0, 0] = 1
    for k in range(N):
        A[k + 1, k] = -1
        A[k + 1, k + 1] = 1
        A[k + 1, N + 1 + k] = -dt
    A[-1, 2] = 1
    b[-1] = waypoint
    K = np.block([[H, A.T], [A, np.zeros((N + 2, N + 2))]])
    return np.linalg.solve(K, np.r_[-g, b])[:size]


class NativeModelTest(unittest.TestCase):
    def test_backends_nodes_updates_normalization_and_lifetime(self):
        N = 4
        for backend in ("ilqr", "ipopt"):
            with self.subTest(backend=backend):
                problem = Problem(N, abstract_casadi_type=cs.MX, thread_map_num=2)
                x = problem.createStateVariable("x", 1)
                u = problem.createInputVariable("u", 1)
                target = problem.createParameter("goal", 1)
                waypoint = problem.createParameter("waypoint", 1)
                timestep = problem.createParameter("timestep", 1)
                problem.setDt(timestep)
                problem.setInitialState([0])
                x.setInitialGuess(0)
                u.setInitialGuess(0)
                model = NativeModel(problem)
                z = cs.MX.sym("z", 2)
                step = model.addNative("step", pyilqr.CasadiFunction(cs.Function("step", [z], [z[0] + z[1]])))
                model.setDynamics(step(cs.vertcat(x, model.dt * u)))
                model.createIntermediateResidual("effort", .7 * u)
                model.createIntermediateResidual("tracking", .2 * (x - target), nodes=[1, 3])
                model.createFinalResidual("terminal", .8 * (x - target))
                model.createConstraint("waypoint_constraint", x - waypoint, nodes=[2])
                target.assign(.9)
                waypoint.assign(.25)
                timestep.assign(.2)
                options = {"ilqr.n_threads": 2, "ilqr.max_iter": 20,
                           "ilqr.merit_der_threshold": 1e-13,
                           "ilqr.huu_reg": 0., "ipopt.tol": 1e-11,
                           "ipopt.print_level": 0, "print_time": False}
                solver = model.make_solver(backend, options)
                if backend == "ilqr":
                    solver.set_iteration_callback(lambda result: True)
                del model, step
                gc.collect()
                with patch.object(pyilqr.Function, "evaluate", side_effect=AssertionError("Python native evaluation")):
                    for dt, goal, mid in ((.2, .9, .25), (.35, -.4, -.1)):
                        timestep.assign(dt)
                        target.assign(goal)
                        waypoint.assign(mid)
                        self.assertTrue(solver.solve())
                        actual = np.r_[solver.getSolutionState().ravel(), solver.getSolutionInput().ravel()]
                        np.testing.assert_allclose(actual, qp_reference(N, dt, goal, mid), atol=2e-7)
                        expected_cost = (.5 * .7**2 * np.sum(actual[N+1:]**2) / N
                                         + .5 * .2**2 * np.sum((actual[[1, 3]] - goal)**2) / N
                                         + .5 * .8**2 * (actual[N] - goal)**2)
                        if backend == "ilqr":
                            cost = 0.
                            for values in solver.ilqr.getCostsValues().values():
                                values = np.asarray(values).ravel()
                                cost += np.nansum(values[:-1]) / N
                                if np.isfinite(values[-1]):
                                    cost += values[-1]
                            self.assertAlmostEqual(cost, expected_cost, places=8)
                        if backend == "ipopt":
                            self.assertAlmostEqual(float(cs.Function("objective", [solver.prob_dict["x"], solver.prob_dict["p"]], [solver.prob_dict["f"]])(actual, solver._getParList())), expected_cost, places=8)


if __name__ == "__main__":
    unittest.main()

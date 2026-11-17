"""Legacy Horizon and native-manager regressions against the compiled extension."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import casadi as cs
import numpy as np

# CTest selects its own freshly built module instead of a stale package copy.
if os.environ.get("HORIZON_PYILQR_EXTENSION"):
    spec = importlib.util.spec_from_file_location("horizon.solvers.pyilqr", os.environ["HORIZON_PYILQR_EXTENSION"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[spec.name] = module

from horizon.problem import Problem
from horizon.solvers import Solver, pyilqr
from horizon.solvers.function_manager import SymbolicFunctionManager

URDF = Path(os.environ.get("KYON_URDF") or str(Path(__file__).resolve().parents[3] /
            "iit-kyon-ros-pkg/kyon_urdf/urdf/kyon.urdf"))


def reference_qp(N, dt, initial, goal_node, goal, weights, active):
    """Independent equality-constrained dense QP for the legacy DSL test."""
    diagonal = np.zeros(2*N+1)
    for k in range(N):
        diagonal[k] = .04/N if k in active else 0
        diagonal[N+1+k] = (1 + (weights[k]**2 if k in active else 0))/N
    diagonal[N] = .09
    A = np.zeros((N+2, 2*N+1)); b = np.zeros(N+2)
    A[0, 0] = 1; b[0] = initial
    for k in range(N):
        A[k+1, k] = -1; A[k+1, k+1] = 1; A[k+1, N+1+k] = -dt
    A[-1, goal_node] = 1; b[-1] = goal
    K = np.block([[2*np.diag(diagonal), A.T], [A, np.zeros((N+2, N+2))]])
    return np.linalg.solve(K, np.r_[np.zeros(2*N+1), b])[:2*N+1]


class HorizonPythonInterop(unittest.TestCase):
    def test_legacy_parameters_nodes_warm_start_and_exports(self):
        for abstract in (cs.SX, cs.MX):
            for gn, threads in ((False, 0), (True, 2)):
                with self.subTest(abstract=abstract.__name__, gn=gn, threads=threads):
                    N = 8; dt = .25
                    problem = Problem(N, abstract_casadi_type=abstract, receding=True)
                    x = problem.createStateVariable("x", 1)
                    u = problem.createInputVariable("u", 1)
                    w = problem.createParameter("weight", 1)
                    goal = problem.createParameter("goal", 1)
                    weights = np.linspace(.8, 1.2, N+1)
                    w.assign(weights[None, :]); goal.assign(.6)
                    problem.setDynamics(u); problem.setDt(dt); problem.setInitialState([0])
                    u.setBounds(-3, 3)
                    problem.createIntermediateResidual("effort", u)
                    tracking = problem.createIntermediateResidual("tracking", cs.vertcat(.2*x, w*u))
                    problem.createFinalResidual("terminal", .3*x)
                    endpoint = problem.createFinalConstraint("endpoint", x-goal)
                    solver = Solver.make_solver("ilqr", problem, {"ilqr.enable_gn": gn,
                        "ilqr.n_threads": threads, "ilqr.max_iter": 30, "ilqr.merit_der_threshold": 1e-10})
                    events = []
                    solver.set_iteration_callback(lambda r: events.append((r.iter, r.cost, r.accepted)) or True)
                    for initial, value, node, active in ((0., .6, N, list(range(N))),
                                                         (.1, -.5, N-2, list(range(0, N, 2)))):
                        problem.setInitialState([initial]); goal.assign(value)
                        tracking.setNodes(active); endpoint.setNodes([node])
                        self.assertTrue(solver.solve())
                        expected = reference_qp(N, dt, initial, node, value, weights, active)
                        np.testing.assert_allclose(solver.x_opt.ravel(), expected[:N+1], atol=2e-6)
                        np.testing.assert_allclose(solver.u_opt.ravel(), expected[N+1:], atol=2e-6)
                        np.testing.assert_allclose(solver.getDt(), dt)
                        np.testing.assert_array_equal(solver.getSolutionDict()["x"], solver.x_opt)
                        self.assertLess(abs(solver.getConstraintsValues()["endpoint"][0, node]), 1e-7)
                        self.assertTrue(solver.getCostsValues())
                        x.setInitialGuess(solver.x_opt); u.setInitialGuess(solver.u_opt)
                    self.assertTrue(events)
                    self.assertTrue(solver.ilqr.getIterationHistory())
                    solver.print_timings()  # Also valid in non-profiling builds.
                    saved = solver.save()
                    restored = cs.Function.deserialize(saved["dynamics"])
                    self.assertAlmostEqual(float(restored(.2, .1, dt)[0]), .225)

    def test_native_graph_parameters_nested_calls_and_lifetime(self):
        x = cs.MX.sym("x"); u = cs.MX.sym("u"); p = cs.MX.sym("p")
        z = cs.SX.sym("z")
        square = pyilqr.CasadiFunction(cs.Function("square", [z], [z*z]))
        functions = SymbolicFunctionManager()
        call = functions.addNative("square", square)
        functions.addExpression("dynamics", [x, u, p], x+u+p, False)
        functions.addResidual("res", [x, u], cs.vertcat(call(call(x)) - .2, u))
        solver = pyilqr.IterativeLQR(functions, "dynamics", 4, {"ilqr.n_threads": 2})
        solver.setResidual(list(range(5)), functions, "res")
        solver.setParameterValue("p", np.full((1, 5), .05))
        lb = np.full((1, 5), -np.inf); ub = -lb; lb[0, 0] = ub[0, 0] = 1
        solver.setStateBounds(lb, ub)
        del call, square, functions
        self.assertTrue(solver.solve(80))
        states = solver.getStateTrajectory()
        np.testing.assert_allclose(np.diff(states), solver.getInputTrajectory()+.05, atol=1e-7)

    @unittest.skipUnless(URDF.is_file() and hasattr(pyilqr, "PinocchioModel"),
                         "Requires a Pinocchio-enabled extension and the Kyon URDF")
    def test_native_kyon_fk_derivatives_and_horizon_solve(self):
        import pinocchio as pin
        from horizon.examples.ilqr_native_pinocchio import make_problem
        model = pin.buildModelFromUrdf(str(URDF)); data = model.createData()
        for threads in (0, 2):
            solver, robot, positions = make_problem(URDF, 12, threads)
            q0 = solver.prb.getState().getInitialGuess()[:, 0]
            value, J = positions.valueAndJacobian(q0)
            pin.computeJointJacobians(model, data, q0); pin.updateFramePlacements(model, data)
            frame_ids = [model.getFrameId(f"contact_{i}") for i in range(1, 5)] + [model.getFrameId("dagana_1_base")]
            expected = np.array([data.oMf[f].translation.copy() for f in frame_ids])
            np.testing.assert_allclose(value.reshape(3, 5).T, expected, atol=1e-12)
            direction = np.linspace(-.03, .03, model.nv); eps = 1e-6
            plus = pin.integrate(model, q0, eps*direction); minus = pin.integrate(model, q0, -eps*direction)
            numerical = (positions.evaluate(plus)-positions.evaluate(minus))/(2*eps)
            np.testing.assert_allclose(J@((plus-minus)/(2*eps)), numerical, atol=1e-8)
            # A numerical solve must not enter Python Pinocchio's FK function.
            with patch.object(pin, "forwardKinematics", side_effect=AssertionError("Python FK called")):
                self.assertTrue(solver.solve())
                solver.prb.getState().setInitialGuess(solver.x_opt)
                solver.prb.getInput().setInitialGuess(solver.u_opt)
                parameter = solver.prb.getParameters("tool_target")
                parameter.assign(parameter.getValues()[:, -1] + [0, .005, 0])
                self.assertTrue(solver.solve())
            trajectory = solver.getSolutionDict()["q"]
            points = np.array([positions.evaluate(q).reshape(3, 5).T for q in trajectory.T])
            self.assertLess(np.linalg.norm(points[:, :4]-points[0, :4], axis=-1).max(), 1e-4)
            target = solver.prb.getParameters("tool_target").getValues()[:, -1]
            self.assertLess(np.linalg.norm(points[-1, 4]-target), .005)
            np.testing.assert_allclose(np.linalg.norm(trajectory[3:7], axis=0), 1, atol=1e-10)
            controls = solver.getSolutionDict()["v"]
            for k in range(12):
                predicted = pin.integrate(model, trajectory[:, k], (2./12)*controls[:, k])
                self.assertLess(np.linalg.norm(pin.difference(model, predicted, trajectory[:, k+1])), 1e-7)
            with self.assertRaisesRegex(NotImplementedError, "Native function graphs"):
                solver.save()
            functions = SymbolicFunctionManager()
            fk = functions.addNative("fk", positions)
            q = cs.MX.sym("q", robot.nq); v = cs.MX.sym("v", robot.nv)
            with self.assertRaisesRegex(ValueError, "Hessians"):
                functions.addCost("unsupported_exact_fk", [q, v], cs.sumsqr(fk(q)))

    def test_ipopt_gauss_newton_curvature_and_native_threaded_maps(self):
        # Nonzero residual distinguishes GN from the exact squared-cost Hessian.
        for abstract in (cs.SX, cs.MX):
            problem = Problem(2, abstract_casadi_type=abstract)
            x = problem.createStateVariable("x", 1)
            u = problem.createInputVariable("u", 1)
            p = problem.createParameter("target", 1)
            problem.setDt(.5)
            problem.createResidual("nonlinear", x*x-p)
            problem.createCost("scalar", .3*x*x)
            solver = Solver.make_solver("ipopt", problem,
                {"ipopt.print_level": 0, "print_time": False}, gauss_newton=True)
            hessian = solver.opts["hess_lag"]
            values = np.r_[np.full(3, 2.), np.zeros(2)]
            H = np.array(hessian(values, np.full(3, 3.), 1.7, []))
            np.testing.assert_allclose(np.diag(H)[:3], 1.7*(8*4+.6))
            np.testing.assert_allclose(np.diag(H)[3:], 0)
            np.testing.assert_allclose(hessian(values, np.full(3, -8.), 0, []), 0)

        N = 8; dt = .25
        trajectories = []
        for backend in ("ilqr", "ipopt"):
            problem = Problem(N, abstract_casadi_type=cs.MX)
            x = problem.createStateVariable("x", 1)
            u = problem.createInputVariable("u", 1)
            p = problem.createParameter("goal", 1); p.assign(.6)
            problem.setDt(dt); problem.setInitialState([0]); u.setBounds(-3, 3)
            functions = SymbolicFunctionManager()
            z = cs.SX.sym("z")
            square = functions.addNative("square", pyilqr.CasadiFunction(cs.Function("square", [z], [z*z])))
            functions.addResidual("tracking", [x, u, p], cs.vertcat(.2*square(x), u))
            functions.addExpression("dynamics", [x, u], x+dt*u, False)
            if backend == "ilqr":
                problem.createFinalConstraint("endpoint", x-p)
                solver = Solver.make_solver(backend, problem,
                    {"ilqr.enable_gn": True, "ilqr.max_iter": 100, "ilqr.merit_der_threshold": 1e-10},
                    function_manager=functions, dynamics="dynamics")
                solver.ilqr.setResidual(list(range(N)), functions, "tracking")
                solver.set_iteration_callback(lambda result: True)
            else:
                native = functions.asCasadi("tracking", 1, 1)
                dyn = functions.asCasadi("dynamics", 1, 1)
                # Horizon uses threaded CasADi maps; this must never call Python.
                problem.createIntermediateConstraint("dynamics", dyn(x, u)-x.getVarOffset(1))
                problem.createFinalConstraint("endpoint", x-p)
                problem.createIntermediateResidual("tracking", native(x, u, p)/np.sqrt(2*N))
                solver = Solver.make_solver(backend, problem,
                    {"ipopt.print_level": 0, "print_time": False, "ipopt.tol": 1e-10}, gauss_newton=True)
            self.assertTrue(solver.solve())
            trajectories.append((solver.x_opt.copy(), solver.u_opt.copy()))
            p.assign(-.4)
            self.assertTrue(solver.solve())
            self.assertAlmostEqual(solver.x_opt[0, -1], -.4, places=7)
        for a, b in zip(*trajectories):
            np.testing.assert_allclose(a, b, atol=2e-5)

    @unittest.skipUnless(URDF.is_file() and hasattr(pyilqr, "PinocchioModel"),
                         "Requires a Pinocchio-enabled extension and the Kyon URDF")
    def test_native_kyon_ipopt_vs_ilqr(self):
        import pinocchio as pin
        import time
        from horizon.examples.ilqr_native_pinocchio import make_problem
        solutions = []
        for backend in ("ilqr", "ipopt"):
            solver, robot, positions = make_problem(URDF, 12, 2, backend)
            started = time.perf_counter()
            with patch.object(pin, "forwardKinematics", side_effect=AssertionError("Python FK called")), \
                 patch.object(pyilqr.Function, "evaluate", side_effect=AssertionError("Python native callback")), \
                 patch.object(pyilqr.Function, "jacobian", side_effect=AssertionError("Python Jacobian callback")):
                self.assertTrue(solver.solve())
            elapsed = time.perf_counter()-started
            q = solver.getSolutionDict()["q"]; v = solver.getSolutionDict()["v"]
            points = np.array([positions.evaluate(state).reshape(3, 5).T for state in q.T])
            target = solver.prb.getParameters("tool_target").getValues()[:, -1]
            error = np.linalg.norm(points[-1, 4]-target)
            drift = np.linalg.norm(points[:, :4]-points[0, :4], axis=-1).max()
            objective = .5*(.05**2*np.sum(v*v)/12 + 100**2*error**2)
            self.assertLess(error, .005); self.assertLess(drift, 1e-7)
            np.testing.assert_allclose(np.linalg.norm(q[3:7], axis=0), 1, atol=1e-8)
            step = robot.step(2./12)
            for k in range(12):
                np.testing.assert_allclose(step.evaluate(np.r_[q[:, k], v[:, k]]), q[:, k+1], atol=2e-7)
            self.assertLess(np.abs(v).max(), 1+1e-7)
            solutions.append((objective, q.copy(), v.copy()))
            print(f"Kyon {backend}: {elapsed:.6f} s, objective={objective:.12g}, target={error:.3g} m, feet={drift:.3g} m")
            # Update a parameter and solve again through the same compiled graph.
            solver.prb.getState().setInitialGuess(q); solver.prb.getInput().setInitialGuess(v)
            solver.prb.getParameters("tool_target").assign(target+[0, .005, 0])
            self.assertTrue(solver.solve())
        self.assertAlmostEqual(solutions[0][0], solutions[1][0], delta=1e-9)
        np.testing.assert_allclose(solutions[0][1], solutions[1][1], atol=5e-4)
        np.testing.assert_allclose(solutions[0][2], solutions[1][2], atol=1e-3)



if __name__ == "__main__":
    unittest.main()

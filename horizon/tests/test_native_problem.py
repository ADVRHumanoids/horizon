"""Problem is the modeling interface; native lowering is deferred to Solver."""
import gc
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import casadi as cs
import numpy as np
from horizon.problem import Problem
from horizon.solvers import Solver, pyilqr
from horizon.native import NativeFunction
from horizon.robot import NativeRobot
from horizon.tests.test_native_robot import URDF


def linear_step():
    z = cs.MX.sym("z", 2)
    return NativeFunction(pyilqr.CasadiFunction(cs.Function("linear_step", [z], [z[0]+z[1]])))


class NativeProblemTest(unittest.TestCase):
    def test_mx_evaluation_and_both_backends_match_problem_residual_semantics(self):
        for backend in ("ilqr", "ipopt"):
            with self.subTest(backend=backend):
                N = 4
                p = Problem(N, abstract_casadi_type=cs.MX, thread_map_num=2)
                x = p.createStateVariable("x", 1)
                u = p.createInputVariable("u", 1)
                dt = p.createParameter("timestep", 1)
                target = p.createParameter("goal", 1)
                dt.assign(.2); target.assign(.9)
                p.setDt(dt); p.setInitialState([0.])
                x.setInitialGuess([0.]); u.setInitialGuess([0.])
                step = linear_step()
                expression = step(x, dt*u)
                evaluator = cs.Function("before_solver", [x,u,dt], [expression])
                self.assertAlmostEqual(float(evaluator(.1,.2,.3)), .16)
                p.setDiscreteDynamics(expression)
                # Mix ordinary and native residuals, including a nested MX function.
                z = cs.MX.sym("z")
                outer = cs.Function("outer", [z], [z], {"never_inline": True})
                p.createIntermediateResidual("effort", .7*u)
                terminal = p.createFinalResidual("tracking", .8*(outer(step(x,0))-target))
                self.assertEqual(terminal.getNodes(), [N])
                del step, evaluator
                gc.collect()
                solver = Solver.make_solver(backend, p, {"ilqr.n_threads":2,"ilqr.huu_reg":0.,
                    "ilqr.merit_der_threshold":1e-12,"ipopt.tol":1e-10,"ipopt.print_level":0,"print_time":False})
                if backend == "ilqr":solver.set_iteration_callback(lambda r: True)
                for h, goal in ((.2,.9),(.35,-.4)):
                    dt.assign(h);target.assign(goal)
                    with patch.object(pyilqr.Function,"evaluate",side_effect=AssertionError("Python native evaluation")):
                        self.assertTrue(solver.solve())
                    expected=.8**2*h*goal/(.7**2+.8**2*N*h**2)
                    np.testing.assert_allclose(solver.getSolutionInput(),expected,atol=2e-7)
                    np.testing.assert_allclose(solver.getSolutionState()[0],np.arange(N+1)*h*expected,atol=2e-7)

    def test_bounded_native_constraints_and_live_bound_changes(self):
        p=Problem(4,abstract_casadi_type=cs.MX)
        x=p.createStateVariable("x",1);u=p.createInputVariable("u",1)
        p.setDt(.2);p.setInitialState([0.]);x.setInitialGuess([0.]);u.setInitialGuess([0.])
        step=linear_step();p.setDiscreteDynamics(step(x,.2*u))
        p.createIntermediateResidual("effort",.7*u)
        p.createFinalResidual("goal",.8*(x-4.))
        force=p.createConstraint("positive_force",step(2.,-u),nodes=range(4),bounds={"lb":[1.5],"ub":[np.inf]})
        with self.assertRaisesRegex(ValueError,"Use IPOPT"):
            Solver.make_solver("ilqr",p)
        solver=Solver.make_solver("ipopt",p,{"ipopt.tol":1e-10,"ipopt.print_level":0,"print_time":False})
        for lower,expected in ((1.5,.5),(1.8,.2)):
            force.setLowerBounds([lower]);self.assertTrue(solver.solve())
            np.testing.assert_allclose(solver.getSolutionInput(),expected,atol=2e-7)

    @unittest.skipUnless(hasattr(pyilqr, "PinocchioModel"), "Requires native Pinocchio")
    def test_full_pose_problem_retains_robot_and_infers_product_manifold(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"robot.urdf";path.write_text(URDF)
            for backend in ("ilqr","ipopt"):
                r=NativeRobot(path);p=Problem(3,abstract_casadi_type=cs.MX)
                q=p.createStateVariable("q",r.nq);v=p.createStateVariable("v",r.nv);tau=p.createInputVariable("tau",r.nv)
                q0=r.configuration_space.neutral();p.setDt(.1);p.setInitialState(np.r_[q0,np.zeros(r.nv)])
                q.setInitialGuess(q0);v.setInitialGuess(np.zeros(r.nv));tau.setInitialGuess(np.zeros(r.nv))
                a,f=r.contactDynamics(q,v,tau,[pyilqr.Contact("body",6)],actuation_matrix=np.eye(r.nv))
                p.setDiscreteDynamics(cs.vertcat(r.integrate(q,.1*v+.005*a),v+.1*a))
                p.createIntermediateResidual("effort",.1*tau)
                target=r.framePose("body",q0)
                p.createFinalResidual("pose",r.framePoseError("body",q,target))
                if backend=="ipopt":p.createConstraint("fz",f[2],nodes=range(3),bounds={"lb":[1.],"ub":[np.inf]})
                del r;gc.collect()
                solver=Solver.make_solver(backend,p,{"ilqr.huu_reg":0.,"ipopt.tol":1e-9,"ipopt.print_level":0,"print_time":False})
                if backend=="ilqr":solver.set_iteration_callback(lambda result:True)
                self.assertTrue(solver.solve())
                np.testing.assert_allclose(solver.getSolutionDict()["q"],np.tile(q0[:,None],(1,4)),atol=1e-8)

    @unittest.skipUnless(hasattr(pyilqr, "PinocchioModel"), "Requires native Pinocchio")
    def test_unactuated_free_body_and_rigid_support(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"robot.urdf"
            path.write_text(URDF)
            robot = NativeRobot(path)
            self.assertEqual(robot.na, 0)
            q = robot.configuration_space.neutral()
            v = np.zeros(robot.nv)
            free_acceleration = robot.forwardDynamics(q, v, np.zeros(0))
            np.testing.assert_allclose(free_acceleration, [0., 0., -9.81, 0., 0., 0.], atol=1e-12)
            acceleration, forces = robot.contactDynamics(q, v, np.zeros(0), [pyilqr.Contact("body", 6)])
            np.testing.assert_allclose(acceleration, np.zeros(robot.nv), atol=1e-12)
            np.testing.assert_allclose(forces, [0., 0., 19.62, 0., 0., 0.], atol=1e-12)

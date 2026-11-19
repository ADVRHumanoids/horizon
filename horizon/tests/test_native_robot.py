"""Native robot API, including torque-to-acceleration/contact-force models."""
import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import casadi as cs
import numpy as np

from horizon.problem import Problem
from horizon.solvers import pyilqr
from horizon.solvers.native_model import NativeModel

URDF = '''<robot name="native_contact_test">
<link name="world"/>
<link name="body"><inertial><mass value="2"/><origin xyz="0 0 0"/>
<inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
<joint name="base" type="floating"><parent link="world"/><child link="body"/></joint>
</robot>'''


@unittest.skipUnless(hasattr(pyilqr, "PinocchioModel"), "Requires native Pinocchio")
class NativeRobotTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native-robot-test-")
        self.path = Path(self.temporary.name) / "robot.urdf"
        self.path.write_text(URDF)
        self.robot = pyilqr.PinocchioModel(str(self.path))

    def tearDown(self):
        self.temporary.cleanup()

    def test_factories_metadata_and_independent_values(self):
        import pinocchio as pin
        robot = self.robot
        model = pin.buildModelFromUrdf(str(self.path))
        data = model.createData()
        q = pin.integrate(model, robot.configuration_space.neutral(), np.linspace(-.2, .3, robot.nv))
        v = np.linspace(-.1, .2, robot.nv)
        a = np.linspace(.3, -.2, robot.nv)
        tau = np.linspace(-2, 1, robot.nv)
        np.testing.assert_allclose(robot.rnea().evaluate(np.r_[q, v, a]), pin.rnea(model, data, q, v, a), atol=1e-12)
        np.testing.assert_allclose(robot.aba().evaluate(np.r_[q, v, tau]), pin.aba(model, data, q, v, tau), atol=1e-12)
        np.testing.assert_allclose(robot.crba().evaluate(q).reshape(robot.nv, robot.nv, order="F"), pin.crba(model, data, q), atol=1e-12)
        pin.centerOfMass(model, data, q, v, a)
        np.testing.assert_allclose(robot.centerOfMass().evaluate(np.r_[q, v, a]), np.r_[data.com[0], data.vcom[0], data.acom[0]], atol=1e-12)
        for reference in (pyilqr.LOCAL, pyilqr.WORLD, pyilqr.LOCAL_WORLD_ALIGNED):
            expected = pin.computeFrameJacobian(model, data, q, model.getFrameId("body"), pin.ReferenceFrame(int(reference.value)))
            np.testing.assert_allclose(robot.jacobian("body", reference).evaluate(q).reshape(6, robot.nv, order="F"), expected, atol=1e-12)
        self.assertEqual(robot.mass, 2.)
        self.assertEqual(robot.joint_nq("base"), robot.nq)
        self.assertEqual(robot.joint_nv("base"), robot.nv)
        self.assertEqual(robot.joint_iq("base"), 0)
        self.assertEqual(robot.joint_iv("base"), 0)
        self.assertIn("base", robot.joint_names)
        self.assertEqual(robot.velocity_limits.shape, (robot.nv,))
        self.assertEqual(robot.effort_limits.shape, (robot.nv,))
        for name in ("integrate", "qdot", "difference", "massMatrix", "forwardDynamics", "gravity", "nonlinearEffects",
                     "kineticEnergy", "potentialEnergy", "comPosition", "comVelocity", "comAcceleration",
                     "centroidalMomentum", "computeCentroidalDynamics", "centroidalDynamics", "ccrba", "centroidalMap"):
            self.assertGreater(getattr(robot, name)().outputSize(), 0)
        for name in ("fk", "framePose", "framePoseError", "frameVelocity", "frameAcceleration", "frameSpatialAcceleration"):
            self.assertGreater(getattr(robot, name)("body").outputSize(), 0)
        with self.assertRaises(ValueError):
            robot.joint_iv("missing")
        with self.assertRaises(ValueError):
            robot.contactDynamics([pyilqr.Contact("body", 2)])
        with self.assertRaises(ValueError):
            robot.contactDynamics(["body"], actuation_matrix=np.zeros((1, 2)))

    def test_contact_graph_both_backends_repeated_solves_and_force_outputs(self):
        robot = self.robot
        nv, nq, N = robot.nv, robot.nq, 3
        q0 = robot.configuration_space.neutral()
        actuation = np.eye(nv)[:, :3]
        contact = pyilqr.Contact("body", 6)
        h = robot.rnea().evaluate(np.r_[q0, np.zeros(2 * nv)])
        for backend in ("ilqr", "ipopt"):
            with self.subTest(backend=backend):
                problem = Problem(N, abstract_casadi_type=cs.MX, thread_map_num=2)
                q = problem.createStateVariable("q", nq)
                v = problem.createStateVariable("v", nv)
                tau = problem.createInputVariable("tau", 3)
                target = problem.createParameter("force_target", 6)
                target.assign(h)
                problem.setDt(.05)
                problem.setInitialState(np.r_[q0, np.zeros(nv)])
                q.setInitialGuess(q0)
                v.setInitialGuess(np.zeros(nv))
                tau.setInitialGuess(np.zeros(3))
                space = pyilqr.ProductSpace([robot.configuration_space, pyilqr.EuclideanSpace(nv)])
                model = NativeModel(problem, state_space=space)
                fd_function = robot.contactDynamics([contact], actuation_matrix=actuation)
                fd = model.addNative("contacts", fd_function)
                response = fd(cs.vertcat(q, v, tau))
                acceleration, force = response[:nv], response[nv:]
                step = model.addNative("step", robot.integrate())
                model.setDynamics(cs.vertcat(step(cs.vertcat(q, model.dt*v + .5*model.dt**2*acceleration)),
                                             v + model.dt*acceleration))
                model.createIntermediateResidual("effort", .1*tau)
                model.createIntermediateResidual("force_tracking", .2*(force-target))
                solver = model.make_solver(backend, {
                    "ilqr.n_threads": 2, "ilqr.max_iter": 25,
                    "ilqr.merit_der_threshold": 1e-12, "ilqr.huu_reg": 0.,
                    "ipopt.tol": 1e-10, "ipopt.print_level": 0, "print_time": False})
                if backend == "ilqr":
                    solver.set_iteration_callback(lambda result: True)
                del model, fd, step
                gc.collect()
                for offset in (np.array([1., 2., -3.]), np.array([-2., .5, 1.])):
                    target_value = h + np.r_[offset, np.zeros(3)]
                    target.assign(target_value)
                    with patch.object(pyilqr.Function, "evaluate", side_effect=AssertionError("Python robot callback")):
                        self.assertTrue(solver.solve())
                    solution = solver.getSolutionDict()
                    np.testing.assert_allclose(solution["tau"], np.tile(-.8*offset[:, None], (1, N)), atol=2e-7)
                    np.testing.assert_allclose(solution["q"], np.tile(q0[:, None], (1, N+1)), atol=1e-9)
                    np.testing.assert_allclose(solution["v"], 0., atol=1e-9)
                    for k in range(N):
                        result = fd_function.evaluate(np.r_[solution["q"][:, k], solution["v"][:, k], solution["tau"][:, k]])
                        np.testing.assert_allclose(result[:nv], 0., atol=1e-10)
                        np.testing.assert_allclose(result[nv:], h-actuation@solution["tau"][:, k], atol=1e-9)

    def test_multiple_floating_base_point_contacts(self):
        import pinocchio as pin
        from horizon.examples.native_pinocchio_support import initial_configuration
        path = Path(__file__).resolve().parents[1] / "cpp/tests/data/kyon_full_spheres.urdf"
        if not path.is_file():
            self.skipTest("Requires the Kyon fixture")
        robot = pyilqr.PinocchioModel(str(path))
        reference = pin.buildModelFromUrdf(str(path))
        data = reference.createData()
        q = initial_configuration(path, robot)
        S = np.eye(robot.nv)[:, 6:]
        contacts = [f"contact_{i}" for i in range(1, 5)]
        fd = robot.contactDynamics(contacts, actuation_matrix=S)
        v = np.linspace(-.01, .02, robot.nv)
        tau = np.linspace(-1., 2., S.shape[1])
        z = np.r_[q, v, tau]
        result = fd.evaluate(z)
        acceleration, forces = result[:robot.nv], result[robot.nv:]
        pin.forwardKinematics(reference, data, q, v, acceleration)
        pin.updateFramePlacements(reference, data)
        for frame in contacts:
            constraint = pin.getFrameClassicalAcceleration(reference, data, reference.getFrameId(frame), pin.LOCAL_WORLD_ALIGNED).linear
            np.testing.assert_allclose(constraint, 0., atol=1e-9)
        inverse = robot.inverseDynamics(contacts)
        np.testing.assert_allclose(inverse.evaluate(np.r_[q, v, acceleration, forces]), S@tau, atol=1e-8)
        jac = fd.jacobian(z)
        eps = 1e-6
        for k in (0, 5, 8, 15):
            direction = np.zeros(robot.nv); direction[k] = eps
            plus, minus = z.copy(), z.copy()
            plus[:robot.nq] = pin.integrate(reference, q, direction)
            minus[:robot.nq] = pin.integrate(reference, q, -direction)
            numerical = (fd.evaluate(plus)-fd.evaluate(minus))/(2*eps)
            projected = jac[:, :robot.nq]@((plus[:robot.nq]-minus[:robot.nq])/(2*eps))
            np.testing.assert_allclose(projected, numerical, atol=2e-6, rtol=1e-6)

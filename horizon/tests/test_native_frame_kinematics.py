"""Native frame and full-pose modeling through Python and both solver backends."""
import gc
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import casadi as cs
import numpy as np

from horizon.problem import Problem
from horizon.solvers import pyilqr
from horizon.solvers.native_model import NativeModel

URDF = Path(os.environ.get("KYON_URDF") or str(Path(__file__).resolve().parents[1] /
            "cpp/tests/data/kyon_full_spheres.urdf"))


@unittest.skipUnless(URDF.is_file() and hasattr(pyilqr, "PinocchioModel"),
                     "Requires the Pinocchio extension and Kyon URDF")
class NativeFrameKinematicsTest(unittest.TestCase):
    def test_frame_values_and_pose_error_target_derivatives(self):
        import pinocchio as pin
        robot = pyilqr.PinocchioModel(str(URDF))
        model = pin.buildModelFromUrdf(str(URDF))
        q = robot.configuration_space.neutral()
        q = pin.integrate(model, q, np.linspace(-.1, .2, robot.nv))
        data = model.createData()
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        frames = ["contact_1", "dagana_1_base"]
        values = robot.framePositions(frames).evaluate(q).reshape(3, 2).T
        for i, frame in enumerate(frames):
            np.testing.assert_allclose(values[i], data.oMf[model.getFrameId(frame)].translation, atol=1e-12)
        pose = robot.framePose(frames[1]).evaluate(q)
        placement = data.oMf[model.getFrameId(frames[1])]
        np.testing.assert_allclose(pose[:3], placement.translation, atol=1e-12)
        np.testing.assert_allclose(pin.Quaternion(pose[3:]).matrix(), placement.rotation, atol=1e-12)
        error = robot.framePoseError(frames[1])
        z = np.r_[q, pose]
        np.testing.assert_allclose(error.evaluate(z), 0, atol=1e-12)
        z[-7:-4] += [.02, -.01, .03]
        target_rotation = pin.Quaternion(placement.rotation @ pin.exp3(np.array([.2, -.3, .1])))
        z[-4:] = 1.2 * target_rotation.coeffs()
        expected = np.r_[[-.02, .01, -.03], pin.log3(target_rotation.matrix().T @ placement.rotation)]
        np.testing.assert_allclose(error.evaluate(z), expected, atol=1e-12)
        jac = error.jacobian(z)
        eps = 1e-7
        for k in range(robot.nq, robot.nq + 7):
            plus, minus = z.copy(), z.copy()
            plus[k] += eps
            minus[k] -= eps
            numerical = (error.evaluate(plus) - error.evaluate(minus)) / (2 * eps)
            np.testing.assert_allclose(jac[:, k], numerical, atol=2e-8)
        with self.assertRaises(ValueError):
            robot.framePose("missing_frame")
        with self.assertRaises(ValueError):
            error.evaluate(np.r_[q, np.zeros(7)])

    def test_full_pose_targets_with_contacts_repeated_solves_and_native_execution(self):
        import pinocchio as pin
        from horizon.examples.native_pinocchio_support import initial_configuration
        reference = pin.buildModelFromUrdf(str(URDF))
        for backend in ("ilqr", "ipopt"):
            with self.subTest(backend=backend):
                robot = pyilqr.PinocchioModel(str(URDF))
                q0 = initial_configuration(URDF, robot)
                N = 8
                problem = Problem(N, abstract_casadi_type=cs.MX, thread_map_num=2)
                q = problem.createStateVariable("q", robot.nq)
                v = problem.createInputVariable("v", robot.nv)
                target = problem.createParameter("pose_target", 7)
                problem.setDt(2. / N)
                q.setBounds(robot.lower_position_limit, robot.upper_position_limit)
                v.setBounds(-np.ones(robot.nv), np.ones(robot.nv))
                problem.setInitialState(q0)
                q.setInitialGuess(q0)
                v.setInitialGuess(np.zeros(robot.nv))
                model = NativeModel(problem, state_space=robot.configuration_space)
                step = model.addNative("step", robot.step(1.))
                model.setDynamics(step(cs.vertcat(q, model.dt * v)))
                model.createIntermediateResidual("effort", .05 * v)
                positions = robot.framePositions([f"contact_{i}" for i in range(1, 5)])
                initial_feet = positions.evaluate(q0)
                feet = model.addNative("contacts", positions)
                model.createConstraint("feet", feet(q) - initial_feet, nodes=range(1, N+1))
                error_function = robot.framePoseError("dagana_1_base")
                error = model.addNative("pose_error", error_function)
                residual = error(cs.vertcat(q, target))
                model.createFinalResidual("tool_pose", cs.vertcat(100 * residual[:3], 30 * residual[3:]))
                pose = robot.framePose("dagana_1_base")
                target.assign(pose.evaluate(q0))
                solver = model.make_solver(backend, {
                    "ilqr.max_iter": 80, "ilqr.n_threads": 2,
                    "ilqr.constr_decomp_type": "svd", "ilqr.huu_reg": 1e-6,
                    "ilqr.merit_der_threshold": 1e-12,
                    "ilqr.constraint_violation_threshold": 1e-8,
                    "ilqr.defect_norm_threshold": 1e-8,
                    "ipopt.max_iter": 200, "ipopt.tol": 1e-10,
                    "ipopt.mu_strategy": "adaptive", "ipopt.mu_min": 1e-13,
                    "ipopt.print_level": 0, "print_time": False})
                if backend == "ilqr":
                    solver.set_iteration_callback(lambda result: True)
                del model, error, step, feet, robot
                gc.collect()
                joint = reference.joints[reference.getJointId("wrist_yaw_1")]
                for offset in (.03, -.02):
                    desired_q = q0.copy()
                    desired_q[joint.idx_q] += offset
                    desired_pose = pose.evaluate(desired_q)
                    target.assign(desired_pose)
                    with patch.object(pyilqr.Function, "evaluate", side_effect=AssertionError("Python native callback")):
                        self.assertTrue(solver.solve(), solver.solver.stats()["return_status"] if backend == "ipopt" else "iLQR failed")
                    solution = solver.getSolutionDict()
                    np.testing.assert_allclose(solution["q"][:, 0], q0, atol=1e-9)
                    actual_error = error_function.evaluate(np.r_[solution["q"][:, -1], desired_pose])
                    self.assertLess(np.linalg.norm(actual_error[:3]), 1e-6)
                    self.assertLess(np.linalg.norm(actual_error[3:]), 1e-6)
                    for state in solution["q"].T:
                        np.testing.assert_allclose(positions.evaluate(state), initial_feet, atol=2e-7)
                    q.setInitialGuess(solution["q"])
                    v.setInitialGuess(solution["v"])


if __name__ == "__main__":
    unittest.main()

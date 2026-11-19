import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import casadi as cs
import numpy as np
from horizon.collision import (Sphere, Box, Capsule, Cylinder, Plane, Mesh, build_sdf,
                               CollisionSpheres, self_collision, environment_collision)
from horizon.robot import NativeRobot
from horizon.problem import Problem
from horizon.solvers import Solver, pyilqr

URDF = '''<robot name="collision_test"><link name="world"/>
<link name="body"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
<collision><geometry><sphere radius="0.1"/></geometry></collision></link>
<joint name="slider" type="prismatic"><parent link="world"/><child link="body"/><axis xyz="1 0 0"/>
<limit lower="-2" upper="2" effort="10" velocity="10"/></joint>
<link name="fixed"><collision><geometry><sphere radius="0.2"/></geometry></collision></link>
<joint name="fixed_joint" type="fixed"><parent link="world"/><child link="fixed"/><origin xyz="1.5 0.3 0.2"/></joint></robot>'''


class CollisionHelpersTest(unittest.TestCase):
    def test_primitive_distances_and_native_grid_gradients(self):
        primitives = [Sphere([0,0,0], .2), Box([1,0,0], [.4,.4,.4]),
                      Capsule([-1,0,-.2],[-1,0,.2],.1), Cylinder([0,1,0],.1,.4)]
        sdf = build_sdf(primitives,bounds=([-2,-2,-2],[2,2,2]),voxel_size=.1)
        for point,expected in (([0,0,0],-.2),([1,0,0],-.2),([-1,0,0],-.1),([0,1,0],-.1)):
            self.assertAlmostEqual(sdf.query(np.array(point,dtype=float))[0],expected,places=6)
        point=np.array([.437,.213,.167]);distance,gradient=sdf.query(point)
        for k in range(3):
            delta=np.eye(3)[k]*1e-6
            self.assertAlmostEqual(gradient[k],(sdf.query(point+delta)[0]-sdf.query(point-delta)[0])/(2e-6),places=7)
        plane=build_sdf([Plane([0,0,0],[0,0,1])],bounds=([-1,-1,-1],[1,1,1]),voxel_size=.2)
        np.testing.assert_allclose(plane.query(np.array([.1,.1,.3]))[1],[0,0,1],atol=1e-7)
        with self.assertRaises(pyilqr.EvaluationError):sdf.query(np.array([10.,0,0]))
        with self.assertRaises(ValueError):build_sdf([],bounds=([0,0,0],[1,1,1]),voxel_size=.1)
        with self.assertRaises(ValueError):build_sdf([Sphere([0,0,0],-1)],bounds=([-1,-1,-1],[1,1,1]),voxel_size=.1)

    def test_transformed_mesh_union_and_invalid_mesh(self):
        import trimesh
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"box.stl";trimesh.creation.box(extents=[.4,.4,.4]).export(path)
            pose=np.eye(4);pose[0,3]=-1.
            sdf=build_sdf([Mesh(path,pose=pose),Sphere([0,0,0],.2)],bounds=([-2,-1,-1],[1,1,1]),voxel_size=.1)
            self.assertAlmostEqual(sdf.query(np.array([-1.,0,0]))[0],-.2,places=6)
            self.assertAlmostEqual(sdf.query(np.array([0.,0,0]))[0],-.2,places=6)
            open_mesh=Path(root)/"open.stl"
            trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]]).export(open_mesh)
            with self.assertRaisesRegex(ValueError,"watertight"):
                build_sdf([Mesh(open_mesh)],bounds=([-1,-1,-1],[1,1,1]),voxel_size=.5)

    @unittest.skipUnless(hasattr(pyilqr, "PinocchioModel"), "Requires native Pinocchio")
    def test_self_and_environment_clearance_native_derivatives(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"robot.urdf";path.write_text(URDF)
            robot=NativeRobot(path)
            geometry=CollisionSpheres.from_urdf(path)
            q=cs.MX.sym("q",1)
            sdf=build_sdf([Sphere([-.5,0,0],.2)],bounds=([-3,-2,-2],[3,2,2]),voxel_size=.1)
            clearance=cs.vertcat(self_collision(robot,q,geometry),environment_collision(robot,q,geometry,sdf))
            evaluate=cs.Function("clearance",[q],[clearance,cs.jacobian(clearance,q)])
            for value in (.237,.743):
                output,jac=evaluate(value)
                numerical=(np.asarray(evaluate(value+1e-6)[0])-np.asarray(evaluate(value-1e-6)[0]))/(2e-6)
                np.testing.assert_allclose(jac,numerical,atol=2e-6)
            self.assertEqual(self_collision(robot,q,geometry,disabled_pairs=[("body","fixed")]).numel(),0)
            with self.assertRaises(FileNotFoundError):self_collision(robot,q,geometry,disabled_pairs=Path(root)/"missing.json")

    @unittest.skipUnless(hasattr(pyilqr, "PinocchioModel"), "Requires native Pinocchio")
    def test_hard_environment_constraint_through_problem(self):
        from scipy.optimize import brentq
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"robot.urdf";path.write_text(URDF)
            robot=NativeRobot(path)
            geometry=CollisionSpheres([("body",[0,.071,.033],.1)])
            sdf=build_sdf([Sphere([0,0,0],.25)],bounds=([-2,-1,-1],[2,1,1]),voxel_size=.05)
            expected=brentq(lambda x:sdf.query(np.array([x,.071,.033]))[0]-.15,.2,.8)
            problem=Problem(2,abstract_casadi_type=cs.MX)
            q=problem.createStateVariable("q",1);v=problem.createInputVariable("v",1)
            problem.setDt(.2);problem.setInitialState([1.]);q.setInitialGuess([1.]);v.setInitialGuess([0.]);v.setBounds([-10.],[10.])
            problem.setDiscreteDynamics(robot.integrate(q,.2*v))
            problem.createFinalResidual("target",q)
            problem.createIntermediateResidual("effort",.001*v)
            clearance=environment_collision(robot,q,geometry,sdf)
            problem.createConstraint("clearance",clearance,bounds={"lb":[.05],"ub":[np.inf]})
            del robot,geometry;gc.collect()
            solver=Solver.make_solver("ipopt",problem,{"ipopt.print_level":0,"ipopt.tol":1e-10,"print_time":False})
            with patch("trimesh.proximity.signed_distance",side_effect=AssertionError("Python geometry at solve time")):
                self.assertTrue(solver.solve())
            self.assertAlmostEqual(solver.getSolutionDict()["q"][0,-1],expected,places=6)

from horizon.problem import Problem
import casadi as cs
import casadi_kin_dyn.py3casadi_kin_dyn as casadi_kin_dyn
from urdf_parser_py.urdf import URDF, Link, Joint, JointLimit, Robot
from horizon.rhc.model_description import FullModelInverseDynamics
from horizon.rhc.taskInterface import TaskInterface
import numpy as np
import yaml
from horizon.utils.analyzer import ProblemAnalyzer

def create_urdf():
    # Define links

    robot = Robot(name="floating_base_robot")

    # Define a floating base link
    world_link = Link(name="world")
    robot.add_link(world_link)

    base_link = Link(name="base_link")
    robot.add_link(base_link)

    # Define the first joint (a floating joint, simulated using a 6-DOF parent link)
    joint = Joint(name="floating_joint", joint_type="floating")
    joint.parent = "world"
    joint.child = "base_link"
    robot.add_joint(joint)

    # Add an arm link
    arm_link = Link(name="arm_link")
    robot.add_link(arm_link)

    # Add a revolute joint between base and arm
    arm_joint = Joint(
        name="middle_joint",
        parent="base_link",
        child="arm_link",
        joint_type="revolute",
        axis=[0, 0, 1],
        limit=JointLimit(lower=-1.57, upper=1.57, effort=10.0, velocity=2.0)
    )
    robot.add_joint(arm_joint)

    return robot.to_xml_string()

tasks_dict = dict()

tasks_dict['solver'] = dict()
tasks_dict['solver']['type'] = 'ilqr'

tasks_dict['costs'] = ['postural_1']

tasks_dict['postural_1'] = dict()
tasks_dict['postural_1']['type'] = 'Postural'
tasks_dict['postural_1']['weight'] = 0.56
# tasks_dict['postural_1']['postural_ref'] = [32.]

with open("test_tasks_config.yaml", "w") as file:
    yaml.dump(tasks_dict, file, default_flow_style=False, sort_keys=False)

ns = 10
dt = 0.01

prb = Problem(ns, receding=True, casadi_type=cs.SX)
prb.setDt(dt)

# Generate and print URDF
urdf = create_urdf()

kin_dyn = casadi_kin_dyn.CasadiKinDyn(urdf)


q_init = dict()
q_init['middle_joint'] = 4.5 * np.ones(1)
base_init = np.array([0, 0, 0, 0, 0, 0, 1])

model = FullModelInverseDynamics(problem=prb,
                                 kd=kin_dyn,
                                 q_init=q_init,
                                 base_init=base_init)

ti = TaskInterface(prb=prb, model=model)

ti.setTaskFromYaml('test_tasks_config.yaml')

prb_anal = ProblemAnalyzer(prb)
prb_anal.print()
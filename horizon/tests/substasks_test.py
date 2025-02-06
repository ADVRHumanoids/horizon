from horizon.problem import Problem
import casadi as cs
import casadi_kin_dyn.py3casadi_kin_dyn as casadi_kin_dyn
from urdf_parser_py.urdf import URDF, Link, Joint, JointLimit
from horizon.rhc.model_description import FullModelInverseDynamics
from horizon.rhc.taskInterface import TaskInterface
import numpy as np
import yaml
from horizon.utils.analyzer import ProblemAnalyzer

def create_urdf():
    # Define links

    base_link = Link(name="base_link")
    middle_link = Link(name="middle_link")
    end_link = Link(name="end_link")

    # Define joints
    base_joint = Joint(
        name="base_joint",
        parent="base_link",
        child="middle_link",
        joint_type="fixed"
    )

    middle_joint = Joint(
        name="middle_joint",
        parent="middle_link",
        child="end_link",
        joint_type="revolute",
        axis=[0, 0, 1],
        limit=JointLimit(lower=-1.57, upper=1.57, effort=10.0, velocity=2.0)
    )

    # Create the URDF model
    robot = URDF(name="simple_robot")
    robot.add_link(base_link)
    robot.add_link(middle_link)
    robot.add_link(end_link)

    robot.add_joint(base_joint)
    robot.add_joint(middle_joint)

    return robot.to_xml_string()

tasks_dict = dict()

tasks_dict['solver'] = dict()
tasks_dict['solver']['type'] = 'ilqr'

# tasks_dict['constraints'] = ['composite_task_1']
tasks_dict['constraints'] = ['composite_task_1']
tasks_dict['costs'] = ['postural_1']

tasks_dict['subtask_1'] = dict()
tasks_dict['subtask_1']['type'] = 'testTask'
tasks_dict['subtask_1']['parameter_1'] = 4.
tasks_dict['subtask_1']['weight'] = 4.


tasks_dict['subtask_2'] = dict()
tasks_dict['subtask_2']['type'] = 'testTask'
tasks_dict['subtask_2']['parameter_1'] = 2.

tasks_dict['composite_task_1'] = dict()
tasks_dict['composite_task_1']['type'] = 'testCompositeTask'
tasks_dict['composite_task_1']['subtask'] = ['subtask_1', 'subtask_2']
tasks_dict['composite_task_1']['weight'] = 0.4

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
q_init['middle_joint'] = np.zeros(1)

model = FullModelInverseDynamics(problem=prb,
                                 kd=kin_dyn,
                                 q_init=q_init,
                                 floating_base=False)

ti = TaskInterface(prb=prb, model=model)

ti.setTaskFromYaml('test_tasks_config.yaml')

prb_anal = ProblemAnalyzer(prb)
# prb_anal.print()
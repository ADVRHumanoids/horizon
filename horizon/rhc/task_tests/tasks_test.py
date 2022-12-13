from horizon.problem import Problem
from casadi_kin_dyn import pycasadi_kin_dyn
from horizon.rhc.model_description import *
import os
import numpy as np
from horizon.rhc.taskInterface import TaskInterface
from horizon.rhc.tasks.interactionTask import InteractionTask
from horizon.utils.actionManager import ActionManager, Step
import casadi as cs
import rospy

# set up model
path_to_examples = os.path.abspath(os.path.dirname(__file__) + "/../../examples")
os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples

urdffile = os.path.join(path_to_examples, 'urdf', 'cogimon.urdf')
urdf = open(urdffile, 'r').read()
rospy.set_param('/robot_description', urdf)

base_init = np.array([0., 0., 0.96, 0., 0.0, 0.0, 1.])

q_init = {"LHipLat":       -0.0,
          "LHipSag":       -0.363826,
          "LHipYaw":       0.0,
          "LKneePitch":    0.731245,
          "LAnklePitch":   -0.307420,
          "LAnkleRoll":    0.0,
          "RHipLat":       0.0,
          "RHipSag":       -0.363826,
          "RHipYaw":       0.0,
          "RKneePitch":    0.731245,
          "RAnklePitch":   -0.307420,
          "RAnkleRoll":    -0.0,
          "WaistLat":      0.0,
          "WaistYaw":      0.0,
          "LShSag":        1.1717860 ,    #0.959931,   #   1.1717860
          "LShLat":        -0.059091562,    #0.007266,   #   -0.059091562
          "LShYaw":        -5.18150657e-02,    #-0.0,       #   -5.18150657e-02
          "LElbj":         -1.85118,    #-1.919862,  #  -1.85118
          "LForearmPlate": 0.0,
          "LWrj1":         -0.523599,
          "LWrj2":         -0.0,
          "RShSag":        1.17128697,  #0.959931,    #   1.17128697
          "RShLat":        6.01664139e-02,  #-0.007266,   #   6.01664139e-02
          "RShYaw":        0.052782481,  #-0.0,        #   0.052782481
          "RElbj":         -1.8513760,  #-1.919862,   #   -1.8513760
          "RForearmPlate": 0.0,
          "RWrj1":         -0.523599,
          "RWrj2":         -0.0}

contact_dict = {
    'l_sole': {
        'type': 'vertex',
        'vertex_frames': [
            'l_foot_lower_left_link',
            'l_foot_upper_left_link',
            'l_foot_lower_right_link',
            'l_foot_upper_right_link',
        ]
    },

    'r_sole': {
        'type': 'vertex',
        'vertex_frames': [
            'r_foot_lower_left_link',
            'r_foot_upper_left_link',
            'r_foot_lower_right_link',
            'r_foot_upper_right_link',
        ]
    }
}


# set up model
urdf = urdf.replace('continuous', 'revolute')
fixed_joints = []
fixed_joints_pos = [q_init[k] for k in fixed_joints]
fixed_joint_map = {k: q_init[k] for k in fixed_joints}
q_init = {k: v for k, v in q_init.items() if k not in fixed_joints}
kd = pycasadi_kin_dyn.CasadiKinDyn(urdf, fixed_joints=fixed_joint_map)

# definition of the problem --> required by the ti
N = 50
tf = 10.0
dt = tf / N

prb = Problem(N, receding=True, casadi_type=cs.SX)
prb.setDt(dt)


# set up model
model = FullModelInverseDynamics(problem=prb,
                                 kd=kd,
                                 q_init=q_init,
                                 base_init=base_init)


# model = SingleRigidBodyDynamicsModel(problem=prb,
#                                      kd=kd,
#                                      q_init=q_init,
#                                      base_init=base_init,
#                                      contact_dict=contact_dict)

ti = TaskInterface(prb=prb,
                   model=model)

ti.setTaskFromYaml(os.path.dirname(__file__) + '/tasks_test.yaml')

print(model.getContacts())

f0 = np.array([0, 0, 315, 0, 0, 0])
init_force = ti.getTask('joint_regularization')
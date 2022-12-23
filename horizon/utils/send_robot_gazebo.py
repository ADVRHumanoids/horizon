# /usr/bin/env python3
import numpy as np
##### for robot and stuff
import xbot_interface.config_options as xbot_opt
from xbot_interface import xbot_interface as xbot
import rospy
from casadi_kin_dyn import py3casadi_kin_dyn
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter, mat_storer
from horizon.ros.replay_trajectory import *
import os

# run ~/iit-centauro-ros-pkg/centauro_gazebo/launch/centauro_world.launch
# run xbot2-core
# run this script
## ==================
## PREPARE TRAJECTORY
## ==================


path_to_examples = os.path.abspath(os.path.dirname(__file__) + "/../../examples")
os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples

urdffile = os.path.join(path_to_examples, 'urdf', 'centauro.urdf')
urdf = open(urdffile, 'r').read()
rospy.set_param('/robot_description', urdf)

ms = mat_storer.matStorer('centauro_up_nice2.mat')
solution = ms.load()

var_q = 'q_res'
dt_res = 0.001
joint_names = solution['joint_names']
q_res = solution[var_q]

n_q = solution[var_q].shape[0]
num_samples = solution[var_q].shape[1]

contacts = ['contact_1', 'contact_2', 'contact_3', 'contact_4']

contact_map_res = {c: solution[f'f_{c}'] for c in contacts}

## PREPARE ROBOT

rospy.init_node('centauro')
opt = xbot_opt.ConfigOptions()

urdf = rospy.get_param('/xbotcore/robot_description')
srdf = rospy.get_param('/xbotcore/robot_description_semantic')

opt.set_urdf(urdf)
opt.set_srdf(srdf)
opt.generate_jidmap()
opt.set_bool_parameter('is_model_floating_base', True)
opt.set_string_parameter('model_type', 'RBDL')
opt.set_string_parameter('framework', 'ROS')
model = xbot.ModelInterface(opt)
robot = xbot.RobotInterface(opt)

robot_state = np.zeros(n_q-7)
robot_state = robot.getJointPosition()

q_robot = q_res[7:, :]
# q_dot_robot = qdot_res[6:, :]
# tau_robot = tau_res[6:, :]


q_homing = q_robot[:, 0]

robot.sense()
rate = rospy.Rate(1./dt_res)

for i in range(100):
    robot.setPositionReference(q_homing)
    # robot.setStiffness(4 *[200, 200, 100]) #[200, 200, 100]
    # robot.setDamping(4 * [50, 50, 30])  # [200, 200, 100]
    robot.move()

# crude homing

input('press a button to replay')

for i in range(num_samples):
    robot.setPositionReference(q_robot[:, i])
    # robot.setVelocityReference(q_dot_robot[:, i])
    # robot.setEffortReference(tau_robot[:, i])
    robot.move()
    rate.sleep()

print("done, if you didn't notice")
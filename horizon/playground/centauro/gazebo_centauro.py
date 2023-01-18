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

# run mon launch ~/iit-centauro-ros-pkg/centauro_gazebo/launch/centauro_world.launch
# check if get_xbot2_config --> ~/iit-centauro-ros-pkg/centauro_config/centauro_basic.yaml
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

from horizon.utils.xbot_handler import XBotHandler
xbh = XBotHandler()
xbh.replay(solution, var_names=['q_res'])

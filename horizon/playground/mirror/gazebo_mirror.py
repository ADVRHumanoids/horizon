# /usr/bin/env python3

import os
import rospkg, rospy
from horizon.utils import mat_storer


# run mon launch ~/iit-mirror-ros-pkg/mirror_gazebo/launch/mirror_world.launch
# check if get_xbot2_config --> ~/iit-mirror-ros-pkg/mirror_config/mirror_basic.yaml
# run xbot2-core
# run this script
## ==================
## PREPARE TRAJECTORY
## ==================

path_to_examples = os.path.abspath(os.path.dirname(__file__) + "/../../examples")
os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples

urdf_path = rospkg.RosPack().get_path('mirror_urdf') + '/urdf/mirror.urdf'
urdf = open(urdf_path, 'r').read()
rospy.set_param('/robot_description', urdf)

# fast_step_in_place.mat done

name = 'mat_files/' + 'mirror_balancing_1_leg.mat'


# name = 'mat_files/' + 'success/' + 'very_good_experiment.mat'
ms = mat_storer.matStorer(name)
solution = ms.load()
 
from horizon.utils.xbot_handler import XBotHandler
xbh = XBotHandler()
xbh.replay(solution, var_names=['q_res'], dt_name='dt_res', homing_duration=2.)

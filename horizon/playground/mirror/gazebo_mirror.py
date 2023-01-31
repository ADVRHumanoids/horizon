# /usr/bin/env python3
import rospy
from horizon.utils import mat_storer
import os
import rospkg

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

name = 'mat_files/' + 'mirror_step_up.mat'
ms = mat_storer.matStorer(name)
solution = ms.load()

from horizon.utils.xbot_handler import XBotHandler
xbh = XBotHandler()
xbh.replay(solution, var_names=['q_res'], dt_name='dt_res')
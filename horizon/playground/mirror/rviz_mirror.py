import rospkg, rospy
from horizon.utils import mat_storer
import subprocess
from pathlib import Path
from horizon.ros import replay_trajectory
from casadi_kin_dyn import pycasadi_kin_dyn
import os

path_to_examples = os.path.abspath(os.path.dirname(__file__) + "/../../examples")
os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples

contacts = [f'arm_{i + 1}_TCP' for i in range(3)]

urdf_path = rospkg.RosPack().get_path('mirror_urdf') + '/urdf/mirror.urdf'
urdf = open(urdf_path, 'r').read()
rospy.set_param('/robot_description', urdf)

# name = 'mat_files/' + 'mirror_demo_3step.mat'
name = 'mat_files/' + 'mirror_demo_dc4_cn99_clea2_in20_en20_tf40.mat'
ms = mat_storer.matStorer(name)
solution = ms.load()

bashCommand = 'rosrun robot_state_publisher robot_state_publisher'
subprocess.Popen(bashCommand.split(), start_new_session=True)

kd = pycasadi_kin_dyn.CasadiKinDyn(urdf)

repl = replay_trajectory.replay_trajectory(solution['dt'], kd.joint_names(), solution['q'], kindyn=kd,
                                           trajectory_markers=contacts)
repl.replay(is_floating_base=True)
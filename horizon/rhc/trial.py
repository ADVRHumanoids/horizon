from horizon.rhc.taskInterface import TaskInterface
from horizon.rhc.tasks.cartesianTask import CartesianTask, Task

import rospkg, rospy
import numpy as np

urdf_path = rospkg.RosPack().get_path('mirror_urdf') + '/urdf/mirror.urdf'
urdf = open(urdf_path, 'r').read()

# contact frames
contacts = [f'arm_{i + 1}_TCP' for i in range(3)]

N = 50
tf = 10.0
dt = tf / N
nf = 3
problem_opts = {'ns': N, 'tf': tf, 'dt': dt}

nc = len(contacts)
model_description = 'whole_body'

q_init = {}

for i in range(3):
    q_init[f'arm_{i + 1}_joint_2'] = -1.9
    q_init[f'arm_{i + 1}_joint_3'] = -2.30
    q_init[f'arm_{i + 1}_joint_5'] = -0.4

base_init = np.array([0, 0, 0.72, 0, 0, 0, 1])

ti = TaskInterface(urdf, q_init, base_init, problem_opts, model_description)

ti.model.setContactFrame('arm_1_TCP')
# ti.setContact()
#
# goalrz = {'type': 'Postural',
#           'name': 'final_base_rz',
#           'indices': [5],
#           'nodes': [N],
#           'fun_type': 'cost',
#           'weight': 1e3}

# limits = {'type': 'JointLimits',
#           'name': 'final_base_rz',
#           'indices': [5],
#           'nodes': [N],
#           'fun_type': 'cost',
#           'weight': 1e3}

cart = {'type': 'Cartesian',
        'frame': 'arm_1_TCP',
        'name': 'final_base_rz',
        'indices': [2],
        'nodes': [N],
        'fun_type': 'cost',
        'weight': 1e3}

# interactive = {'type': 'Force',
#                'frame': 'arm_1_TCP',
#                'name': 'faboulous',
#                'indices': [2],
#                'nodes': [N-1],
#                'fun_type': 'cost',
#                'weight': 1e3}

# cart1 = CartesianTask('arm_1_TCP', prb=ti.prb, kin_dyn=ti.kd, name='daniele', nodes=[N])
# ti.setTaskFromDict(goalrz)
# ti.setTaskFromDict(limits)
ti.setTaskFromDict(cart)
# ti.setTaskFromDict(interactive)

print(ti.model.getContacts())

# ti.setTask(cart1)


from horizon.problem import Problem
from horizon.rhc.taskInterface import TaskInterface
from horizon.rhc.tasks.cartesianTask import CartesianTask, Task
from horizon.rhc.model_description import *
from horizon.ros import replay_trajectory
from horizon.solvers import Solver
import rospkg, rospy
import numpy as np
import casadi as cs
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.utils.tf_broadcaster import TFBroadcaster

urdf_path = rospkg.RosPack().get_path('repair_urdf') + '/urdf/repair.urdf'
urdf = open(urdf_path, 'r').read()
rospy.set_param('/robot_description', urdf)

solver_type = 'ipopt'  # ilqr
transcription_method = 'multiple_shooting'  # can choose between 'multiple_shooting' and 'direct_collocation'
transcription_opts = dict(integrator='RK4')  # integrator used by the multiple_shooting

solver_opt = dict(type=solver_type)

def add_cartesian_tasks_vel():
    cart_vel_1 = {'type': 'Cartesian',
                  'distal_link': 'arm_1_tcp',
                  'name': 'arm_1_tcp_ee_vel_world',
                  'indices': [0, 1, 2, 3, 4, 5],
                  'indices': [0, 1, 2, 3, 4, 5],
                  'nodes': range(1, N),
                  'cartesian_type': 'velocity'}

    cart_vel_2 = {'type': 'Cartesian',
                  'distal_link': 'arm_2_tcp',
                  'base_link': 'arm_1_tcp',
                  'name': 'arm_2_tcp_ee_vel_rel',
                  'indices': [0, 1, 2, 3, 4, 5],
                  'nodes': range(1, N),
                  'cartesian_type': 'velocity'}

    ee_cart_1 = ti.setTaskFromDict(cart_vel_1)
    goal_vec_1 = [0., 0., 0., 0., 0., 0.2]
    ee_cart_1.setRef(goal_vec_1)


    ee_cart_2 = ti.setTaskFromDict(cart_vel_2)
    goal_vec_2 = [0., 0., 0., 0., 0., 0.2]
    ee_cart_2.setRef(goal_vec_2)


def add_cartesian_tasks_pos():
    cart_1 = {'type': 'Cartesian',
              'distal_link': 'arm_1_tcp',
              'name': 'arm_1_tcp_ee_world',
              'indices': [0, 1, 2, 3, 4, 5],
              'nodes': [N]}

    cart_2 = {'type': 'Cartesian',
              'distal_link': 'arm_2_tcp',
              'base_link': 'arm_1_tcp',
              'name': 'arm_2_tcp_ee_rel',
              'indices': [0, 1, 2, 3, 4, 5],
              'nodes': [N]}

    ee_cart_1 = ti.setTaskFromDict(cart_1)
    ee_cart_2 = ti.setTaskFromDict(cart_2)

    # goal_vec = [0, 0, 0., 0, 0, 0, 1]
    goal_vec = [0.5, -0.2, 0.5, 0, 0, 0, 1]
    # goal_vec = [0.5, -0.2, 0.5, 0, 0.7071068, 0, 0.7071068]
    goal_vec_1 = [0., 0.3, 0., 0, 0.7071068, 0, 0.7071068]  # 90deg on the y
    goal_vec_1 = [0., 0.3, 0., 0, 0, 0.3826834, 0.9238795]  # 45deg on the x
    # goal_vec = [0.5, -0.2, 0.5, 0.2705981, 0.2705981, 0, 0.9238795]

    tf = TFBroadcaster()
    tf.publish('arm_1_tcp_ee_goal', goal_vec)

    ee_cart_1.setRef(goal_vec)
    ee_cart_2.setRef(goal_vec_1)


# robot model
q_init = {}

q_init[f'arm_1_joint_1'] = 0.
q_init[f'arm_1_joint_2'] = -0.76
q_init[f'arm_1_joint_3'] = 1.5
q_init[f'arm_1_joint_4'] = -1.3
q_init[f'arm_1_joint_5'] = 0.
q_init[f'arm_1_joint_6'] = -0.4
q_init[f'arm_1_joint_7'] = 0.
q_init[f'arm_2_joint_1'] = 0.
q_init[f'arm_2_joint_2'] = -0.76
q_init[f'arm_2_joint_3'] = -1.5
q_init[f'arm_2_joint_4'] = -1.3
q_init[f'arm_2_joint_5'] = 0.
q_init[f'arm_2_joint_6'] = -0.4
q_init[f'arm_2_joint_7'] = 0.

kd = pycasadi_kin_dyn.CasadiKinDyn(urdf)
kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
# joint_names = kd.joint_names()[2:]

N = 50
tf = 2.0
dt = tf / N
nf = 3

prb = Problem(N, receding=False)  # logging_level=logging.DEBUG
prb.setDt(dt)

# set up model
model = FullModelInverseDynamics(problem=prb,
                                 kd=kd,
                                 q_init=q_init,
                                 floating_base=False)

ti = TaskInterface(prb, model)

# cartesian_task
add_cartesian_tasks_vel()


q = ti.prb.getVariables('q')
v = ti.prb.getVariables('v')
a = ti.prb.getVariables('a')

# TODO: being the only cost, why changing its weight changes also the solution??
ti.prb.createIntermediateResidual('min_q', a)

q.setBounds(model.q0, model.q0, nodes=0)
v.setBounds(model.v0, model.v0, nodes=0)

q.setInitialGuess(model.q0)

ti.setSolverOptions(solver_opt)
ti.finalize()
ti.bootstrap()
solution = ti.solution

# ===========================================================================================
# ================================== to wrap ================================================
# ===========================================================================================


# os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
# subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=spot'])
# rospy.loginfo("'spot' visualization started.")

## single replay


q_sol = solution['q']
repl = replay_trajectory.replay_trajectory(dt, kd.joint_names()[1:], q_sol, kindyn=kd)
repl.sleep(1.)
repl.replay(is_floating_base=False)

# plot_flag = True
# if plot_flag:
#     import matplotlib.pyplot as plt
#     import matplotlib
#
#     plt.figure()
#     for frame in :
#         FK = cs.Function.deserialize(ti.kd.fk(contact))
#         pos = FK(q=solution['q'])['ee_pos']
#
#         plt.title(f'feet position - plane_xy')
#         plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[1, :]).flatten(), linewidth=2.5)
#         plt.scatter(np.array(pos[0, 0]), np.array(pos[1, 0]))
#         plt.scatter(np.array(pos[0, -1]), np.array(pos[1, -1]), marker='x')
#
#
#
#         plt.title(f'feet position - plane_xz')
#         plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[2, :]).flatten(), linewidth=2.5)
#         plt.scatter(np.array(pos[0, 0]), np.array(pos[2, 0]))
#         plt.scatter(np.array(pos[0, -1]), np.array(pos[2, -1]), marker='x')
#
#     hplt = plotter.PlotterHorizon(ti.prb, solution)
#     hplt.plotVariables([elem.getName() for elem in forces], show_bounds=True, gather=2, legend=False)
#     hplt.plotVariables(['q'], show_bounds=True, gather=2, legend=False)
#     matplotlib.pyplot.show()
#
#

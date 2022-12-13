import logging
import os
import numpy as np
from horizon.rhc.taskInterface import TaskInterface
from horizon.problem import Problem
from horizon.rhc.model_description import *
from horizon.rhc.tasks.interactionTask import InteractionTask
from horizon.utils.actionManager import ActionManager, Step
from casadi_kin_dyn import py3casadi_kin_dyn
import casadi as cs
import rospy

rospy.init_node('centauro_up_node')
# set up model
path_to_examples = os.path.abspath(os.path.dirname(__file__) + "/../../examples")
os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples

urdffile = os.path.join(path_to_examples, 'urdf', 'centauro.urdf')
urdf = open(urdffile, 'r').read()
rospy.set_param('/robot_description', urdf)

fixed_joint_map = {'torso_yaw': 0.00,   # 0.00,

                    'j_arm1_1': 1.50,   # 1.60,
                    'j_arm1_2': 0.1,    # 0.,
                    'j_arm1_3': 0.2,   # 1.5,
                    'j_arm1_4': -2.2,  # 0.3,
                    'j_arm1_5': 0.00,   # 0.00,
                    'j_arm1_6': -1.3,   # 0.,
                    'j_arm1_7': 0.0,    # 0.0,

                    'j_arm2_1': 1.50,   # 1.60,
                    'j_arm2_2': 0.1,    # 0.,
                    'j_arm2_3': -0.2,   # 1.5,
                    'j_arm2_4': -2.2,   #-0.3,
                    'j_arm2_5': 0.0,    # 0.0,
                    'j_arm2_6': -1.3,   # 0.,
                    'j_arm2_7': 0.0,    # 0.0,
                    'd435_head_joint': 0.0,
                    'velodyne_joint': 0.0,

                    # 'hip_yaw_1': -0.746,
                    # 'hip_pitch_1': -1.254,
                    # 'knee_pitch_1': -1.555,
                    # 'ankle_pitch_1': -0.3,
                    #
                    # 'hip_yaw_2': 0.746,
                    # 'hip_pitch_2': 1.254,
                    # 'knee_pitch_2': 1.555,
                    # 'ankle_pitch_2': 0.3,
                    #
                    # 'hip_yaw_3': 0.746,
                    # 'hip_pitch_3': 1.254,
                    # 'knee_pitch_3': 1.555,
                    # 'ankle_pitch_3': 0.3,
                    #
                    # 'hip_yaw_4': -0.746,
                    # 'hip_pitch_4': -1.254,
                    # 'knee_pitch_4': -1.555,
                    # 'ankle_pitch_4': -0.3,
                    }

# initial config
q_init = {
    'hip_yaw_1': -0.746,
    'hip_pitch_1': -1.254,
    'knee_pitch_1': -1.555,
    'ankle_pitch_1': -0.3,

    'hip_yaw_2': 0.746,
    'hip_pitch_2': 1.254,
    'knee_pitch_2': 1.555,
    'ankle_pitch_2': 0.3,

    'hip_yaw_3': 0.746,
    'hip_pitch_3': 1.254,
    'knee_pitch_3': 1.555,
    'ankle_pitch_3': 0.3,

    'hip_yaw_4': -0.746,
    'hip_pitch_4': -1.254,
    'knee_pitch_4': -1.555,
    'ankle_pitch_4': -0.3,
}

wheels = [f'j_wheel_{i + 1}' for i in range(4)]
q_init.update(zip(wheels, 4 * [0.]))

ankle_yaws = [f'ankle_yaw_{i + 1}' for i in range(4)]
# q_init.update(zip(ankle_yaws, 4 * [0.]))
q_init.update(dict(ankle_yaw_1=np.pi/4))
q_init.update(dict(ankle_yaw_2=-np.pi/4))
q_init.update(dict(ankle_yaw_3=-np.pi/4))
q_init.update(dict(ankle_yaw_4=np.pi/4))

q_init.update(fixed_joint_map)

base_init = np.array([0, 0, 0.718565, 0, 0, 0, 1])

# set up model description
urdf = urdf.replace('continuous', 'revolute')

kd = pycasadi_kin_dyn.CasadiKinDyn(urdf, fixed_joints=fixed_joint_map)
kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
q_init = {k: v for k, v in q_init.items() if k not in fixed_joint_map.keys()}

# set up problem
N = 100
tf = 10.0
dt = tf / N

prb = Problem(N, receding=True)  # logging_level=logging.DEBUG
prb.setDt(dt)

# set up model
model = FullModelInverseDynamics(problem=prb,
                                 kd=kd,
                                 q_init=q_init,
                                 base_init=base_init,
                                 fixed_joint_map=fixed_joint_map)
# else:
#     model = SingleRigidBodyDynamicsModel(problem=prb,
#                                          kd=kd,
#                                          q_init=q_init,
#                                          base_init=base_init,
#                                          contact_dict=contact_dict)


ti = TaskInterface(prb=prb,
                   model=model)

ti.setTaskFromYaml(os.path.dirname(__file__) + '/centauro_config.yaml')

f0 = np.array([0, 0, 280, 0, 0, 0])
init_force = ti.getTask('joint_regularization')
# init_force.setRef(1, f0)
# init_force.setRef(2, f0)

final_base_x = ti.getTask('final_base_xy')
final_base_x.setRef([.6, 0, 0, 0, 0, 0, 1])


# final_base_y = ti.getTask('base_posture')
# final_base_y.setRef([0, 0, 0.718565, 0, 0, 0, 1])

opts = dict()
am = ActionManager(ti, opts)
goal_step = [0.0, 0.0, 0.3]
n_start_step = 30
n_goal_step = 35

# am._walk([10, 40], [0, 3])
am._step(Step(frame='contact_1', k_start=n_start_step, k_goal=n_goal_step, goal=goal_step))
am._step(Step(frame='contact_2', k_start=n_start_step, k_goal=n_goal_step, goal=goal_step))

# contact_1 = ti.getTask('foot_contact_contact_1')
# contact_1.setNodes(list(range(5)) + list(range(15, 50)))

# rolling_1 = ti.getTask('rolling_contact_1')
# rolling_1.setNodes(list(range(5)) + list(range(15, 50)))

# todo: horrible API
# l_contact.setNodes(list(range(5)) + list(range(15, 50)))
# r_contact.setNodes(list(range(0, 25)) + list(range(35, 50)))


# ===============================================================
# ===============================================================
# ===============================================================
q = ti.prb.getVariables('q')
v = ti.prb.getVariables('v')
a = ti.prb.getVariables('a')

# ti.prb.createFinalResidual("min_qf", 1e2 * (q[7:] - ti.model.q0[7:]))
# ti.prb.createFinalCost("min_rot", 1e-1 * (q[3:4] - ti.model.q0[3:4]))
# ti.prb.createFinalCost("min_rot", 1e1 * (q[3:5] - ti.model.q0[3:5]))
# adding minimization of angular momentum
# cd_fun = ti.model.kd.computeCentroidalDynamics()
# h_lin, h_ang, dh_lin, dh_ang = cd_fun(q, v, a)
# ti.prb.createIntermediateResidual('min_angular_mom', 0.1 * dh_ang)

# continuity of forces
continuity_force_w = 5e-2
for f_name, f_var in model.fmap.items():
    f_var_prev = f_var.getVarOffset(-1)
    prb.createIntermediateResidual(f'continuity_force_{f_name}', continuity_force_w * (f_var - f_var_prev), range(1, prb.getNNodes() - 2))

q.setBounds(ti.model.q0, ti.model.q0, nodes=0)
v.setBounds(ti.model.v0, ti.model.v0, nodes=0)

q.setInitialGuess(ti.model.q0)

for cname, cforces in ti.model.cmap.items():
    for c in cforces:
        c.setInitialGuess(f0[:c.size1()])

replay_motion = True
plot_sol = not replay_motion

# todo how to add?
ti.prb.createFinalConstraint('final_v', v)
# ================================================================
# ================================================================
# ===================== stuff to wrap ============================
# ================================================================
# ================================================================
from horizon.ros import replay_trajectory

ti.finalize()
ti.bootstrap()
solution = ti.solution
ti.resample(0.001)
ti.save_solution('centauro_up.mat')
ti.replay_trajectory()
exit()

n_nodes = ti.prb.getNNodes()
nodes_vec = np.zeros([n_nodes])

for i in range(1, n_nodes):
    nodes_vec[i] = nodes_vec[i - 1] + ti.prb.getDt()

num_samples = ti.solution['tau_res'].shape[1]
dt_res = ti.solution['dt_res']

nodes_vec_res = np.zeros([num_samples + 1])
for i in range(1, num_samples + 1):
    nodes_vec_res[i] = nodes_vec_res[i - 1] + dt_res


tau_sol_res = ti.solution['tau_res']
tau_sol_base = tau_sol_res[:6, :]


from matplotlib import pyplot as plt
threshold = 5
## get index of values greater than a given threshold for each dimension of the vector, and remove all the duplicate values (given by the fact that there are more dimensions)
indices_exceed = np.unique(np.argwhere(np.abs(tau_sol_base) > threshold)[:, 1])
# these indices corresponds to some nodes ..
values_exceed = nodes_vec_res[indices_exceed]

## search for duplicates and remove them, both in indices_exceed and values_exceed
indices_duplicates = np.where(np.in1d(values_exceed, nodes_vec))
value_duplicates = values_exceed[indices_duplicates]

values_exceed = np.delete(values_exceed, np.where(np.in1d(values_exceed, value_duplicates)))
indices_exceed = np.delete(indices_exceed, indices_duplicates)

## base vector nodes augmented with new nodes + sort
nodes_vec_augmented = np.concatenate((nodes_vec, values_exceed))
nodes_vec_augmented.sort(kind='mergesort')

plot_tau_base = True
if plot_tau_base:
    plt.figure()
    for dim in range(6):
        plt.plot(nodes_vec_res[:-1], np.array(tau_sol_res[dim, :]))
    plt.title('tau on base')


    plt.hlines([threshold], nodes_vec[0], nodes_vec[-1], linestyles='dashed', colors='k', linewidth=0.4)
    plt.hlines([-threshold], nodes_vec[0], nodes_vec[-1], linestyles='dashed', colors='k', linewidth=0.4)

    plt.show()

print(nodes_vec_augmented.shape)
from horizon.utils import refiner

ref = refiner.Refiner(prb, nodes_vec_augmented, ti.solver_bs)

ref.resetProblem()
ref.resetFunctions()
ref.resetVarBounds()
ref.resetInitialGuess()
ref.addProximalCosts()
ref.solveProblem()
sol_var, sol_cnsrt, sol_dt = ref.getSolution()

new_prb = ref.getAugmentedProblem()

# ms = mat_storer.matStorer(f'refiner_spot_jump.mat')
# sol_cnsrt_dict = dict()
# for name, item in new_prb.getConstraints().items():
#     lb, ub = item.getBounds()
#     lb_mat = np.reshape(lb, (item.getDim(), len(item.getNodes())), order='F')
#     ub_mat = np.reshape(ub, (item.getDim(), len(item.getNodes())), order='F')
#     sol_cnsrt_dict[name] = dict(val=sol_cnsrt[name], lb=lb_mat, ub=ub_mat, nodes=item.getNodes())
#
# from horizon.variables import Variable, SingleVariable, Parameter, SingleParameter

# info_dict = dict(n_nodes=new_prb.getNNodes(), times=nodes_vec_augmented, dt=sol_dt)
# ms.store({**sol_var, **sol_cnsrt_dict, **info_dict})
# =========================================================================













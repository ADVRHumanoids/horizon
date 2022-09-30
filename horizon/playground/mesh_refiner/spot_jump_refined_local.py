# /usr/bin/env python3

############
#experiment to augment the nodes with supporting nodes.
# This is useful when simply resampling the trajectory make it unfeasible (especially for the tau on the floating base, which become non zero).
# The underlying idea is LOCALLY refining the trajectory by injecting more nodes so that, when resampling, less errors pop up.
#to visualize the trajectory use vis_refiner_local.py and to resample + send it to gazebo use send_to_gazebo.py
##################
import time

import horizon.variables
from horizon import problem
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.solvers import Solver
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter, mat_storer
import os
from horizon.ros.replay_trajectory import *
import numpy as np
import matplotlib.pyplot as plt

## ==================
## PREPARE TRAJECTORY
## ==================
transcription_method = 'multiple_shooting'  # direct_collocation
transcription_opts = dict(integrator='RK4')

path_to_examples = '../../examples/'
urdffile = path_to_examples + 'urdf/spot.urdf'
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

n_c = 4
n_q = kindyn.nq()
n_v = kindyn.nv()
n_f = 3

ms = mat_storer.matStorer('refining_mat/spot_jump.mat')
# ms = mat_storer.matStorer('spot_jump_refined_local.mat')
prev_solution = ms.load()

n_nodes = prev_solution['n_nodes'][0][0]

node_start_step = prev_solution['node_start_step'][0][0]
node_end_step = prev_solution['node_end_step'][0][0]
node_peak = prev_solution['node_peak'][0][0]
jump_height = prev_solution['jump_height'][0][0]

prev_q = prev_solution['q']
prev_q_dot = prev_solution['q_dot']
prev_q_ddot = prev_solution['q_ddot']

prev_f_list = list()
for i in range(n_c):
    prev_f_list.append(prev_solution[f'f{i}'])

prev_tau = prev_solution['inverse_dynamics']['val'][0][0]
contacts_name = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
prev_contact_map = dict(zip(contacts_name, prev_f_list))

joint_names = kindyn.joint_names()
if 'universe' in joint_names: joint_names.remove('universe')
if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')

if 'dt' in prev_solution:
    prev_dt = prev_solution['dt'].flatten()
elif 'constant_dt' in prev_solution:
    prev_dt = prev_solution['constant_dt'].flatten()[0]
elif 'param_dt' in prev_solution:
    prev_dt = prev_solution['param_dt'].flatten()

dt_res = 0.001

q_sym = cs.SX.sym('q', n_q)
q_dot_sym = cs.SX.sym('q_dot', n_v)
q_ddot_sym = cs.SX.sym('q_ddot', n_v)
x_dot = utils.double_integrator_with_floating_base(q_sym, q_dot_sym, q_ddot_sym)
x = cs.vertcat(q_sym, q_dot_sym)

dae = {'x': x, 'p': q_ddot_sym, 'ode': x_dot, 'quad': 1}
q_res, qdot_res, qddot_res, contact_map_res, tau_sol_res = resampler_trajectory.resample_torques(
    prev_q, prev_q_dot, prev_q_ddot, prev_dt, dt_res, dae, prev_contact_map,
    kindyn,
    cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

f_res_list = list()
for f in prev_f_list:
    f_res_list.append(resampler_trajectory.resample_input(f, prev_dt, dt_res))

num_samples = tau_sol_res.shape[1]

nodes_vec = np.zeros([n_nodes + 1])
for i in range(1, n_nodes + 1):
    nodes_vec[i] = nodes_vec[i - 1] + prev_dt[i - 1]

nodes_vec_res = np.zeros([num_samples + 1])
for i in range(1, num_samples + 1):
    nodes_vec_res[i] = nodes_vec_res[i - 1] + dt_res

plot_flag = True

if plot_flag:
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.ticker import FormatStrFormatter
    w = 7195
    h = 3841
    fig_size = [19.20, 10.80]

    plt.rcParams['text.usetex'] = True
    plt.rcParams["mathtext.fontset"] = 'cm'
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # for i_f in range(len(f_res_list)):
    #     plt.figure()
    #     for dim in range(f_res_list[i_f].shape[0]):
    #         plt.plot(nodes_vec_res[:-1], np.array(f_res_list[i_f][dim, :]))
    #
    #     for dim in range(prev_f_list[i_f].shape[0]):
    #         plt.scatter(nodes_vec[:-1], np.array(prev_f_list[i_f][dim, :]))
    #
    # plt.figure()
    # for dim in range(q_res.shape[0]):
    #     plt.plot(nodes_vec_res, np.array(q_res[dim, :]))
    #
    # for dim in range(prev_q.shape[0]):
    #     plt.scatter(nodes_vec, np.array(prev_q[dim, :]))
    # plt.title('q')
    #
    # plt.figure()
    # for dim in range(qdot_res.shape[0]):
    #     plt.plot(nodes_vec_res, np.array(qdot_res[dim, :]))
    #
    # for dim in range(prev_q_dot.shape[0]):
    #     plt.scatter(nodes_vec, np.array(prev_q_dot[dim, :]))
    # plt.title('qdot')
    #
    # plt.figure()
    # for dim in range(qddot_res.shape[0]):
    #     plt.plot(nodes_vec_res[:-1], np.array(qddot_res[dim, :]))
    #
    # for dim in range(prev_q_ddot.shape[0]):
    #     plt.scatter(nodes_vec[:-1], np.array(prev_q_ddot[dim, :]))
    # plt.title('q_ddot')


    fig, ax = plt.subplots()
    fig.set_size_inches(fig_size[0], fig_size[1])
    for dim in range(6):
        ax.plot(nodes_vec_res[:-1], np.array(tau_sol_res[dim, :]), linewidth=1)
        ax.scatter(nodes_vec[:-1], np.array(prev_tau[dim, :]), s=30, facecolors='none', edgecolors='#d62728', zorder=3)

    ax.set_xticks(nodes_vec[:-1])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.grid(alpha=0.4)

    ax.yaxis.set_major_locator(plt.MultipleLocator(2))

    wanted_time_label = {0, 20, 40, 49}
    label_list = list(range(nodes_vec[:-1].shape[0]))
    label_list = [e for e in label_list if e not in wanted_time_label]
    xticks = ax.xaxis.get_major_ticks()
    for i_hide in label_list:
        xticks[i_hide].label1.set_visible(False)

    plt.xlim([0, nodes_vec[-1]])

    # for dim in range(6):
    #     ax.plot(nodes_vec[:-1], np.array(prev_tau[dim, :]))
    # plt.title('tau on base')

    # ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))
    # x


    # plt.figure()
    # for dim in range(tau_sol_res.shape[0] - 6):
    #     plt.plot(nodes_vec_res[:-1], np.array(tau_sol_res[6 + dim, :]))
    # for dim in range(prev_tau.shape[0] - 6):
    #     plt.scatter(nodes_vec[:-1], np.array(prev_tau[6 + dim, :]))
    # plt.title('tau')
    plt.show()


tau_sol_base = tau_sol_res[:6, :]

threshold = 5.
## get index of values greater than a given threshold for each dimension of the vector, and remove all the duplicate values (given by the fact that there are more dimensions)
indices_exceed = np.unique(np.argwhere(np.abs(tau_sol_base) > threshold)[:, 1])
# these indices corresponds to some nodes ..
values_exceed = nodes_vec_res[indices_exceed]


## search for duplicates and remove them, both in indices_exceed and values_exceed
indices_duplicates = np.where(np.in1d(values_exceed, nodes_vec))
value_duplicates = values_exceed[indices_duplicates]

values_exceed = np.delete(values_exceed, np.where(np.in1d(values_exceed, value_duplicates)))
indices_exceed = np.delete(indices_exceed, indices_duplicates)

print('number of supplementary nodes:', len(indices_exceed))

## base vector nodes augmented with new nodes + sort
nodes_vec_augmented = np.concatenate((nodes_vec, values_exceed))
nodes_vec_augmented.sort(kind='mergesort')

print(nodes_vec_augmented)
# new number of nodes

# ===========================================================

old_nodes = n_nodes

old_node_start_step = node_start_step
old_node_end_step = node_end_step
old_node_peak = node_peak

time_start_step = nodes_vec[old_node_start_step]
time_end_step = nodes_vec[old_node_end_step]
time_peak = nodes_vec[old_node_peak]

node_start_step = np.where(abs(time_start_step - nodes_vec_augmented) < dt_res)
if node_start_step:
    node_start_step = int(node_start_step[0][0])
else:
    raise Exception('something is wrong with time_start_step')

node_end_step = np.where(abs(time_end_step - nodes_vec_augmented) < dt_res)  # dt_res
if node_end_step:
    node_end_step = int(node_end_step[0][0])
else:
    raise Exception('something is wrong with time_start_step')

node_peak = np.where(abs(time_peak - nodes_vec_augmented) < dt_res)  # dt_res
if node_peak:
    node_peak = int(node_peak[0][0])
else:
    raise Exception('something is wrong with node_peak')

# ===================================================================================
# ============================== REFINER ============================================
# ===================================================================================

new_n_nodes = nodes_vec_augmented.shape[0]
new_dt_vec = np.diff(nodes_vec_augmented)

# ===================================== remove dt that are too small ====================================================
remove_small_dt = False
if remove_small_dt:
    index_granularity = np.where(new_dt_vec < 9e-4)[0]
    if index_granularity.shape[0] > 0:
        # removed values from the augmented vector of nodes
        removed_values = nodes_vec_augmented[index_granularity]
        nodes_vec_augmented = np.delete(nodes_vec_augmented, index_granularity)

        print(f'{index_granularity.shape[0]} dt are too small. Removing the corresponding values: {removed_values} ({new_dt_vec[index_granularity]})')
        # search in all the values exceeded which one are removed
        index_removed = np.searchsorted(values_exceed, removed_values)

        # update the indices_exceed removing the one too small: for the zip
        indices_exceed = np.delete(indices_exceed, index_removed)

        new_dt_vec = np.diff(nodes_vec_augmented)

        index_granularity = np.where(new_dt_vec < 9e-4)
        if new_dt_vec[index_granularity].size != 0:
            raise Exception('Failed removing the small dt values')


        new_n_nodes = nodes_vec_augmented.shape[0]

# ======================================================================================================================
# ## which index of the augmented vector were added to base vector nodes
old_values = np.in1d(nodes_vec_augmented, nodes_vec)
new_indices = np.arange(len(nodes_vec_augmented))[~old_values]
base_indices = np.arange(len(nodes_vec_augmented))[old_values]

zip_indices_new = dict(zip(new_indices, indices_exceed))


ms = mat_storer.matStorer(f'{os.path.splitext(os.path.basename(__file__))[0]}.mat')

n_nodes = new_n_nodes - 1  # in new_n_nodes the last node is there already

plot_nodes = True
if plot_nodes:
    plt.figure()
    # nodes old
    plt.scatter(nodes_vec, np.zeros([nodes_vec.shape[0]]), edgecolors='blue', facecolor='none')
    plt.scatter(values_exceed, np.zeros([values_exceed.shape[0]]), edgecolors='red', facecolor='none')
    plt.scatter(nodes_vec[old_node_start_step], 0, marker='x', color='black')
    plt.vlines([nodes_vec[old_node_start_step], nodes_vec[old_node_end_step]],
               plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], linestyles='dashed', colors='k', linewidth=0.4)

    plt.scatter(nodes_vec[old_node_peak], 0, marker='x', color='black')
    plt.scatter(nodes_vec[old_node_end_step], 0, marker='x', color='black')

    plt.scatter(nodes_vec_augmented[node_start_step], 0, marker='x', color='green')
    plt.scatter(nodes_vec_augmented[node_peak], 0, marker='x', color='green')
    plt.scatter(nodes_vec_augmented[node_end_step], 0, marker='x', color='green')
    plt.vlines([nodes_vec_augmented[node_start_step], nodes_vec_augmented[node_end_step]],
               plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], linestyles='dashed', colors='r', linewidth=0.4)

    # for dim in range(6):
    #     plt.plot(nodes_vec_res[:-1], np.array(tau_sol_res[dim, :]))
    # for dim in range(6):
    #     plt.scatter(nodes_vec[:-1], np.array(prev_tau[dim, :]))
    # plt.title('tau on base')
    plt.show()


print('old_node_start_step', old_node_start_step)
print('old_node_end_step', old_node_end_step)
print('old_node_peak', old_node_peak)
print('node_start_step', node_start_step)
print('node_end_step', node_end_step)
print('node_peak', node_peak)

print('nodes_vec shape', nodes_vec.shape)
print('new_nodes_vec shape', nodes_vec_augmented.shape)

print('base_indices', base_indices)
print('new_indices', new_indices)
print('n_nodes', n_nodes)

# print('prev_dt_vec', prev_dt)
# print('new_dt_vec', new_dt_vec)
# ======================================================================================================================
# SET PROBLEM STATE AND INPUT VARIABLES
prb = problem.Problem(n_nodes)
q = prb.createStateVariable('q', n_q)
q_dot = prb.createStateVariable('q_dot', n_v)
q_ddot = prb.createInputVariable('q_ddot', n_v)

f_list = list()
for i in range(n_c):
    f_list.append(prb.createInputVariable(f'f{i}', n_f))

# SET CONTACTS MAP
contact_map = dict(zip(contacts_name, f_list))

# SET DYNAMICS
# dt = prb.createParameter("dt", 1)  # variable dt as input
dt = prb.createParameter("dt", 1, nodes=range(0, n_nodes))  # parameter dt as input
# dt = prb.createInputVariable("dt", 1)  # variable dt as input
# dt = 0.01
# Computing dynamics
x, x_dot = utils.double_integrator_with_floating_base(q, q_dot, q_ddot)
prb.setDynamics(x_dot)
prb.setDt(dt)
# SET BOUNDS
# q bounds
q_min = [-10., -10., -10., -1., -1., -1., -1.]  # floating base
q_min.extend(kindyn.q_min()[7:])
q_min = np.array(q_min)

q_max = [10., 10., 10., 1., 1., 1., 1.]  # floating base
q_max.extend(kindyn.q_max()[7:])
q_max = np.array(q_max)

q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0.0, 0.9, -1.5238505,
                   0.0, 0.9, -1.5202315,
                   0.0, 0.9, -1.5300265,
                   0.0, 0.9, -1.5253125])

# q_dot bounds
q_dot_lim = 100. * np.ones(n_v)
# q_ddot bounds
q_ddot_lim = 100. * np.ones(n_v)
# f bounds
f_lim = 10000. * np.ones(n_f)

# dt bounds
dt_min = 0.01  # [s]
dt_max = 0.1  # [s]

# set bounds and of q
q.setBounds(q_min, q_max)
q.setBounds(q_init, q_init, 0)
# set bounds of q_dot
q_dot_init = np.zeros(n_v)
q_dot.setBounds(-q_dot_lim, q_dot_lim)
q_dot.setBounds(q_dot_init, q_dot_init, 0)
# set bounds of q_ddot
q_ddot.setBounds(-q_ddot_lim, q_ddot_lim)
# set bounds of f
# for f in f_list:
#     f.setBounds(-f_lim, f_lim)

f_min = [-10000., -10000., -10.]
f_max = [10000., 10000., 10000.]
for f in f_list:
    f.setBounds(f_min, f_max)

# SET INITIAL GUESS
k = 0
for node in range(n_nodes+1):
    if node in base_indices:
        q.setInitialGuess(prev_q[:, k], node)
        k += 1
    if node in zip_indices_new.keys():
        q.setInitialGuess(q_res[:, zip_indices_new[node]], node)

k = 0
for node in range(n_nodes+1):
    if node in base_indices:
        q_dot.setInitialGuess(prev_q_dot[:, k], node)
        k += 1
    # if node in zip_indices_new.keys():
    #     q_dot.setInitialGuess(qdot_res[:, zip_indices_new[node]], node)

k = 0
for node in range(n_nodes):
    if node in base_indices:
        q_ddot.setInitialGuess(prev_q_ddot[:, k], node)
        k += 1
    # if node in zip_indices_new.keys():
    #     q_ddot.setInitialGuess(qddot_res[:, zip_indices_new[node]], node)

for i_f in range(len(f_list)):
    k = 0
    for node in range(n_nodes):
        if node in base_indices:
            f_list[i_f].setInitialGuess(prev_f_list[i_f][:, k], node)
            k += 1
        # if node in zip_indices_new.keys():
        #     f_list[i_f].setInitialGuess(f_res_list[i_f][:, zip_indices_new[node]], node)

plot_ig = False
if plot_ig:
    # ========================================================================================================
    plt.figure()

    for dim in range(q_res.shape[0]):
        plt.plot(nodes_vec_res, q_res[dim, :], '--')

    for dim in range(prev_q.shape[0]):
        plt.scatter(nodes_vec, prev_q[dim, :], color='red')

    q_to_print = q.getInitialGuess()
    q_to_print_matrix = np.reshape(q_to_print, (n_q, n_nodes + 1), order='F')

    for dim in range(q_to_print_matrix.shape[0]):
        plt.scatter(nodes_vec_augmented, q_to_print_matrix[dim, :], edgecolors='blue', facecolor='none')
    # ========================================================================================================
    plt.figure()

    for dim in range(qdot_res.shape[0]):
        plt.plot(nodes_vec_res, qdot_res[dim, :], '--')

    for dim in range(prev_q_dot.shape[0]):
        plt.scatter(nodes_vec, prev_q_dot[dim, :], color='red')

    q_dot_to_print = q_dot.getInitialGuess()
    q_dot_to_print_matrix = np.reshape(q_dot_to_print, (n_v, n_nodes + 1), order='F')

    for dim in range(q_dot_to_print_matrix.shape[0]):
        plt.scatter(nodes_vec_augmented, q_dot_to_print_matrix[dim, :], edgecolors='blue', facecolor='none')
    # ========================================================================================================
    plt.figure()

    for dim in range(qddot_res.shape[0]):
        plt.plot(nodes_vec_res[:-1], qddot_res[dim, :], '--')

    for dim in range(prev_q_ddot.shape[0]):
        plt.scatter(nodes_vec[:-1], prev_q_ddot[dim, :], color='red')

    q_ddot_to_print = q_ddot.getInitialGuess()
    q_ddot_to_print_matrix = np.reshape(q_ddot_to_print, (n_v, n_nodes), order='F')

    for dim in range(q_ddot_to_print_matrix.shape[0]):
        plt.scatter(nodes_vec_augmented[:-1], q_ddot_to_print_matrix[dim, :], edgecolors='blue', facecolor='none')
    # ========================================================================================================
    for i_f in range(len(f_list)):
        plt.figure()
        #
        for dim in range(f_res_list[i_f].shape[0]):
            plt.plot(nodes_vec_res[:-1], f_res_list[i_f][dim, :], '--')
        #
        for dim in range(prev_f_list[i_f].shape[0]):
            plt.scatter(nodes_vec[:-1], prev_f_list[i_f][dim, :], color='red')

        plt.plot(nodes_vec_res[:-1], f_res_list[i_f][2, :], '--')
        plt.scatter(nodes_vec[:-1], prev_f_list[i_f][2, :], color='red')
        f_to_print = f_list[i_f].getInitialGuess()
        f_to_print_matrix = np.reshape(f_to_print, (n_f, n_nodes), order='F')

        for dim in range(f_to_print_matrix.shape[0]):
            plt.scatter(nodes_vec_augmented[:-1], f_to_print_matrix[dim, :], edgecolors='blue', facecolor='none')

    # ========================================================================================================
    plt.show()

# SET TRANSCRIPTION METHOD
th = Transcriptor.make_method(transcription_method, prb, opts=transcription_opts)

# SET INVERSE DYNAMICS CONSTRAINTS
tau_lim = np.array([0., 0., 0., 0., 0., 0.,  # Floating base
                    1000., 1000., 1000.,  # Contact 1
                    1000., 1000., 1000.,  # Contact 2
                    1000., 1000., 1000.,  # Contact 3
                    1000., 1000., 1000.])  # Contact 4

tau = kin_dyn.InverseDynamics(kindyn, contact_map.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED).call(q, q_dot,
                                                                                                             q_ddot,
                                                                                                             contact_map)
prb.createIntermediateConstraint("inverse_dynamics", tau, bounds=dict(lb=-tau_lim, ub=tau_lim))

# SET FINAL VELOCITY CONSTRAINT
prb.createFinalConstraint('final_velocity', q_dot)

# SET CONTACT POSITION CONSTRAINTS
active_leg = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']

mu = 1
R = np.identity(3, dtype=float)  # environment rotation wrt inertial frame

# base_link stays at the same position
# FK = cs.Function.deserialize(kindyn.fk('base_link'))
# p_base = FK(q=q)['ee_pos']
# p_base_start = FK(q=q_init)['ee_pos']

# prb.createCost(f"base_link_pos", 1000*cs.sumsqr(p_base[0:2] - p_base_start[0:2]))

# DFK = cs.Function.deserialize(kindyn.frameVelocity('base_link', cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))
# v_base = DFK(q=q, qdot=q_dot)['ee_vel_linear']
# 2. velocity of each base_link must be zero
# prb.createConstraint(f"base_link_vel_before_step", v_base, nodes=range(0, node_start_step))
# prb.createConstraint(f"base_link_vel_after_step", v_base, nodes=range(node_end_step, n_nodes + 1))
# COM = cs.Function.deserialize(kindyn.centerOfMass())
# p_com = COM(q=q_init)['com']
# exit()
fb_during_jump = np.array([q_init[0], q_init[1], q_init[2] + jump_height, 0.0, 0.0, 0.0, 1.0])
q_final = q_init

for frame, f in contact_map.items():
    # 2. velocity of each end effector must be zero
    FK = cs.Function.deserialize(kindyn.fk(frame))
    p = FK(q=q)['ee_pos']
    p_start = FK(q=q_init)['ee_pos']
    p_goal = p_start + [0., 0., jump_height]
    DFK = cs.Function.deserialize(kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))
    v = DFK(q=q, qdot=q_dot)['ee_vel_linear']
    DDFK = cs.Function.deserialize(kindyn.frameAcceleration(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))
    a = DDFK(q=q, qdot=q_dot)['ee_acc_linear']

    prb.createConstraint(f"{frame}_vel_before_lift", v, nodes=range(0, node_start_step))
    prb.createConstraint(f"{frame}_vel_after_lift", v, nodes=range(node_end_step, n_nodes + 1))

    # friction cones must be satisfied
    fc, fc_lb, fc_ub = kin_dyn.linearized_friction_cone(f, mu, R)
    prb.createIntermediateConstraint(f"{frame}_fc_before_lift", fc, nodes=range(0, node_start_step), bounds=dict(lb=fc_lb, ub=fc_ub))
    prb.createIntermediateConstraint(f"{frame}_fc_after_lift", fc, nodes=range(node_end_step, n_nodes), bounds=dict(lb=fc_lb, ub=fc_ub))

    prb.createConstraint(f"{frame}_no_force_during_lift", f, nodes=range(node_start_step, node_end_step))

    prb.createConstraint(f"lift_{frame}_leg", p - p_goal, nodes=node_peak)
    prb.createConstraint(f"land_{frame}_leg", p - p_start, nodes=node_end_step)


# SET COST FUNCTIONS
# prb.createCost(f"jump_fb", 10000 * cs.sumsqr(q[2] - fb_during_jump[2]), nodes=node_start_step)
prb.createCost("min_q_dot", 1. * cs.sumsqr(q_dot))
# prb.createFinalCost(f"final_nominal_pos", 1000 * cs.sumsqr(q - q_init))
for f in f_list:
    prb.createIntermediateCost(f"min_{f.getName()}", 0.01 * cs.sumsqr(f)) #0.01

prb.createIntermediateCost("min_qddot", 1 * cs.sumsqr(q_ddot)) #1

######################################## proximal auxiliary cost function #######################################
weight = 1e9
k = 0
for node in range(n_nodes):
    if node in base_indices:
        prb.createCost(f"q_close_to_old_node_{node}", weight * cs.sumsqr(q - prev_q[:, k]), nodes=node)
        k = k+1
    if node in zip_indices_new.keys():
        prb.createCost(f"q_close_to_res_node_{node}", weight * cs.sumsqr(q - q_res[:, zip_indices_new[node]]), nodes=node)

k = 0
for node in range(n_nodes):
    if node in base_indices:
        prb.createCost(f"qdot_close_to_old_node_{node}", weight * cs.sumsqr(q_dot - prev_q_dot[:, k]), nodes=node)
        k = k+1
    if node in zip_indices_new.keys():
        prb.createCost(f"qdot_close_to_res_node_{node}", weight * cs.sumsqr(q_dot - qdot_res[:, zip_indices_new[node]]), nodes=node)

# =============
# SOLVE PROBLEM
# =============
opts = {'ipopt.tol': 0.01,
        'ipopt.constr_viol_tol': 0.01,
        'ipopt.max_iter': 4000,
        'ipopt.linear_solver': 'ma57'}

for i in range(len(new_dt_vec)):
    dt.assign(new_dt_vec[i], nodes=i)

sol = Solver.make_solver('ipopt', prb, opts)
tic = time.time()
sol.solve()
toc = time.time() - tic
print('time elapsed solving:', toc)

solution = sol.getSolutionDict()
solution_constraints = sol.getConstraintSolutionDict()

solution_constraints_dict = dict()
for name, item in prb.getConstraints().items():
    lb, ub = item.getBounds()
    lb_mat = np.reshape(lb, (item.getDim(), len(item.getNodes())), order='F')
    ub_mat = np.reshape(ub, (item.getDim(), len(item.getNodes())), order='F')
    solution_constraints_dict[name] = dict(val=solution_constraints[name], lb=lb_mat, ub=ub_mat, nodes=item.getNodes())

# ================================== resample refined trajectory ==================================
f_list = list()
for i in range(n_c):
    f_list.append(solution[f'f{i}'])
contact_map = dict(zip(contacts_name, f_list))

q_sym = cs.SX.sym('q', n_q)
q_dot_sym = cs.SX.sym('q_dot', n_v)
q_ddot_sym = cs.SX.sym('q_ddot', n_v)
x_dot = utils.double_integrator_with_floating_base(q_sym, q_dot_sym, q_ddot_sym)
x = cs.vertcat(q_sym, q_dot_sym)

dae = {'x': x, 'p': q_ddot_sym, 'ode': x_dot, 'quad': 1}
q_res, qdot_res, qddot_res, contact_map_res, tau_sol_res = resampler_trajectory.resample_torques(
    solution['q'], solution['q_dot'], solution['q_ddot'], new_dt_vec, dt_res, dae, contact_map,
    kindyn,
    cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

f_res_list = list()
for f in prev_f_list:
    f_res_list.append(resampler_trajectory.resample_input(f, new_dt_vec, dt_res))

solution_resampled = dict()
solution_resampled['q_res'] = q_res
solution_resampled['tau_sol_res'] = tau_sol_res
solution_resampled['f_res_list'] = f_res_list

from horizon.variables import Variable, SingleVariable, Parameter, SingleParameter

info_dict = dict(n_nodes=n_nodes, times=nodes_vec_augmented, node_start_step=node_start_step, node_end_step=node_end_step, node_peak=node_peak, jump_height=jump_height)
if isinstance(dt, Variable) or isinstance(dt, SingleVariable):
    ms.store({**solution, **solution_constraints_dict, **solution_resampled, **info_dict})
elif isinstance(dt, Parameter) or isinstance(dt, SingleParameter):
    dt_dict = dict(param_dt=new_dt_vec)
    ms.store({**solution, **solution_constraints_dict, **solution_resampled, **info_dict, **dt_dict})
else:
    dt_dict = dict(constant_dt=dt)
    ms.store({**solution, **solution_constraints_dict, **solution_resampled, **info_dict, **dt_dict})




#!/usr/bin/env python3
from horizon import problem
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter, mat_storer
from horizon.transcriptions.transcriptor import Transcriptor
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from horizon.solvers import solver
import os, argparse
from itertools import filterfalse
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time

# this file uses data_spawner.mat
path_to_examples = os.path.abspath(__file__ + "/../../../examples")
urdffile = os.path.join(path_to_examples, 'urdf', 'spot.urdf')
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)


n_c = 4
n_q = kindyn.nq()
n_v = kindyn.nv()
n_f = 3

ms = mat_storer.matStorer(f'/home/francesco/hhcm_workspace/src/horizon/horizon/utils/resampler_playground/data_spawner.mat')
solution = ms.load()

contacts_name = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
contact_map = {contact: solution[f'force_{contact}'] for contact in contacts_name}

n_nodes = solution['n_nodes'][0][0]
dt = solution['dt'][0]


cumulative_dt = np.zeros(len(dt) + 1)
for i in range(len(dt)):
    cumulative_dt[i + 1] = dt[i] + cumulative_dt[i]


# COMPUTING TAU from FORCES AND Q

fmap = dict()
for contact in contacts_name:
    fmap[contact] = solution[f'force_{contact}']
id_fn = kin_dyn.InverseDynamics(kindyn, fmap.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)


tau_dim = n_q - 1
tau_sol = np.zeros([tau_dim, n_nodes])
for i in range(n_nodes):
    fmap_i = dict()
    for frame, wrench in fmap.items():
        fmap_i[frame] = wrench[:, i]
    tau_i = id_fn.call(solution['q'][:, i], solution['q_dot'][:, i], solution['q_ddot'][:, i], fmap_i)
    tau_sol[:, i] = tau_i.toarray().flatten()

solution['tau'] = tau_sol
# ==================================================================================================
# ALL OF THIS IS JUST FOR THE RESAMPLER. THIS IS CRAZY =============================================
# ==================================================================================================
# define the resampled dt
dt_res = 0.55
print(f'resampling from {dt[0]} to {dt_res}')

# define the dynamics of the system for the integrator
q = cs.SX.sym('q', n_q)
q_dot = cs.SX.sym('q_dot', n_v)
q_ddot = cs.SX.sym('q_dot', n_v)

f_list = []
for i in range(n_c):
    f_list.append(cs.SX.sym(f'f{i}', n_f))

u = cs.vertcat(q_ddot, f_list[0], f_list[1], f_list[2], f_list[3])

x = cs.vertcat(q, q_dot)
x_dot = utils.double_integrator_with_floating_base(q, q_dot, q_ddot)

u_res = resampler_trajectory.resample_input(solution['u_opt'], dt, dt_res)
n_nodes_res = u_res.shape[1]

dae = {'x': x, 'p': u, 'ode': x_dot, 'quad': 1}
print('elapsed_time:')
tic = time.time()
x_res1 = resampler_trajectory.resampler_old(solution['x_opt'], solution['u_opt'], dt, dt_res, dae=dae, f_int=None)
elapsed_time = time.time() - tic
print('old method:', elapsed_time)

tic = time.time()
x_res = resampler_trajectory.resampler(solution['x_opt'], solution['u_opt'], dt, dt_res, dae=dae, f_int=None)
elapsed_time = time.time() - tic
print('new method:', elapsed_time)

print(np.isclose(x_res, x_res1).all())

dt_res = dt_res * np.ones(n_nodes_res)
solution['dt_res'] = dt_res
solution['x_opt_res'] = x_res
solution['u_opt_res'] = u_res

solution['q_res'] = solution['x_opt_res'][:n_q, :]
solution['q_dot_res'] = solution['x_opt_res'][n_q:, :]
solution['q_ddot_res'] = solution['u_opt_res'][:n_v, :]

i = n_v
for contact in contacts_name:
    solution[f'force_{contact}_res'] = solution['u_opt_res'][i:i+3, :]
    i += 3

cumulative_dt_res = np.zeros(len(dt_res) + 1)
for i in range(len(dt_res)):
    cumulative_dt_res[i + 1] = dt_res[i] + cumulative_dt_res[i]

fmap_res = dict()
for contact in contacts_name:
    fmap_res[contact] = solution[f'force_{contact}_res']

tau_sol_res = np.zeros([tau_dim, n_nodes_res])
for i in range(n_nodes_res):
    fmap_res_i = dict()
    for frame, wrench in fmap_res.items():
        fmap_res_i[frame] = wrench[:, i]
    tau_res_i = id_fn.call(solution['q_res'][:, i], solution['q_dot_res'][:, i], solution['q_ddot_res'][:, i], fmap_res_i)
    tau_sol_res[:, i] = tau_res_i.toarray().flatten()

solution['tau_res'] = tau_sol_res
# ==================================================================================================
# ==================================================================================================
plt.figure()
#
range_plot_q = [9] # range(solution['q'].shape[0])
for dim in range_plot_q:
    # old solution
    plt.scatter(cumulative_dt, solution['q'][dim, :])
    # plt.scatter(cumulative_dt[-1], solution['q'][dim, -1], c='r')
    # resampled solution
    plt.scatter(cumulative_dt_res, solution['q_res'][dim, :], s=4)
    # plt.scatter(cumulative_dt_res[-1], solution['q_res'][dim, -1], s=4, c='b')

plt.show()
exit()
# for contact in contacts_name:
#     plt.figure()
#     plt.scatter(cumulative_dt[:-1], solution[f'force_{contact}'][2, :])
#     plt.plot(cumulative_dt_res[:-1], solution[f'force_{contact}_res'][2, :])
#
# plt.show()
plt.figure()
range_plot_tau = range(6)
for dim in range_plot_tau:
    plt.scatter(cumulative_dt[:-1], solution[f'tau'][dim, :])
    plt.plot(cumulative_dt_res[:-1], solution[f'tau_res'][dim, :])

plt.show()

exit()

pos_contact_list = list()
fig = plt.figure()
fig.suptitle('Contacts')
gs = gridspec.GridSpec(2, 2)
i = 0
for contact in contacts_name:
    ax = fig.add_subplot(gs[i])
    ax.set_title('{}'.format(contact))
    i += 1
    FK = kindyn.fk(contact)
    pos = FK(q=solution['q'])['ee_pos'].toarray()

    for dim in range(3):
        ax.plot(cumulative_dt, pos[dim, :])

plt.figure()
for contact in contacts_name:
    FK = kindyn.fk(contact)
    pos = FK(q=solution['q'])['ee_pos']

    plt.title(f'feet position - plane_xy')
    plt.scatter(np.array(pos[0, :]), np.array(pos[1, :]), linewidth=0.1)

plt.figure()
for contact in contacts_name:
    FK = kindyn.fk(contact)
    pos = FK(q=solution['q'])['ee_pos']

    plt.title(f'feet position - plane_xz')
    plt.scatter(np.array(pos[0, :]), np.array(pos[2, :]), linewidth=0.1)

plt.show()
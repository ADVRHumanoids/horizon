from horizon import problem
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.solvers import Solver
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter, mat_storer
import os
from horizon.ros.replay_trajectory import *
import numpy as np
import matplotlib.pyplot as plt

transcription_method = 'multiple_shooting'  # direct_collocation
transcription_opts = dict(integrator='RK4')

path_to_examples = '../../../examples/'
urdffile = path_to_examples + 'urdf/spot.urdf'
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

n_c = 4
n_q = kindyn.nq()
n_v = kindyn.nv()
n_f = 3

ms = mat_storer.matStorer('spot_jump.mat')
prev_solution = ms.load()

n_nodes = prev_solution['n_nodes'][0][0]

node_action = prev_solution['node_action'][0]
stance_orientation = prev_solution['stance_orientation'][0]

prev_q = prev_solution['q']
prev_q_dot = prev_solution['q_dot']
prev_q_ddot = prev_solution['q_ddot']

contacts_name = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
prev_f_list = [prev_solution[f'force_{i}'] for i in contacts_name]

prev_tau = prev_solution['dynamic_feasibility']['val'][0][0]
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


current_path = os.path.abspath(__file__ + '/..')

ms = mat_storer.matStorer(current_path + '/refining_jump.mat') # _fourth_cycle
solution_refined = ms.load()
nodes_vec_refined = solution_refined['times'][0]


tau_sol_base = tau_sol_res[:6, :]
threshold = 2.8
## get index of values greater than a given threshold for each dimension of the vector, and remove all the duplicate values (given by the fact that there are more dimensions)
indices_exceed = np.unique(np.argwhere(np.abs(tau_sol_base) > threshold)[:, 1])
# these indices corresponds to some nodes ..
values_exceed = nodes_vec_res[indices_exceed]

## search for duplicates and remove them, both in indices_exceed and values_exceed
indices_duplicates = np.where(np.in1d(values_exceed, nodes_vec))
value_duplicates = values_exceed[indices_duplicates]

values_exceed = np.delete(values_exceed, np.where(np.in1d(values_exceed, value_duplicates)))
indices_exceed = np.delete(indices_exceed, indices_duplicates)

added_nodes_lims = [1.53, 1.62]
# ======================================================================================================================
# ======================================================================================================================

save_path = '/home/francesco/Documents/all_of_horizon/plots/mesh_refiner/'
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

w = 7195
h = 3841
fig_size = [19.20, 10.80]

plt.rcParams['text.usetex'] = True
plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

aspect_ratio = 0.15
gs = gridspec.GridSpec(3, 1)
# gs.hspace = -0.6

fig = plt.figure(frameon=True)
fig.set_size_inches(fig_size[0], fig_size[1])
fig.tight_layout()

ax = fig.add_subplot(gs[0])
for dim in range(6):
    ax.plot(nodes_vec_res[:-1], np.array(tau_sol_res[dim, :]), linewidth=3)
    ax.scatter(nodes_vec[:-1], np.array(prev_tau[dim, :]), s=30, facecolors='none', edgecolors='#d62728', zorder=3)

linewidth_thresholds = 1.5
transparency_thresholds = 0.3
plt.axhline(y = threshold, color = 'r', linestyle = '--', linewidth=linewidth_thresholds, alpha=transparency_thresholds)
plt.axhline(y = - threshold, color = 'r', linestyle = '--', linewidth=linewidth_thresholds, alpha=transparency_thresholds)

plt.axvline(x = added_nodes_lims[0], color = 'b', linestyle = '--', linewidth=linewidth_thresholds, alpha=transparency_thresholds)
plt.axvline(x = added_nodes_lims[1], color = 'b', linestyle = '--', linewidth=linewidth_thresholds, alpha=transparency_thresholds)

ax.set_xticks(nodes_vec[:-1])
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.grid(alpha=0.4)

ax.yaxis.set_major_locator(plt.MultipleLocator(4))

ax.set_xticklabels([])
plt.subplots_adjust(wspace=0, hspace=0)


plt.xlim([0, nodes_vec[-1]])
plt.ylim([-8, 8])

# ax.set_xlabel(r'time [s]')
ax.set_ylabel(r'effort [N]')
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*aspect_ratio)
# =============================================================
# =============== after mesh refinement =======================
# =============================================================
ax = fig.add_subplot(gs[2])
tau_ref = solution_refined['dynamic_feasibility']['val'][0][0]
for dim in range(6):
    ax.plot(nodes_vec_refined[:-1], np.array(tau_ref[dim, :]), linewidth=3)
for dim in range(6):
    ax.scatter(nodes_vec[:-1], np.array(prev_tau[dim, :]), s=30, facecolors='none', edgecolors='#d62728', zorder=3)
    plt.scatter(values_exceed, np.zeros([values_exceed.shape[0]]), marker='|', zorder=4, c='blue')
plt.xlim([added_nodes_lims[0] - 0.05, added_nodes_lims[1] + 0.05])
plt.ylim([-8, 8])
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*0.08)
ax.set_xlabel(r'time [s]')

tau_sol_ref_res = solution_refined['tau_sol_res']

dt_res = 0.001
num_samples = tau_sol_ref_res.shape[1]
nodes_vec_res = np.zeros([num_samples + 1])
for i in range(1, num_samples + 1):
    nodes_vec_res[i] = nodes_vec_res[i - 1] + dt_res

ax = fig.add_subplot(gs[1])
tau_ref = solution_refined['dynamic_feasibility']['val'][0][0]
for dim in range(6):
    ax.plot(nodes_vec_res[:-1], np.array(tau_sol_ref_res[dim, :]), linewidth=3)
for dim in range(6):
    ax.scatter(nodes_vec[:-1], np.array(prev_tau[dim, :]), s=30, facecolors='none', edgecolors='#d62728', zorder=3)
    plt.scatter(values_exceed, np.zeros([values_exceed.shape[0]]), marker='|', zorder=4, c='blue')

plt.axhline(y = threshold, color = 'r', linestyle = '--', linewidth=linewidth_thresholds)
plt.axhline(y = - threshold, color = 'r', linestyle = '--', linewidth=linewidth_thresholds)

plt.axvline(x = added_nodes_lims[0], color = 'b', linestyle = '--', linewidth=linewidth_thresholds)
plt.axvline(x = added_nodes_lims[1], color = 'b', linestyle = '--', linewidth=linewidth_thresholds)

ax.set_xticks(nodes_vec[:-1])
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.grid(alpha=0.4)

ax.yaxis.set_major_locator(plt.MultipleLocator(4))

wanted_time_label = {0, 20, 40, 49}
label_list = list(range(nodes_vec[:-1].shape[0]))
label_list = [e for e in label_list if e not in wanted_time_label]
xticks = ax.xaxis.get_major_ticks()
for i_hide in label_list:
    xticks[i_hide].label1.set_visible(False)

plt.xlim([0, nodes_vec[-1]])
plt.ylim([-8, 8])

ax.set_xlabel(r'time [s]')
ax.set_ylabel(r'effort [N]')
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*aspect_ratio)

# plt.show()

plt.savefig(save_path + "/tau_on_bases", dpi=500, bbox_inches='tight')

# contacts_name = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
# contact_map = dict(zip(contacts_name, [prev_solution['f0'], prev_solution['f1'], prev_solution['f2'], prev_solution['f3']]))

# f_list = ['f0', 'f1', 'f2', 'f3']
# for f_ind in f_list:
#     plt.figure()
#     for dim in range(solution_refined[f_ind].shape[0]):
#         plt.plot(nodes_vec_refined[:-1], np.array(solution_refined[f_ind][dim, :]), '--')
#
#     for dim in range(prev_solution[f_ind].shape[0]):
#         plt.plot(nodes_vec[:-1], prev_solution[f_ind][dim, :])
#
#     plt.title(f_ind)


fig, ax = plt.subplots()
fig.set_size_inches(fig_size[0], fig_size[1])
# fig.tight_layout()

ax.set_xlabel(r'time [s]')
ax.set_ylabel(r'q [rad]')


for dim in range(solution_refined['q'].shape[0]):
    plt.plot(nodes_vec_refined, np.array(solution_refined['q'][dim, :]), '--', color='#ff7f0e', linewidth=3) #  dashes=(15, 8)
             # marker='o',markersize=4,markerfacecolor='none',markeredgecolor='black',markeredgewidth=0.5,linewidth=3)

for dim in range(prev_solution['q'].shape[0]):
    plt.scatter(nodes_vec, np.array(prev_solution['q'][dim, :]), color='#1f77b4', zorder=3, s=30) #facecolors='none'

ax.set_xticks(nodes_vec[:-1])
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.grid(alpha=0.4)
plt.xlim([0, nodes_vec[-1]])

wanted_time_label = {0, 20, 40, 49}
label_list = list(range(nodes_vec[:-1].shape[0]))
label_list = [e for e in label_list if e not in wanted_time_label]
xticks = ax.xaxis.get_major_ticks()
for i_hide in label_list:
    xticks[i_hide].label1.set_visible(False)

aspect_ratio = 0.2
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*aspect_ratio)

plt.savefig(save_path + "/q_refined", dpi=500, bbox_inches='tight')

plt.show()

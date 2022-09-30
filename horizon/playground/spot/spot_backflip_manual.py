import casadi as cs
from horizon.utils import utils, kin_dyn, mat_storer
import numpy as np
import time
import os
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
import horizon.transcriptions.integrators as integ


urdffile = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../urdf', 'spot.urdf')
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

n_c = 4
n_q = kindyn.nq()
n_v = kindyn.nv()
n_f = 3

t_tot = 2.5
N = 100


t_jump = 1.
t_land = 2.

dt = t_tot/N

node_start_step = int(t_jump/dt)
node_end_step = int(t_land/dt)


N_control = N
N_states = N + 1
# ====================
# unraveled dimensions
# ====================
q_dim = N_states * n_q
q_dot_dim = N_states * n_v
q_ddot_dim = N_control * n_v
f_dim = N_control * n_c * n_f
# dt_dim = N_control * 1

q = cs.SX.sym('q', n_q)
q_dot = cs.SX.sym('q_dot', n_v)
q_ddot = cs.SX.sym('q_ddot', n_v)

# forces
f_list = list()
for i in range(n_c):
    f_list.append(cs.SX.sym(f'f{i}', n_f))

fs = cs.vertcat(*f_list)
state = cs.vertcat(q, q_dot)
input = cs.vertcat(q_ddot, fs)

x_dot = utils.double_integrator_with_floating_base(q, q_dot, q_ddot)
dae = {'x': state, 'p': input, 'ode': x_dot, 'quad': 1}
integrator = integ.RK4(dae, opts=dict(tf=dt))

int_map = integrator.map(N_control, 'thread', 15)

q_i = cs.MX.sym("q_i", n_q, N_states)
q_dot_i = cs.MX.sym("q_dot_i", n_v, N_states)

q_ddot_i = cs.MX.sym("q_ddot_i", n_v, N_control)
f_i = cs.MX.sym('f_i', n_f * n_c, N_control)
# dt_i = cs.MX.sym('dt_i', 1, N_control)

X = cs.vertcat(q_i, q_dot_i)
U = cs.vertcat(q_ddot_i, f_i)

X_int = int_map(X[:, :N], U) #dt_i because it's N+1
# starting from node 1
g_multi_shoot = X_int[0] - X[:, 1:]

q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0.0, 0.9, -1.5238505,
                   0.0, 0.9, -1.5202315,
                   0.0, 0.9, -1.5300265,
                   0.0, 0.9, -1.5253125])

# =====================================================================================
# =====================================================================================
#
mu = 1
R = np.identity(3, dtype=float)  # environment rotation wrt inertial frame

g_v_list = list()
g_no_force_list = list()
g_land_list = list()

contact_map_i = dict(lf_foot=f_i[0:3, :],
                     rf_foot=f_i[3:6, :],
                     lh_foot=f_i[6:9, :],
                     rh_foot=f_i[9:12, :])

id_fn_i = kin_dyn.InverseDynamicsMap(N, kindyn, contact_map_i.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
g_tau_i = id_fn_i.call(q_i[:, :N], q_dot_i[:, :N], q_ddot_i, contact_map_i)

active_leg = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']

for frame, f in contact_map_i.items():
    FK = cs.Function.deserialize(kindyn.fk(frame))
    DFK = cs.Function.deserialize(kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))

    v = DFK(q=q_i, qdot=q_dot_i)['ee_vel_linear']
    g_v_list.append(v[:, :node_start_step])
    g_v_list.append(v[:, node_end_step:])


    g_no_force_list.append(f[:, node_start_step:node_end_step])


g_land_list.append(q_i[:6, N] - q_init[:6])
# FK = cs.Function.deserialize(kindyn.fk('base_link'))
# p_base = FK(q=q_i)['ee_pos']
# p_base_init = FK(q=q_init)['ee_pos']
# g_land_list.append(p_base[:, N] - p_base_init)
# g_land_list.append(q_i[:, N] - q_init)

# g_land_list.append(q - p_base_init)
# g_land = p_i[:, 90] - p_start
# prb.createConstraint(f"land_{frame}_leg", p - p_start, nodes=node_end_step + 2)

# prb.createConstraint(f"lift_{frame}_leg", p - p_goal, nodes=25)
# prb.createConstraint(f"land_{frame}_leg", p - p_start, nodes=node_end_step + 2)

f_list = list()
# f_list.append(1000 * cs.sumsqr(q_i - q_init))
# f_list.append(1e-6 * cs.sumsqr(f_i))
f_list.append(1e-6 * cs.sumsqr(f_i[0:3, :]))
f_list.append(1e-6 * cs.sumsqr(f_i[3:6, :]))
f_list.append(1e-6 * cs.sumsqr(f_i[6:9, :]))
f_list.append(1e-6 * cs.sumsqr(f_i[9:12, :]))
#
f_list.append(1e-2 * cs.sumsqr(q_i[7:, :] - q_init[7:]))
# f_list.append(1e-3 * cs.sumsqr(q_dot_i))
f_list.append(1e-6 * cs.sumsqr(q_ddot_i))

f_list.append(1e2 * cs.sumsqr(q_i[7:, N] - q_init[7:]))

f_list.append(10 * cs.sumsqr(q_i[3:6] - q_init[3:6]))
# for frame, f in contact_map.items():
#     FK = cs.Function.deserialize(kindyn.fk(frame))
#     p_ee = FK(q=q_i)['ee_pos']
#     p_ee_init = FK(q=q_init)['ee_pos']
#     f_list.append(1e6 * cs.sumsqr(p_ee[2, N] - p_ee_init[2]))
# f_list.append(1 * cs.sumsqr(q_ddot_i))
# =====================================================================================
# =====================================================================================
# f_list.append(cs.sumsqr(q_base))
# ====================
# setting initial condition
# ====================
q0 = np.zeros([n_q, N_states])
q_dot0 = np.zeros([n_v, N_states])
q_ddot0 = np.zeros([n_v, N_control])
f0 = np.zeros([n_c * n_f, N_control])
# dt0 = 0.01 * np.ones([1, N_control])
# ====================
# lower bound variables
# ====================
q_lbw = -np.inf * np.ones([n_q, N_states])
q_dot_lbw = -np.inf * np.ones([n_v, N_states])
q_ddot_lbw = -np.inf * np.ones([n_v, N_control])
f_lbw = -np.inf * np.ones([n_c * n_f, N_control])

# ====================
# upper bound variables
# ====================
q_ubw = np.inf * np.ones([n_q, N_states])
q_dot_ubw = np.inf * np.ones([n_v, N_states])
q_ddot_ubw = np.inf * np.ones([n_v, N_control])
f_ubw = np.inf * np.ones([n_c * n_f, N_control])

# ===============================
# ==== BOUNDS INITIALIZATION ====
# ===============================
# initial guess q
q0[:, 0] = q_init
f0[2, 0] = 55
f0[5, 0] = 55
f0[8, 0] = 55
f0[11, 0] = 55


# initial q
q_lbw[:, 0] = q_init
q_ubw[:, 0] = q_init

# zero initial velocity
q_dot_lbw[:, 0] = np.zeros([1, n_v])
q_dot_ubw[:, 0] = np.zeros([1, n_v])

# zero final velocity
q_dot_lbw[:, N] = np.zeros([1, n_v])
q_dot_ubw[:, N] = np.zeros([1, n_v])

# q bounds
# q_min = [-10., -10., -10., -1., -1., -1., -1.]  # floating base
# q_min.extend(kindyn.q_min()[7:])
# q_min = np.array(q_min)
#
# q_max = [10., 10., 10., 1., 1., 1., 1.]  # floating base
# q_max.extend(kindyn.q_max()[7:])
# q_max = np.array(q_max)
#
# q_lbw[:, :] = np.tile(q_min, (N_states, 1)).T
# q_ubw[:, :] = np.tile(q_max, (N_states, 1)).T

# z component of f cannot go lower than 0
f_lbw[2, :] = np.zeros(f_lbw[2, :].shape)
f_lbw[5, :] = np.zeros(f_lbw[5, :].shape)
f_lbw[8, :] = np.zeros(f_lbw[8, :].shape)
f_lbw[11, :] = np.zeros(f_lbw[11, :].shape)

# dt bounds
# dt_min = 0.001
# dt_max = 0.01
# dt_lbw = np.tile(dt_min, (N_control, 1)).T
# dt_ubw = np.tile(dt_max, (N_control, 1)).T

# ======================================
tau_lbg = np.zeros(g_tau_i[:6, :].shape)
tau_ubg = np.zeros(g_tau_i[:6, :].shape)

# ======================================
multi_shoot_lbg = np.zeros(g_multi_shoot.shape)
multi_shoot_ubg = np.zeros(g_multi_shoot.shape)

# ======================================
# g_lift_lbw = list()
# for elem in g_lift_list:
#     g_lift_lbw.append(np.zeros(elem.shape))
#
# g_lift_ubw = list()
# for elem in g_lift_list:
#     g_lift_ubw.append(np.zeros(elem.shape))

# ======================================
v_lbg = list()
for elem in g_v_list:
    v_lbg.append(np.zeros(elem.shape))

v_ubg = list()
for elem in g_v_list:
    v_ubg.append(np.zeros(elem.shape))

# ======================================
# g_fc_lbw = list()
# for elem in g_fc_dict:
#     g_fc_lbw.append(-np.inf * np.ones(elem.shape))
#
# g_fc_ubw = list()
# for elem in g_fc_dict:
#     g_fc_ubw.append(np.zeros(elem.shape))

# ======================================
no_force_lbg = list()
for elem in g_no_force_list:
    no_force_lbg.append(np.zeros(elem.shape))

no_force_ubg = list()
for elem in g_no_force_list:
    no_force_ubg.append(np.zeros(elem.shape))

# ======================================
# q_land
land_lbg = list()
for elem in g_land_list:
    land_lbg.append(np.zeros(elem.shape))

land_ubg = list()
for elem in g_land_list:
    land_ubg.append(np.zeros(elem.shape))

# ======================================================================================================================
# reshape stuff ========================================================================================================
# ======================================================================================================================

# VARIABLES
q0_flat = np.reshape(q0, [1, q_dim], order='F')
q_dot0_flat = np.reshape(q_dot0, [1, q_dot_dim], order='F')
q_ddot0_flat = np.reshape(q_ddot0, [1, q_ddot_dim], order='F')
f0_flat = np.reshape(f0, [1, f_dim], order='F')
# dt0_flat = np.reshape(dt0, [1, dt_dim], order='F')
w0 = np.concatenate((q0_flat, q_dot0_flat, q_ddot0_flat, f0_flat), axis=1) #dt0_flat

q_lbw_flat = np.reshape(q_lbw, [1, q_dim], order='F')
q_dot_lbw_flat = np.reshape(q_dot_lbw, [1, q_dot_dim], order='F')
q_ddot_lbw_flat = np.reshape(q_ddot_lbw, [1, q_ddot_dim], order='F')
f_lbw_flat = np.reshape(f_lbw, [1, f_dim], order='F')
# dt_lbw_flat = np.reshape(dt_lbw, [1, dt_dim], order='F')
lbw = np.concatenate((q_lbw_flat, q_dot_lbw_flat, q_ddot_lbw_flat, f_lbw_flat), axis=1) #dt_ubw_flat

q_ubw_flat = np.reshape(q_ubw, [1, q_dim], order='F')
q_dot_ubw_flat = np.reshape(q_dot_ubw, [1, q_dot_dim], order='F')
q_ddot_ubw_flat = np.reshape(q_ddot_ubw, [1, q_ddot_dim], order='F')
f_ubw_flat = np.reshape(f_ubw, [1, f_dim], order='F')
# dt_ubw_flat = np.reshape(dt_ubw, [1, dt_dim], order='F')
ubw = np.concatenate((q_ubw_flat, q_dot_ubw_flat, q_ddot_ubw_flat, f_ubw_flat), axis=1) #dt_ubw_flat

tau_lbg_flat = np.reshape(tau_lbg, [1, g_tau_i[:6, :].shape[0] * g_tau_i[:6, :].shape[1]], order='F')
tau_ubg_flat = np.reshape(tau_ubg, [1, g_tau_i[:6, :].shape[0] * g_tau_i[:6, :].shape[1]], order='F')

# CONSTRAINTS
multi_shoot_lbg_flat = np.reshape(multi_shoot_lbg, [1, g_multi_shoot.shape[0] * g_multi_shoot.shape[1]], order='F')
multi_shoot_ubg_flat = np.reshape(multi_shoot_ubg, [1, g_multi_shoot.shape[0] * g_multi_shoot.shape[1]], order='F')

# lift_lbg_flat = list()
# for elem in g_lift_lbw:
#     lift_lbg_flat.append(np.reshape(elem, [1, elem.shape[0] * elem.shape[1]], order='F'))
#
# lift_ubg_flat = list()
# for elem in g_lift_ubw:
#     lift_ubg_flat.append(np.reshape(elem, [1, elem.shape[0] * elem.shape[1]], order='F'))

v_lbg_flat = list()
for elem in v_lbg:
    v_lbg_flat.append(np.reshape(elem, [1, elem.shape[0] * elem.shape[1]], order='F'))

v_ubg_flat = list()
for elem in v_ubg:
    v_ubg_flat.append(np.reshape(elem, [1, elem.shape[0] * elem.shape[1]], order='F'))

# fc_lbg_flat = list()
# for elem in g_fc_lbw:
#     fc_lbg_flat.append(np.reshape(elem, [1, elem.shape[0] * elem.shape[1]], order='F'))
#
# fc_ubg_flat = list()
# for elem in g_fc_ubw:
#     fc_ubg_flat.append(np.reshape(elem, [1, elem.shape[0] * elem.shape[1]], order='F'))

no_force_lbg_flat = list()
for elem in no_force_lbg:
    no_force_lbg_flat.append(np.reshape(elem, [1, elem.shape[0] * elem.shape[1]], order='F'))

no_force_ubg_flat = list()
for elem in no_force_ubg:
    no_force_ubg_flat.append(np.reshape(elem, [1, elem.shape[0] * elem.shape[1]], order='F'))

land_lbg_flat = list()
for elem in land_lbg:
    land_lbg_flat.append(np.reshape(elem, [1, elem.shape[0] * elem.shape[1]], order='F'))

land_ubg_flat = list()
for elem in land_ubg:
    land_ubg_flat.append(np.reshape(elem, [1, elem.shape[0] * elem.shape[1]], order='F'))
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


w = cs.veccat(q_i, q_dot_i, q_ddot_i, f_i) # dt_i
g = cs.veccat(g_multi_shoot, g_tau_i[:6, :], *g_v_list, *g_no_force_list, *g_land_list) # *g_lift_list, *g_fc_dict # *g_no_force_list # g_land
f = sum(f_list)

lbg = np.concatenate((multi_shoot_lbg_flat, tau_lbg_flat, *v_lbg_flat, *no_force_lbg_flat, *land_lbg_flat), axis=1)  # *lift_lbg_flat, *fc_lbg_flat
ubg = np.concatenate((multi_shoot_ubg_flat, tau_ubg_flat, *v_ubg_flat, *no_force_ubg_flat, *land_ubg_flat), axis=1)  # *lift_ubg_flat, *fc_ubg_flat

prob_dict = {'f': f, 'x': w, 'g': cs.vec(g)}

opts = {
        'ipopt.tol': 0.001,
        'ipopt.constr_viol_tol': 0.001,
        'ipopt.max_iter': 2000,
        'ipopt.linear_solver': 'ma57'}
        # 'verbose': True,
        # banned options:
        # 'ipopt.hessian_approximation': 'limited-memory',
        # 'expand': True,

# create solver from prob
tic = time.time()
solver = cs.nlpsol('solver', 'ipopt', prob_dict, opts)
toc = time.time()
print('time elapsed loading:', toc - tic)

# print(ubw)
# exit()
tic = time.time()
solution = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
toc = time.time()
print('time elapsed solving:', toc - tic)

ms = mat_storer.matStorer(f'{os.path.splitext(os.path.basename(__file__))[0]}.mat')
ms.store(dict(a=np.array(solution['x'])))

#!/usr/bin/env python

from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
import casadi as cs
import numpy as np
from horizon import problem
from horizon.utils import utils, kin_dyn, mat_storer
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.utils.plotter import PlotterHorizon
from horizon.solvers import solver
import matplotlib.pyplot as plt
import os
import time
import horizon.transcriptions.integrators as integ
from horizon.ros import utils as horizon_ros_utils

try:
    from horizon.ros.replay_trajectory import *
    do_replay = True
except ImportError:
    do_replay = False

path_to_examples = os.path.abspath(__file__ + "/../../examples/")
# Loading URDF model in pinocchio
urdffile = os.path.join(path_to_examples, 'urdf', 'cart_pole.urdf')
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

nq = kindyn.nq()
nv = kindyn.nv()

# OPTIMIZATION PARAMETERS
tf = 5.0  # [s]
ns = 100 # number of shooting nodes
dt = tf/ns

N_control = ns
N_states = ns + 1
# ====================
# unraveled dimensions
# ====================
q_dim = N_states * nq
q_dot_dim = N_states * nv
q_ddot_dim = N_control * nv

# Creates problem STATE variables
q = cs.MX.sym("q", nq)
qdot = cs.MX.sym("qdot", nv)
qddot = cs.MX.sym("qddot", nv)

x = cs.vertcat(q, qdot)
# Creates double integrator
xdot = utils.double_integrator(qdot, qddot)
dae = {'x': x, 'p': qddot, 'ode': xdot, 'quad': 1}
integrator = integ.RK4(dae)
# int_map = integrator
int_map = integrator.map(N_control, 'thread', 15)

type_sym = cs.SX
q_i = type_sym.sym("q_i", nq, N_states)
q_dot_i = type_sym.sym("q_dot_i", nv, N_states)
q_ddot_i = type_sym.sym("q_ddot_i", nv, N_control)

X = cs.vertcat(q_i, q_dot_i)
U = q_ddot_i

X_int = int_map(X[:, :ns], U, dt)[0] #dt_i because it's N+1
# starting from node 1
g_multi_shoot = X_int - X[:, 1:]

g_up = q_i[1, ns] - np.pi
g_up_all = q_i[1, :] - np.pi

g_vel = q_dot_i[:, ns]


# Limits
q_min = [-0.5, -2.*np.pi]
q_max = [0.5, 2.*np.pi]
q_init = [0., 0.]

qddot_lims = np.array([1000., 1000.])
qdot_lims = np.array([100., 100.])


q0 = np.zeros([nq, N_states])
q_dot0 = np.zeros([nv, N_states])
q_ddot0 = np.zeros([nv, N_control])

q_lbw = np.tile(q_min, (N_states, 1)).T
q_dot_lbw = np.tile(-qdot_lims, (N_states, 1)).T
q_ddot_lbw = np.tile(-qddot_lims, (N_control, 1)).T

q_ubw = np.tile(q_max, (N_states, 1)).T
q_dot_ubw = np.tile(qdot_lims, (N_states, 1)).T
q_ddot_ubw = np.tile(qddot_lims, (N_control, 1)).T

# initial q
q_lbw[:, 0] = q_init
q_ubw[:, 0] = q_init
# zero initial velocity
q_dot_lbw[:, 0] = np.zeros([1, nv])
q_dot_ubw[:, 0] = np.zeros([1, nv])

id_fn_i = kin_dyn.InverseDynamicsMap(ns, kindyn)
g_tau_i = id_fn_i.call(q_i[:, :ns], q_dot_i[:, :ns], q_ddot_i)

multi_shoot_lbg = np.zeros(g_multi_shoot.shape)
multi_shoot_ubg = np.zeros(g_multi_shoot.shape)

tau_lims = np.array([1000., 0.])
tau_lbg = np.tile(-tau_lims, (N_control, 1)).T
tau_ubg = np.tile(tau_lims, (N_control, 1)).T

up_lbg = np.zeros(g_up.shape)
up_ubg = np.zeros(g_up.shape)

up_all_lbg = - cs.inf * np.ones(g_up_all.shape)
up_all_lbg[:, ns] = 0
up_all_ubg = cs.inf * np.ones(g_up_all.shape)
up_all_ubg[:, ns] = 0

vel_lbg = np.zeros(g_vel.shape)
vel_ubg = np.zeros(g_vel.shape)

f_list = list()
f_list.append(cs.sumsqr(q_ddot_i))

# ========================================================================
q0_flat = np.reshape(q0, [1, q_dim], order='F')
q_dot0_flat = np.reshape(q_dot0, [1, q_dot_dim], order='F')
q_ddot0_flat = np.reshape(q_ddot0, [1, q_ddot_dim], order='F')
w0 = np.concatenate((q0_flat, q_dot0_flat, q_ddot0_flat), axis=1) #dt0_flat


q_lbw_flat = np.reshape(q_lbw, [1, q_dim], order='F')
q_dot_lbw_flat = np.reshape(q_dot_lbw, [1, q_dot_dim], order='F')
q_ddot_lbw_flat = np.reshape(q_ddot_lbw, [1, q_ddot_dim], order='F')
lbw = np.concatenate((q_lbw_flat, q_dot_lbw_flat, q_ddot_lbw_flat), axis=1) #dt_ubw_flat

q_ubw_flat = np.reshape(q_ubw, [1, q_dim], order='F')
q_dot_ubw_flat = np.reshape(q_dot_ubw, [1, q_dot_dim], order='F')
q_ddot_ubw_flat = np.reshape(q_ddot_ubw, [1, q_ddot_dim], order='F')
ubw = np.concatenate((q_ubw_flat, q_dot_ubw_flat, q_ddot_ubw_flat), axis=1) #dt_ubw_flat

# CONSTRAINTS
multi_shoot_lbg_flat = np.reshape(multi_shoot_lbg, [1, g_multi_shoot.shape[0] * g_multi_shoot.shape[1]], order='F')
multi_shoot_ubg_flat = np.reshape(multi_shoot_ubg, [1, g_multi_shoot.shape[0] * g_multi_shoot.shape[1]], order='F')

tau_lbg_flat = np.reshape(tau_lbg, [1, g_tau_i.shape[0] * g_tau_i.shape[1]], order='F')
tau_ubg_flat = np.reshape(tau_ubg, [1, g_tau_i.shape[0] * g_tau_i.shape[1]], order='F')

up_lbg_flat = np.reshape(up_lbg, [1, g_up.shape[0] * g_up.shape[1]], order='F')
up_ubg_flat = np.reshape(up_ubg, [1, g_up.shape[0] * g_up.shape[1]], order='F')

up_all_lbg_flat = np.reshape(up_all_lbg, [1, g_up_all.shape[0] * g_up_all.shape[1]], order='F')
up_all_ubg_flat = np.reshape(up_all_ubg, [1, g_up_all.shape[0] * g_up_all.shape[1]], order='F')

vel_lbg_flat = np.reshape(vel_lbg, [1, g_vel.shape[0] * g_vel.shape[1]], order='F')
vel_ubg_flat = np.reshape(vel_ubg, [1, g_vel.shape[0] * g_vel.shape[1]], order='F')

w = cs.veccat(q_i, q_dot_i, q_ddot_i) # dt_i
# g = cs.veccat(g_multi_shoot, g_tau_i, g_up, g_vel)
g = cs.veccat(g_multi_shoot, g_tau_i, g_up_all, g_vel)
f = sum(f_list)

# lbg = np.concatenate((multi_shoot_lbg_flat, tau_lbg_flat, up_lbg_flat, vel_lbg_flat), axis=1)
# ubg = np.concatenate((multi_shoot_ubg_flat, tau_ubg_flat, up_ubg_flat, vel_ubg_flat), axis=1)

lbg = np.concatenate((multi_shoot_lbg_flat, tau_lbg_flat, up_all_lbg_flat, vel_lbg_flat), axis=1)
ubg = np.concatenate((multi_shoot_ubg_flat, tau_ubg_flat, up_all_ubg_flat, vel_ubg_flat), axis=1)

prob_dict = {'f': f, 'x': w, 'g': cs.vec(g)}

opts = {'ipopt.tol': 1e-4,
        'ipopt.max_iter': 2000}

# create solver from prob
tic = time.time()
solver = cs.nlpsol('solver', 'ipopt', prob_dict, opts)
toc = time.time()
print('time elapsed loading:', toc - tic)


tic = time.time()
solution = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
toc = time.time()
print('time elapsed solving:', toc - tic)

# ms = mat_storer.matStorer(f'{os.path.splitext(os.path.basename(__file__))[0]}.mat')
# ms.store(dict(a=np.array(solution['x'])))

i = 0
i_new = q_dim
q_sol = solution['x'][i:i_new]
i = i_new
i_new = i_new + q_dot_dim
q_dot_sol = solution['x'][i:i_new]
i = i_new
i_new = i_new + q_ddot_dim
q_ddot_sol = solution['x'][i:i_new]


q_sol = np.reshape(q_sol, [nq, N_states], order='F')
q_dot_sol = np.reshape(q_dot_sol, [nv, N_states], order='F')
q_ddot_sol = np.reshape(q_ddot_sol, [nv, N_control], order='F')

time = np.arange(0.0, tf+1e-6, tf/ns)

plt.figure()
plt.plot(time, q_sol[0, :])
plt.plot(time, q_sol[1, :])
plt.suptitle('$\mathrm{Base \ Position}$', size = 20)
plt.xlabel('$\mathrm{[sec]}$', size = 20)
plt.ylabel('$\mathrm{[m]}$', size = 20)

plt.show()
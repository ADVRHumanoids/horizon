# /usr/bin/env python3
import numpy as np
##### for robot and stuff
import xbot_interface.config_options as xbot_opt
from xbot_interface import xbot_interface as xbot
import rospy
from casadi_kin_dyn import py3casadi_kin_dyn
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter, mat_storer, refiner
from horizon.ros.replay_trajectory import *
import os
import matplotlib.pyplot as plt

ms = mat_storer.matStorer('/tmp/dioboy.mat')
solution = ms.load()

n_q = solution['q'].shape[0]
num_samples = solution['q'].shape[1]
num_samples_res = solution['q_res'].shape[1]

dt = solution['dt'][0][0]
dt_res = solution['dt_res'][0][0]
joint_names = solution['joint_names']

q = solution['q']
q_res = solution['q_res']
a = solution['a']

tau = solution['tau']
tau_res = solution['tau_res']

contacts = ['contact_1', 'contact_2', 'contact_3', 'contact_4']

contact_map_res = {c: solution[f'f_{c}'] for c in contacts}

# ==== plot stuff ====
t_nodes = np.zeros([num_samples])
for i in range(1, num_samples):
    t_nodes[i] = t_nodes[i - 1] + dt

t_nodes_res = np.zeros([num_samples_res])
for i in range(1, num_samples_res):
    t_nodes_res[i] = t_nodes_res[i - 1] + dt_res

# plot q
plt.figure()
for dim in range(a.shape[0]):
    plt.plot(t_nodes[:-1], a[dim, :])

plt.title('a')

plt.show()
exit()
for dim in range(q.shape[0]):
    plt.scatter(t_nodes, q[dim, :])

for dim in range(q_res.shape[0]):
    plt.plot(t_nodes_res, q_res[dim, :])
plt.title('q')

# plot tau base
plt.figure()
for dim in range(tau[:6].shape[0]):
    plt.scatter(t_nodes[:-1], tau[dim, :])

for dim in range(tau_res[:6].shape[0]):
    plt.plot(t_nodes_res[:-1], tau_res[dim, :])
plt.title('tau_base')

# plot tau
plt.figure()
for dim in range(tau.shape[0]):
    plt.scatter(t_nodes[:-1], tau[dim, :])

for dim in range(tau_res.shape[0]):
    plt.plot(t_nodes_res[:-1], tau_res[dim, :])
plt.title('tau')

plt.show()








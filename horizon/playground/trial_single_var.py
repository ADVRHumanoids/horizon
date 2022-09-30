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
from horizon.ros import utils as horizon_ros_utils
from logging import DEBUG

try:
    from horizon.ros.replay_trajectory import *
    do_replay = True
except ImportError:
    do_replay = False
# Loading URDF model in pinocchio
urdffile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'urdf', 'cart_pole.urdf')
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

nq = kindyn.nq()
nv = kindyn.nv()

print("nq: ", nq)
print("nv: ", nv)

# OPTIMIZATION PARAMETERS
tf = 5.0  # [s]
ns = 150  # number of shooting nodes
dt = tf/ns
use_ms = True

# Create horizon problem
prb = problem.Problem(ns, logging_level=DEBUG)

# Creates problem STATE variables
q = prb.createStateVariable("q", nq)
qdot = prb.createStateVariable("qdot", nv)

# Creates problem CONTROL variables
qddot = prb.createInputVariable("qddot", nv)

dt_param = prb.createSingleParameter('dt_par', 1)
dt_some = prb.createVariable('dt_some', 1, range(0, 51))
dt1 = prb.createSingleVariable('dt1', 1)
dt2 = prb.createSingleVariable('dt2', 1)
dt1.setBounds(0.01, 0.05)
dt2.setBounds(0.06, 0.1)
dt_some.setBounds(0.01, 0.1)

# Creates double integrator
xdot = utils.double_integrator(qdot, qddot)
prb.setDynamics(xdot)

dt_all = int(ns/2) * [dt] + int(ns/2) * [dt1]
prb.setDt(dt_all)

# Limits
q_min = [-0.5, -2.*np.pi]
q_max = [0.5, 2.*np.pi]
q_init = [0., 0.]

qdot_lims = np.array([100., 100.])
qdot_init = [0., 0.]

qddot_lims = np.array([1000., 1000.])
qddot_init = [0., 0.]

q.setBounds(q_min, q_max)
q.setBounds(q_init, q_init, nodes=0)
qdot.setBounds((-qdot_lims).tolist(), qdot_lims.tolist())
qdot.setBounds(qdot_init, qdot_init, nodes=0)
qddot.setBounds((-qddot_lims).tolist(), qddot_lims.tolist())


q.setInitialGuess(q_init)
qdot.setInitialGuess(qdot_init)
qddot.setInitialGuess(qddot_init)

# Cost function
prb.createIntermediateCost("qddot", cs.sumsqr(qddot))

# Dynamics
if use_ms:
    th = Transcriptor.make_method('multiple_shooting', prb, opts=dict(integrator='RK4'))
else:
    th = Transcriptor.make_method('direct_collocation', prb) # opts=dict(degree=5)

prb.createFinalConstraint("up", q[1] - np.pi)
prb.createFinalConstraint("final_qdot", qdot)


tau_lims = np.array([1000., 0.])
tau = kin_dyn.InverseDynamics(kindyn).call(q, qdot, qddot)
iv = prb.createIntermediateConstraint("inverse_dynamics", tau, bounds=dict(lb=-tau_lims, ub=tau_lims))

# Creates problem
tic = time.time()
solver = solver.Solver.make_solver('ipopt', prb, opts={'ipopt.tol': 1e-4,'ipopt.max_iter': 2000})
toc = time.time()
print('time elapsed loading:', toc - tic)

# new_nodes = 100
# prb.setNNodes(new_nodes)
#
# for cnsrt_name, cnsrt_fun in prb.getConstraints().items():
#     print(cnsrt_fun.getNodes())
#
# exit()

tic = time.time()
solver.solve()
toc = time.time()
print('time elapsed solving:', toc - tic)
solution = solver.getSolutionDict()
dt_sol = solver.getDt()

print(dt_sol)

q_hist = solution["q"]

time = np.arange(0.0, tf+1e-6, tf/ns)
plt.figure()
plt.plot(time, q_hist[0,:])
plt.plot(time, q_hist[1,:])
plt.suptitle('$\mathrm{Base \ Position}$', size = 20)
plt.xlabel('$\mathrm{[sec]}$', size = 20)
plt.ylabel('$\mathrm{[m]}$', size = 20)

plot_all = True
if plot_all:
    hplt = PlotterHorizon(prb, solution)
    # hplt.plotVariables()
    hplt.plotFunction('inverse_dynamics', dim=[1], show_bounds=True)
    plt.show()

if do_replay:
    joint_list=["cart_joint", "pole_joint"]
    replay_trajectory(tf/ns, joint_list, q_hist).replay(is_floating_base=False)









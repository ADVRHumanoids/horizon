#!/usr/bin/env python3

'''
An example of the cart-pole problem: find the trajectory of the cart so that the pole reaches the upright position.
'''

# import the necessary modules
import casadi as cs
import numpy as np
from horizon import problem
from horizon.utils import utils, kin_dyn, mat_storer, plotter
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.solvers import solver
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
import matplotlib.pyplot as plt
import os

# get path to the examples folder
path_to_examples = os.path.abspath(__file__ + "/../../../")

rviz_replay = True
plot_sol = True

# Create CasADi interface to Pinocchio
urdffile = os.path.join(path_to_examples, 'urdf', 'cart_pole.urdf')
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

# Get dimension of pos and vel
nq = kindyn.nq()
nv = kindyn.nv()

# Optimization parameters
tf = 5.0  # [s]
ns = 50  # number of nodes
dt = tf/ns

# options for horizon transcription
transcription_method = 'direct_collocation'  # can choose between 'multiple_shooting' and 'direct_collocation'
transcription_opts = dict(integrator='RK4') # integrator used by the multiple_shooting

# Create horizon problem
prb = problem.Problem(ns)

# creates the variables for the problem
# design choice: position and the velocity as state and acceleration as input

# STATE variables
q = prb.createStateVariable("q", nq)
qdot = prb.createStateVariable("qdot", nv)
# CONTROL variables
qddot = prb.createInputVariable("qddot", nv)

# Creates double integrator
xdot = utils.double_integrator(qdot, qddot)

# Set dynamics of the system and the relative dt
prb.setDynamics(xdot)
prb.setDt(dt)


# ================== Set BOUNDS and INITIAL GUESS  ===============================
# joint limits + initial pos
q_min = [-0.5, -2.*np.pi]
q_max = [0.5, 2.*np.pi]
q_init = [0., 0.]
# velocity limits + initial vel
qdot_lims = np.array([100., 100.])
qdot_init = [0., 0.]
# acceleration limits
qddot_lims = np.array([1000., 1000.])
qddot_init = [0., 0.]

# Set bounds
q.setBounds(q_min, q_max)
q.setBounds(q_init, q_init, nodes=0)
qdot.setBounds(-qdot_lims, qdot_lims)
qdot.setBounds(qdot_init, qdot_init, nodes=0)
qddot.setBounds(-qddot_lims, qddot_lims)

# Set initial guess
q.setInitialGuess(q_init)
qdot.setInitialGuess(qdot_init)
qddot.setInitialGuess(qddot_init)

# ================== Set TRANSCRIPTION METHOD ===============================
th = Transcriptor.make_method(transcription_method, prb, opts=transcription_opts)

# ====================== Set CONSTRAINTS ===============================

# Set dynamic feasibility:
# the cart can apply a torque, the joint connecting the cart to the pendulum is UNACTUATED
# the torques are computed using the inverse dynamics, as the input of the problem is the cart acceleration
tau_lims = np.array([1000., 0.])
tau = kin_dyn.InverseDynamics(kindyn).call(q, qdot, qddot)
iv = prb.createIntermediateConstraint("dynamic_feasibility", tau, bounds=dict(lb=-tau_lims, ub=tau_lims))

# at the last node, the pendulum is upright
prb.createFinalConstraint("up", q[1] - np.pi)
# at the last node, the system velocity is zero
prb.createFinalConstraint("final_qdot", qdot)

# ====================== Set COSTS ===============================

# minimize the acceleration of system (regularization of the input)
prb.createIntermediateCost("qddot", cs.sumsqr(qddot))

# ==================== BUILD PROBLEM ===============================
# the solver class accept different solvers, such as 'ipopt', 'ilqr', 'gnsqp'.
# Different solver are useful (and feasible) in different situations.
solv = solver.Solver.make_solver('ipopt', prb, opts={'ipopt.tol': 1e-4,'ipopt.max_iter': 2000})

# ==================== SOLVE PROBLEM ===============================
solv.solve()

# the solution is retrieved in the form of a dictionary ('variable_name' = values)
solution = solv.getSolutionDict()
# the dt is retrieved as a vector of values (size: number of intervals --> n_nodes - 1)
dt_sol = solv.getDt()
# ====================== PLOT SOLUTION =======================

if plot_sol:
    # Horizon expose a plotter to simplify the generation of graphs
    # Once instantiated, variables and constraints can be plotted with ease

    time = np.arange(0.0, tf+1e-6, tf/ns)
    plt.figure()
    plt.plot(time, solution['q'][0,:])
    plt.plot(time, solution['q'][1,:])
    plt.suptitle('$\mathrm{Base \ Position}$', size = 20)
    plt.xlabel('$\mathrm{[sec]}$', size = 20)
    plt.ylabel('$\mathrm{[m]}$', size = 20)

    hplt = plotter.PlotterHorizon(prb, solution)
    ## plot the desired dimensions of the constraint 'dynamic_feasibility'
    hplt.plotFunction('dynamic_feasibility', dim=[1], show_bounds=True)
    plt.show()

# ====================== REPLAY SOLUTION =======================
if rviz_replay:

    # set ROS stuff and launchfile
    import rospy
    from horizon.ros.replay_trajectory import replay_trajectory
    import subprocess
    # temporary add the example path to the environment
    os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
    subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=cart_pole'])
    rospy.loginfo("'cart_pole' visualization started.")

    # visualize the robot in RVIZ
    joint_list= ["cart_joint", "pole_joint"]
    replay_trajectory(tf/ns, joint_list, solution['q']).replay(is_floating_base=False)








#!/usr/bin/env python3

from casadi_kin_dyn import pycasadi_kin_dyn
import casadi as cs
import numpy as np
from horizon import problem
from horizon.utils import utils, kin_dyn
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.utils.plotter import PlotterHorizon
from horizon.solvers import solver
import matplotlib.pyplot as plt
import os

rviz_replay = True
plot_sol = True

# Loading URDF model in pinocchio
path_to_examples = os.path.abspath(__file__ + "/../../../")

urdffile = os.path.join(path_to_examples, 'urdf', 'cart_pole_xy.urdf')
urdf = open(urdffile, 'r').read()

# Create casadi interface to pinocchio
kindyn = pycasadi_kin_dyn.CasadiKinDyn(urdf)
nq = kindyn.nq()
nv = kindyn.nv()

# Optimization parameters
tf = 5.0  # [s]
ns = 80  # number of nodes
dt = tf / ns

# options for horizon transcription
transcription_method = 'multiple_shooting' # can choose between 'multiple_shooting' and 'direct_collocation'
transcription_opts = dict(integrator='EULER')# integrator used by the multiple_shooting


# Create horizon problem
prb = problem.Problem(ns)

# creates the variables for the problem
# design choice: position and the velocity as state and acceleration as input

# q0: cart x-displacement
# q1: cart y-displacement
# q2: pole rotation

# STATE variables
q = prb.createStateVariable("q", nq)
qdot = prb.createStateVariable("qdot", nv)

# CONTROL variables
qddot = prb.createInputVariable("qddot", nv)

# create a parameter for a reference trajectory.
# Parameters can be left symbolic when building the problem, and must be set to a specific numerical value before solving.
# Parameters are very useful to avoid hard-coded values or trajectories: once the problem is built, the parameters values
# can be changed at will before solving the problem.
# In this specific case, this parameter is the reference trajectory of the cart pole on the y-axis
q_ref = prb.createParameter('q_ref', 1)

# Create double integrator
xdot = utils.double_integrator(qdot, qddot)

# Set dynamics of the system and the relative dt
prb.setDynamics(xdot)
prb.setDt(dt)

# ================== Set BOUNDS and INITIAL GUESS  ===============================
# joint limits + initial pos
q_min = [-0.5, -0.5, -2. * np.pi]
q_max = [0.5, 0.5, 2. * np.pi]
q_init = [0., 0., 0.]
# velocity limits + initial vel
qdot_lims = np.array([100., 100., 100.])
qdot_init = [0., 0., 0.]
# acceleration limits
qddot_lims = np.array([1000., 1000., 1000.])
qddot_init = [0., 0., 0.]

# Set bounds
q.setBounds(q_min, q_max)
q.setBounds(q_init, q_init, nodes=0)
qdot.setBounds((-qdot_lims).tolist(), qdot_lims.tolist())
qdot.setBounds(qdot_init, qdot_init, nodes=0)
qddot.setBounds((-qddot_lims).tolist(), qddot_lims.tolist())

q.setInitialGuess(q_init)
qdot.setInitialGuess(qdot_init)
qddot.setInitialGuess(qddot_init)

# Cost function

# ================== Set TRANSCRIPTION METHOD ===============================
th = Transcriptor.make_method(transcription_method, prb, opts=transcription_opts)

# ====================== Set CONSTRAINTS ===============================

# Set dynamic feasibility:
# the cart can apply a torque, the joint connecting the cart to the pendulum is UNACTUATED
# the torques are computed using the inverse dynamics, as the input of the problem is the cart acceleration
tau_lims = np.array([1000., 1000., 0.])
tau = kin_dyn.InverseDynamics(kindyn).call(q, qdot, qddot)
prb.createIntermediateConstraint("dynamic_feasibility", tau, bounds=dict(lb=-tau_lims, ub=tau_lims))

# at the last node, the pendulum is upright
prb.createFinalConstraint("up", q[2] - np.pi)
# at the last node, the system velocity is zero
prb.createFinalConstraint("final_qdot", qdot)
# the trajectory on the y-axis must track q_ref: notice how q_ref is a parameter and NOT a fixed trajectory.
# It can be changed at will after building the problem
cnrst_ref = prb.createConstraint('sinusoidal_ref', q[1] - q_ref, range(10, ns+1))

# ====================== Set COSTS ===============================

# minimize the acceleration of system (regularization of the input)
prb.createIntermediateCost("qddot", cs.sumsqr(qddot))

# ==================== BUILD PROBLEM ===============================
# the solver class accept different solvers, such as 'ipopt', 'ilqr', 'gnsqp'.
# Different solver are useful (and feasible) in different situations.
solver = solver.Solver.make_solver('ipopt', prb, opts={'ipopt.tol': 1e-4, 'ipopt.max_iter': 2000})

# the problem is built. Depending on the size of the problem, it may require some time
# Setting the numerical values of the parameter AFTER the problem is build
# This values can be changed at will before solving the problem
# In this case, a cosine function is set as the trajectory of the cart on the y-axis
cos_fun = 1/3 * np.cos(np.linspace(np.pi/2, 4*2*np.pi, ns+1))
cos_fun = np.atleast_2d(cos_fun)
q_ref.assign(cos_fun)

# ==================== SOLVE PROBLEM ===============================
solver.solve()
# the solution is retrieved in the form of a dictionary ('variable_name' = values)

solution = solver.getSolutionDict()

time = np.arange(0.0, tf + 1e-6, tf / ns)
plt.figure()
plt.plot(time, solution['q'][0, :])
plt.plot(time, solution['q'][1, :])
plt.suptitle('$\mathrm{Base \ Position}$', size=20)
plt.xlabel('$\mathrm{[sec]}$', size=20)
plt.ylabel('$\mathrm{[m]}$', size=20)


if plot_sol:
    # Horizon expose a plotter to simplify the generation of graphs
    # Once instantiated, variables and constraints can be plotted with ease

    hplt = PlotterHorizon(prb, solution)
    hplt.plotVariables()
    hplt.plotFunctions()
    plt.show()

if rviz_replay:

    # set ROS stuff and launchfile
    import rospy
    from horizon.ros.replay_trajectory import replay_trajectory
    import subprocess
    # temporary add the example path to the environment
    os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
    subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=cart_pole_xy'])
    rospy.loginfo("'cart_pole' visualization started.")

    # visualize the robot in RVIZ
    joint_list = ["cart_joint_x", "cart_joint_y", "pole_joint"]
    replay_trajectory(tf/ns, joint_list, solution['q']).replay(is_floating_base=False)
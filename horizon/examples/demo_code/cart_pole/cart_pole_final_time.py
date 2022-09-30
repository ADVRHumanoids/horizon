#!/usr/bin/env python3

from casadi_kin_dyn import pycasadi_kin_dyn
from horizon.transcriptions.transcriptor import Transcriptor
import casadi as cs
import numpy as np
from horizon import problem
from horizon.utils import utils
from horizon.transcriptions import integrators
from horizon.solvers import solver
import matplotlib.pyplot as plt
import os

# get path to the examples folder
path_to_examples = os.path.abspath(__file__ + "/../../../")

rviz_replay = True
plot_sol = True

# Create CasADi interface to Pinocchio
urdffile = os.path.join(path_to_examples, 'urdf', 'cart_pole.urdf')
urdf = open(urdffile, 'r').read()
kindyn = pycasadi_kin_dyn.CasadiKinDyn(urdf)

# Get dimension of pos and vel
nq = kindyn.nq()
nv = kindyn.nv()

# Optimization parameters
ns = 100  # number of nodes

# Create horizon problem
prb = problem.Problem(ns)

# creates the variables for the problem
# design choice: position and the velocity as state and torque as input

# Create problem STATE variables: x = [q, qdot]
q = prb.createStateVariable("q", nq)
qdot = prb.createStateVariable("qdot", nv)

# Create problem CONTROL variables: tau = [u, 0], embed under-actuation in control
u = prb.createInputVariable("u", 1)
tau = cs.vertcat(u, 0.)

# Create final time variable so that the duration of the trajectory is included in the optimization problem
tf = prb.createVariable("tf", 1)
# the dt of the system is the final time (variable) divided by the number of nodes
dt = tf/ns

# Set dynamics of the system and the relative dt.
fd = kindyn.aba()  # this is the forward dynamics function
qddot = fd(q=q, v=qdot, tau=tau)['a'] # qddot = M^-1(tau - h)
xdot = utils.double_integrator(qdot, qddot) # xdot = [qdot, qddot]
prb.setDynamics(xdot)
prb.setDt(dt)

# setting up a custom integrator for the multiple shooting constraint
L = 0.5*cs.dot(qdot, qdot)  # Objective term
dae = {'x': x, 'p': u, 'ode': xdot, 'quad': L}
F_integrator = integrators.RK4(dae, opts=None, casadi_type=cs.SX)

# ================== Set BOUNDS and INITIAL GUESS  ===============================
# joint limits + initial pos
q_min = [-0.5, -2.*np.pi]
q_max = [0.5, 2.*np.pi]
q_init = [0., 0.]
# velocity limits + initial vel
qdot_lims = np.array([100., 100.])
qdot_init = [0., 0.]
# torque limits
tau_lims = np.array([3000])
tau_init = [0]
# final time bounds
tf_min = 1.
tf_max = 10.
tf_init = 3.

# Set bounds
q.setBounds(q_min, q_max)
q.setBounds(q_init, q_init, nodes=0)
qdot.setBounds(-qdot_lims, qdot_lims)
qdot.setBounds(qdot_init, qdot_init, nodes=0)
u.setBounds(-tau_lims, tau_lims)
tf.setBounds(tf_min, tf_max)

# Set initial guess
q.setInitialGuess(q_init)
qdot.setInitialGuess(qdot_init)
tf.setInitialGuess(tf_init)

# ====================== Set CONSTRAINTS ===============================
q_prev = q.getVarOffset(-1)
qdot_prev = qdot.getVarOffset(-1)
u_prev = u.getVarOffset(-1)
x_prev, _ = utils.double_integrator(qdot_prev, fd(q=q_prev, v=qdot_prev, tau=cs.vertcat(u_prev, 0.))['a'])
x_int = F_integrator(x0=x_prev, p=u_prev, time=dt)
prb.createConstraint("multiple_shooting", x_int["xf"] - x, nodes=list(range(1, ns+1)), bounds=dict(lb=np.zeros(nv+nq), ub=np.zeros(nv+nq)))

prb.createFinalConstraint("up", q[1] - np.pi)
prb.createFinalConstraint("final_qdot", qdot)


# ====================== Set COSTS ===============================

prb.createIntermediateCost("tau", cs.sumsqr(tau))
prb.createCost("min_tf", 1000.*cs.sumsqr(tf))


# ==================== BUILD PROBLEM ===============================
# the solver class accept different solvers, such as 'ipopt', 'ilqr', 'gnsqp'.
# Different solver are useful (and feasible) in different situations.

solv = solver.Solver.make_solver('ipopt', prb, opts=None)

# ==================== SOLVE PROBLEM ===============================
solv.solve()

# the solution is retrieved in the form of a dictionary ('variable_name' = values)
solution = solv.getSolutionDict()
# the dt is retrieved as a vector of values (size: number of intervals --> n_nodes - 1)
dt_sol = solv.getDt()
# ====================== PLOT SOLUTION =======================

q_hist = solution["q"]
tf_sol = solution["tf"]

print(f'Tf: {solution["tf"].flatten()}')

if plot_sol:
    # Horizon expose a plotter to simplify the generation of graphs
    # Once instantiated, variables and constraints can be plotted with ease

    time = np.arange(0.0, solution["tf"] + 1e-6, solution["tf"] / ns)
    plt.figure()
    plt.plot(time, solution["q"][0,:])
    plt.plot(time, solution["q"][1,:])
    plt.suptitle('$\mathrm{Base \ Position}$', size = 20)
    plt.xlabel('$\mathrm{[sec]}$', size = 20)
    plt.ylabel('$\mathrm{[m]}$', size = 20)

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
    replay_trajectory(tf_sol/ns, joint_list, solution['q']).replay(is_floating_base=False)

else:
    print("To visualize the robot trajectory, start the script with the '--replay' option.")





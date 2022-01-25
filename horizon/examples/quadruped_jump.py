#!/usr/bin/env python
import logging

import rospy
import casadi as cs
import numpy as np
from horizon import problem
from horizon.utils import utils, kin_dyn, resampler_trajectory
from horizon.transcriptions import integrators
from horizon.solvers import solver
from horizon.utils.plotter import PlotterHorizon
from horizon.ros.replay_trajectory import *
import matplotlib.pyplot as plt
import os, argparse


parser = argparse.ArgumentParser(description='cart-pole problem: moving the cart so that the pole reaches the upright position')
parser.add_argument('--replay', help='visualize the robot trajectory in rviz', action='store_true')
args = parser.parse_args()

rviz_replay = False
plot_sol = True
resample = True

if args.replay:
    from horizon.ros.replay_trajectory import *
    import roslaunch, rospkg, rospy
    rviz_replay = True
    plot_sol = False
    resample = True

urdffile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'urdf', 'quadruped_template.urdf')
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

# OPTIMIZATION PARAMETERS
n_nodes = 40  # number of nodes
n_c = 4  # number of contacts
n_q = kindyn.nq()  # number of DoFs - NB: 7 DoFs floating base (quaternions)
DoF = n_q - 7  # Contacts + anchor_rope + rope
n_v = kindyn.nv()  # Velocity DoFs
n_f = 3  # Force DOfs

contact_names = ['Contact1', 'Contact2', 'Contact3', 'Contact4']

# Create horizon problem
prb = problem.Problem(n_nodes)

# Creates problem STATE variables
q = prb.createStateVariable("q", n_q)
qdot = prb.createStateVariable("qdot", n_v)

# Creates problem CONTROL variables
qddot = prb.createInputVariable("qddot", n_v)

f_list = list()
for i in range(n_c):
    f_list.append(prb.createInputVariable(f'f{i}', n_f))

dt_res = prb.createInputVariable("dt", 1)

x, xdot = utils.double_integrator_with_floating_base(q, qdot, qddot)

prb.setDynamics(xdot)
prb.setDt(dt_res)
# Formulate discrete time dynamics
L = 0.5*cs.dot(qdot, qdot)  # Objective term
dae = {'x': x, 'p': qddot, 'ode': xdot, 'quad': L}
F_integrator = integrators.RK4(dae, {}, cs.SX)

# Limits
q_min = kindyn.q_min()
q_max = kindyn.q_max()
q_min[:3] = [-10, -10, -10]
q_max[:3] = [10, 10, 10]

q_init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, # floating base
          0.349999, 0.349999, -0.635, # contact 1
          0.349999, -0.349999, -0.635, # contact 2
          -0.349999, -0.349999, -0.635, # contact 3
          -0.349999, 0.349999, -0.635] # contact 4

q.setBounds(q_min, q_max)
q.setBounds(q_init, q_init, 0)
q.setInitialGuess(q_init)

qdot_min = -100.*np.ones(n_v)
qdot_max = -qdot_min
qdot_init = np.zeros(n_v)
qdot.setBounds(qdot_min, qdot_max)
qdot.setBounds(qdot_init, qdot_init, 0)
qdot.setInitialGuess(qdot_init)

qddot_min = -100.*np.ones(n_v)
qddot_max = -qddot_min
qddot_init = np.zeros(n_v)
qddot.setBounds(qddot_min, qddot_max)
qddot.setInitialGuess(qddot_init)

f_min = -10000.*np.ones(n_f)
f_max = -f_min
f_init = np.zeros(n_f)

for f in f_list:
    f.setBounds(f_min, f_max)
    f.setInitialGuess(f_init)

dt_min = 0.03 # [s]
dt_max = 0.15 # [s]
dt_init = dt_min
dt_res.setBounds(dt_min, dt_max)
dt_res.setInitialGuess(dt_init)

# SET UP COST FUNCTION
lift_node = 10
touch_down_node = 30
q_fb_trg = np.array([q_init[0], q_init[1], q_init[2] + 0.9, 0.0, 0.0, 0.0, 1.0])

# prb.createCost("jump", 10.*cs.dot(q[0:3] - q_fb_trg[0:3], q[0:3] - q_fb_trg[0:3]), nodes=list(range(lift_node, touch_down_node)))
prb.createCost("min_qdot", 10.*cs.sumsqr(qdot))

# prb.createFinalResidual(f"final_nominal_pos", cs.sqrt(1000) * (q - q_init))
# for f in f_list:
#     prb.createIntermediateResidual(f"min_{f.getName()}", cs.sqrt(0.01) * f)

# Constraints
q_prev = q.getVarOffset(-1)
qdot_prev = qdot.getVarOffset(-1)
qddot_prev = qddot.getVarOffset(-1)
dt_prev = dt_res.getVarOffset(-1)
x_prev, _ = utils.double_integrator_with_floating_base(q_prev, qdot_prev, qddot_prev)
x_int = F_integrator(x0=x_prev, p=qddot_prev, time=dt_prev)
prb.createConstraint("multiple_shooting", x_int["xf"] - x, nodes=list(range(1, n_nodes + 1)), bounds=dict(lb=np.zeros(n_v + n_q), ub=np.zeros(n_v + n_q)))

tau_min = np.array([0., 0., 0., 0., 0., 0.,  # Floating base
            -1000., -1000., -1000.,  # Contact 1
            -1000., -1000., -1000.,  # Contact 2
            -1000., -1000., -1000.,  # Contact 3
            -1000., -1000., -1000.])  # Contact 4

tau_max = - tau_min

contact_map = dict(zip(contact_names, f_list))
tau = kin_dyn.InverseDynamics(kindyn, contact_map.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED).call(q, qdot, qddot, contact_map)
prb.createConstraint("inverse_dynamics", tau, nodes=range(0, n_nodes), bounds=dict(lb=tau_min, ub=tau_max))
prb.createFinalConstraint('final_velocity', qdot)

# GROUND
mu = 0.8 # friction coefficient
R = np.identity(3, dtype=float) # environment rotation wrt inertial frame

# foot

for frame, f in zip(contact_names, f_list):
    # BEFORE AND AFTER FLIGHT PHASE
    FK = cs.Function.deserialize(kindyn.fk(frame))
    DFK = cs.Function.deserialize(kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))
    p = FK(q=q)['ee_pos']
    pd = FK(q=q_init)['ee_pos']
    v = DFK(q=q, qdot=qdot)['ee_vel_linear']

    prb.createConstraint(f"{frame}_vel_before_jump", v, nodes=range(0, lift_node))
    prb.createConstraint(f"{frame}_vel_after_jump", v, nodes=range(touch_down_node, n_nodes + 1))

    fc, fc_lb, fc_ub = kin_dyn.linearized_friciton_cone(f, mu, R)
    prb.createConstraint(f"{frame}_friction_cone_before_jump", fc, nodes=range(0, lift_node), bounds=dict(lb=fc_lb, ub=fc_ub))
    prb.createConstraint(f"{frame}_friction_cone_after_jump", fc, nodes=range(touch_down_node, n_nodes), bounds=dict(lb=fc_lb, ub=fc_ub))

    # LANDING
    prb.createConstraint(f"{frame}_after_jump", p - pd, nodes=touch_down_node)

    # DURING FLIGHT PHASE
    prb.createConstraint(f"{frame}_no_force_during_jump", f, nodes=range(lift_node, touch_down_node))

# Create problem
opts = {'ipopt.tol': 0.001,
        'ipopt.constr_viol_tol': 0.001,
        'ipopt.max_iter': 5000}#,
        # 'ipopt.linear_solver': 'ma57'}

solver = solver.Solver.make_solver('ipopt', prb, opts)
solver.solve()

solution = solver.getSolutionDict()

tau_sol = np.zeros(solution["qddot"].shape)
ID = kin_dyn.InverseDynamics(kindyn, contact_names, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
for i in range(n_nodes):
    contact_map_i = dict(zip(contact_names, [solution['f0'][:, i], solution['f1'][:, i], solution['f2'][:, i], solution['f3'][:, i]]))
    tau_sol[:, i] = ID.call(solution["q"][:, i], solution["qdot"][:, i], solution["qddot"][:, i], contact_map_i).toarray().flatten()

if resample:
    # resampling
    dt_res = 0.001
    contact_map_sol = dict(zip(contact_names, [solution['f0'], solution['f1'], solution['f2'], solution['f3']]))
    q_res, qdot_res, qddot_res, contact_map_sol_res, tau_res = resampler_trajectory.resample_torques(
                                                                            solution["q"], solution["qdot"], solution["qddot"], solution["dt"].flatten(),
                                                                            dt_res, dae, contact_map_sol, kindyn,
                                                                            cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

if plot_sol:

    hplt = PlotterHorizon(prb, solution)
    hplt.plotVariables(show_bounds=False, legend=True)
    # hplt.plotFunctions(show_bounds=False)

    plt.figure()
    for i in range(6):
        plt.plot(tau_sol[i, :])
    plt.suptitle('$\mathrm{Base \ Forces}$', size=20)
    plt.xlabel('$\mathrm{sample}$', size=20)
    plt.ylabel('$\mathrm{[N]}$', size=20)

    plt.show()

    if resample:

        time = np.arange(0.0, q_res.shape[1] * dt_res, dt_res)
        plt.figure()
        for i in range(6):
            plt.plot(time[:-1], tau_res[i, :])
        plt.suptitle('$\mathrm{Base \ Forces \ Resampled}$', size=20)
        plt.xlabel('$\mathrm{[sec]}$', size=20)
        plt.ylabel('$\mathrm{[N]}$', size=20)

        plt.figure()
        for i in range(q_res.shape[0]):
            plt.plot(time, q_res[i, :])
        plt.suptitle('$\mathrm{q \ Resampled}$', size=20)
        plt.xlabel('$\mathrm{[sec]}$', size=20)
        plt.ylabel('$\mathrm{q}$', size=20)

        plt.figure()
        for i in range(qdot_res.shape[0]):
            plt.plot(time, qdot_res[i, :])
        plt.suptitle('$\mathrm{\dot{q} \ Resampled}$', size=20)
        plt.xlabel('$\mathrm{[sec]}$', size=20)
        plt.ylabel('$\mathrm{\dot{q}}$', size=20)

        plt.figure()
        for i in range(qddot_res.shape[0]):
            plt.plot(time[:-1], qddot_res[i, :])
        plt.suptitle('$\mathrm{\ddot{q} \ Resampled}$', size=20)
        plt.xlabel('$\mathrm{[sec]}$', size=20)
        plt.ylabel('$\mathrm{\ddot{q}}$', size=20)

    plt.show()

if rviz_replay:

    # set ROS stuff and launchfile
    r = rospkg.RosPack()
    path_to_examples = r.get_path('horizon_examples')

    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.parent.ROSLaunchParent(uuid, [path_to_examples + "/replay/launch/quadruped_template.launch"])
    launch.start()
    rospy.loginfo("quadruped_jump' visualization started.")

    # visualize the robot in RVIZ
    joint_list = ['Contact1_x', 'Contact1_y', 'Contact1_z',
                  'Contact2_x', 'Contact2_y', 'Contact2_z',
                  'Contact3_x', 'Contact3_y', 'Contact3_z',
                  'Contact4_x', 'Contact4_y', 'Contact4_z']

    repl = replay_trajectory(dt_res, joint_list, q_res, contact_map_sol_res, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)
    repl.replay()

else:
    print("To visualize the robot trajectory, start the script with the '--replay")


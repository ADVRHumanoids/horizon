#!/usr/bin/env python
import logging

import rospy
import casadi as cs
import numpy as np
from horizon import problem
from horizon.utils import utils, kin_dyn, resampler_trajectory
from horizon.transcriptions import integrators
from horizon.solvers import solver
from horizon.ros.replay_trajectory import *
import matplotlib.pyplot as plt
import os
import time
from horizon.ros import utils as horizon_ros_utils

horizon_ros_utils.roslaunch("horizon_examples", "quadruped_template.launch")
time.sleep(3.)


urdffile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'urdf', 'quadruped_template.urdf')
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

# OPTIMIZATION PARAMETERS
ns = 40  # number of shooting nodes
nc = 4  # number of contacts
nq = kindyn.nq()  # number of DoFs - NB: 7 DoFs floating base (quaternions)
DoF = nq - 7  # Contacts + anchor_rope + rope
nv = kindyn.nv()  # Velocity DoFs
nf = 3  # Force DOfs

# Create horizon problem
prb = problem.Problem(ns)

# Creates problem STATE variables
q = prb.createStateVariable("q", nq)
qdot = prb.createStateVariable("qdot", nv)

# Creates problem CONTROL variables
dt = prb.createVariable("dt", 1, nodes=list(range(0, ns)))

u = prb.createInputVariable("actuated_torques", nv - 6)
f1 = prb.createInputVariable("f1", nf)
f2 = prb.createInputVariable("f2", nf)
f3 = prb.createInputVariable("f3", nf)
f4 = prb.createInputVariable("f4", nf)
tau = cs.vertcat(cs.SX.zeros(6, 1), u)
contact_names = ['Contact1', 'Contact2', 'Contact3', 'Contact4']
fd = kin_dyn.ForwardDynamics(kindyn, contact_names, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
frame_force_mapping = {'Contact1': f1, 'Contact2': f2, 'Contact3': f3, 'Contact4': f4}
qddot = fd.call(q, qdot, tau, frame_force_mapping)
xdot = utils.double_integrator_with_floating_base(q, qdot, qddot)

prb.setDynamics(xdot)
prb.setDt(dt)

# Formulate discrete time dynamics
L = 0.5*cs.dot(qdot, qdot)  # Objective term
dae = {'x': x, 'p': prb.getInput().getVars(), 'ode': xdot, 'quad': L}
F_integrator = integrators.RK4(dae, {}, cs.SX)


# Limits
disp_z = 0.2

q_min = [-10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0,  # Floating base
                  0.1, 0.1, -0.635,
                  0.1, -0.5, -0.635,
                  -0.6, -0.5, -0.635,
                  -0.6, 0.1, -0.635]

q_max = [10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0,  # Floating base
                  0.6, 0.5, -0.635 + disp_z,
                  0.6, -0.1, -0.635 + disp_z,
                  -0.1, -0.1, -0.635 + disp_z,
                  -0.1, 0.5, -0.635 + disp_z]

q_init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0.349999, 0.349999, -0.635,
                   0.349999, -0.349999, -0.635,
                   -0.349999, -0.349999, -0.635,
                   -0.349999, 0.349999, -0.635]
q.setBounds(q_min, q_max)
q.setBounds(q_init, q_init, 0)
q.setInitialGuess(q_init)

qdot_min = (-100.*np.ones(nv)).tolist()
qdot_max = (100.*np.ones(nv)).tolist()
qdot_init = np.zeros(nv).tolist()
qdot.setBounds(qdot_min, qdot_max)
qdot.setBounds(qdot_init, qdot_init, 0)
qdot.setInitialGuess(qdot_init)

u_min = (-1000.*np.ones(nv-6)).tolist()
u_max = (1000.*np.ones(nv-6)).tolist()
u_init = np.zeros(nv-6).tolist()
u.setBounds(u_min, u_max)
u.setInitialGuess(u_init)

f_min = (-10000.*np.ones(nf)).tolist()
f_max = (10000.*np.ones(nf)).tolist()
f_init = np.zeros(nf).tolist()
f1.setBounds(f_min, f_max)
f1.setInitialGuess(f_init)
f2.setBounds(f_min, f_max)
f2.setInitialGuess(f_init)
f3.setBounds(f_min, f_max)
f3.setInitialGuess(f_init)
f4.setBounds(f_min, f_max)
f4.setInitialGuess(f_init)

dt_min = [0.03] #[s]
dt_max = [0.15] #[s]
dt_init = [dt_min]
dt.setBounds(dt_min, dt_max)
dt.setInitialGuess(dt_init)

# SET UP COST FUNCTION
lift_node = 10
touch_down_node = 30
q_fb_trg = np.array([q_init[0], q_init[1], q_init[2] + 0.9, 0.0, 0.0, 0.0, 1.0]).tolist()

prb.createCost("jump", 10.*cs.dot(q[0:3] - q_fb_trg[0:3], q[0:3] - q_fb_trg[0:3]), nodes=list(range(lift_node, touch_down_node)))
prb.createCost("min_qdot", 50.*cs.dot(qdot, qdot))
#prb.createCost("min_f", 0.0001*cs.sumsqr(f1+f2+f3+f4), nodes=list(range(0, ns)))
prb.createCost("min_u", 0.0001*cs.sumsqr(u), nodes=list(range(0, ns)))
#f1_prev = f1.getVarOffset(-1)
#f2_prev = f2.getVarOffset(-1)
#f3_prev = f3.getVarOffset(-1)
#f4_prev = f4.getVarOffset(-1)
#prb.createCost("min_df", 0.001*cs.sumsqr(f1-f1_prev+f2-f2_prev+f3-f3_prev+f4-f4_prev), nodes=list(range(1, ns)))

# Constraints
q_prev = q.getVarOffset(-1)
qdot_prev = qdot.getVarOffset(-1)
dt_prev = dt.getVarOffset(-1)

u_prev = u.getVarOffset(-1)
f1_prev = f1.getVarOffset(-1)
f2_prev = f2.getVarOffset(-1)
f3_prev = f3.getVarOffset(-1)
f4_prev = f4.getVarOffset(-1)
tau_prev = cs.vertcat(cs.SX.zeros(6, 1), u_prev)
frame_force_mapping_prev = {'Contact1': f1_prev, 'Contact2': f2_prev, 'Contact3': f3_prev, 'Contact4': f4_prev}
qddot_prev = fd.call(q_prev, qdot_prev, tau_prev, frame_force_mapping_prev)
x_prev, _ = utils.double_integrator_with_floating_base(q_prev, qdot_prev, qddot_prev)

x_int = F_integrator(x0=x_prev, p=cs.vertcat(u_prev, f1_prev, f2_prev, f3_prev, f4_prev), time=dt_prev)
prb.createConstraint("multiple_shooting", x_int["xf"] - x, nodes=list(range(1, ns+1)), bounds=dict(lb=np.zeros(nv+nq), ub=np.zeros(nv+nq)))


#prb.createConstraint("final_velocity", qdot, nodes=ns+1, bounds=dict(lb=np.zeros((nv, 1)), ub=np.zeros((nv, 1))))

# GROUND
mu = 0.8 # friction coefficient
R = np.identity(3, dtype=float) # environment rotation wrt inertial frame

# foot
forces = [f1, f2, f3, f4]
for frame, f in zip(contact_names, forces):
    # BEFORE AND AFTER FLIGHT PHASE
    FK = cs.Function.deserialize(kindyn.fk(frame))
    p = FK(q=q)['ee_pos']
    pd = FK(q=q_init)['ee_pos']
    prb.createConstraint(f"{frame}_before_jump", p - pd, nodes=list(range(0, lift_node)), bounds=dict(lb=[0., 0., 0.], ub=[0., 0., 0.]))
    prb.createConstraint(f"{frame}_after_jump", p - pd, nodes=list(range(touch_down_node, ns+1)), bounds=dict(lb=[0., 0., 0.], ub=[0., 0., 0.]))

    DFK = cs.Function.deserialize(kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))
    v = DFK(q=q, qdot=qdot)['ee_vel_linear']
    prb.createConstraint(f"{frame}_vel_before_jump", v, nodes=list(range(0, lift_node)), bounds=dict(lb=[0., 0., 0.], ub=[0., 0., 0.]))
    prb.createConstraint(f"{frame}_vel_after_jump", v, nodes=list(range(touch_down_node, ns + 1)), bounds=dict(lb=[0., 0., 0.], ub=[0., 0., 0.]))

    fc, fc_lb, fc_ub = kin_dyn.linearized_friction_cone(f, mu, R)
    prb.createConstraint(f"{frame}_friction_cone_before_jump", fc, nodes=list(range(0, lift_node)), bounds=dict(lb=fc_lb, ub=fc_ub))
    prb.createConstraint(f"{frame}_friction_cone_after_jump", fc, nodes=list(range(touch_down_node, ns)), bounds=dict(lb=fc_lb, ub=fc_ub))

    # DURING FLIGHT PHASE
    prb.createConstraint(f"{frame}_no_force_during_jump", f, nodes=list(range(lift_node, touch_down_node)), bounds=dict(lb=[0., 0., 0.], ub=[0., 0., 0.]))

# Create problem
opts = {'ipopt.tol': 0.01,
        'ipopt.constr_viol_tol': 0.01,
        'ipopt.max_iter': 5000,
        'ipopt.linear_solver': 'ma57'}

solver = solver.Solver.make_solver('ipopt', prb, opts)
solver.solve()

solution = solver.getSolutionDict()

q_hist = solution["q"]
qdot_hist = solution["qdot"]
u_hist = solution["actuated_torques"]
tau_hist = np.zeros((qdot_hist.shape[0], qdot_hist.shape[1]-1))
for i in range(tau_hist.shape[1]):
    tau_hist[6:, i] = solution["actuated_torques"][:, i]
f1_hist = solution["f1"]
f2_hist = solution["f2"]
f3_hist = solution["f3"]
f4_hist = solution["f4"]
dt_hist = solution["dt"]

qddot_hist = np.zeros(tau_hist.shape)
FD = kin_dyn.ForwardDynamics(kindyn, ['Contact1', 'Contact2', 'Contact3', 'Contact4'], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
for i in range(ns):
    frame_force_mapping_i = {'Contact1': f1_hist[:, i], 'Contact2': f2_hist[:, i], 'Contact3': f3_hist[:, i], 'Contact4': f4_hist[:, i]}
    qddot_hist[:, i] = FD.call(q_hist[:, i], qdot_hist[:, i], tau_hist[:, i], frame_force_mapping_i).toarray().flatten()

# print(q_hist.shape)
# print(qdot_hist.shape)
# print(np.array(cs.vertcat(u_hist, f1_hist, f2_hist, f3_hist, f4_hist).shape))
# exit()
# # resampling
dt = 0.001
frame_force_hist_mapping = {'Contact1': f1_hist, 'Contact2': f2_hist, 'Contact3': f3_hist, 'Contact4': f4_hist}
q_res, qdot_res, input_res = resampler_trajectory.second_order_resample_integrator(q_hist, qdot_hist, np.array(cs.vertcat(u_hist, f1_hist, f2_hist, f3_hist, f4_hist)),
    dt_hist.flatten(), dt, dae)

tau_res = resampler_trajectory.resample_input(tau_hist, dt_hist.flatten(), dt)
f1_res = resampler_trajectory.resample_input(f1_hist, dt_hist.flatten(), dt)
f2_res = resampler_trajectory.resample_input(f2_hist, dt_hist.flatten(), dt)
f3_res = resampler_trajectory.resample_input(f3_hist, dt_hist.flatten(), dt)
f4_res = resampler_trajectory.resample_input(f4_hist, dt_hist.flatten(), dt)

frame_force_res_mapping = {'Contact1': f1_res, 'Contact2': f2_res, 'Contact3': f3_res, 'Contact4': f4_res}



PLOTS = True
if PLOTS:
    time = np.arange(0.0, q_res.shape[1]*dt, dt)

    plt.figure()
    for i in range(3):
        plt.plot(time, q_res[i, :])
    plt.suptitle('$\mathrm{Base \ Position \ Resampled}$', size=20)
    plt.xlabel('$\mathrm{[sec]}$', size=20)
    plt.ylabel('$\mathrm{[m]}$', size=20)

    plt.figure()
    for i in range(6):
        plt.plot(tau_hist[i, :])
    plt.suptitle('$\mathrm{Base \ Forces}$', size=20)
    plt.xlabel('$\mathrm{sample}$', size=20)
    plt.ylabel('$\mathrm{[N]}$', size=20)

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

    #plt.figure()
    #for i in range(qddot_res.shape[0]):
    #    plt.plot(time[:-1], qddot_res[i, :])
    #plt.suptitle('$\mathrm{\ddot{q} \ Resampled}$', size=20)
    #plt.xlabel('$\mathrm{[sec]}$', size=20)
    #plt.ylabel('$\mathrm{\ddot{q}}$', size=20)

    plt.show()




joint_list = ['Contact1_x', 'Contact1_y', 'Contact1_z',
              'Contact2_x', 'Contact2_y', 'Contact2_z',
              'Contact3_x', 'Contact3_y', 'Contact3_z',
              'Contact4_x', 'Contact4_y', 'Contact4_z']


repl = replay_trajectory(dt, joint_list, q_res, frame_force_res_mapping, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)
repl.sleep(1.)
repl.replay()
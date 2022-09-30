#!/usr/bin/env python
import logging

import rospy
import casadi as cs
import numpy as np
from horizon import problem
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter
from horizon.ros.replay_trajectory import *
from horizon.transcriptions import integrators
from horizon.solvers import solver
import matplotlib.pyplot as plt
import os, rospkg
import time
from horizon.ros import utils as horizon_ros_utils

# Loading URDF model in pinocchio
r = rospkg.RosPack()
path_to_examples = r.get_path('horizon_examples')
urdffile = os.path.join(path_to_examples, 'urdf', 'roped_template.urdf')
urdf = open(urdffile, 'r').read()

kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

plot_sol = True
# OPTIMIZATION PARAMETERS
ns = 70  # number of shooting nodes
nc = 3  # number of contacts
nq = kindyn.nq()  # number of DoFs - NB: 7 DoFs floating base (quaternions)
DoF = nq - 7  # Contacts + anchor_rope + rope
nv = kindyn.nv()  # Velocity DoFs
nf = 3  # 2 feet contacts + rope contact with wall, Force DOfs

# Create horizon problem
prb = problem.Problem(ns)

# Creates problem STATE variables
q = prb.createStateVariable("q", nq)
qdot = prb.createStateVariable("qdot", nv)

# Creates problem CONTROL variables
qddot = prb.createInputVariable("qddot", nv)
f1 = prb.createInputVariable("f1", nf)
f2 = prb.createInputVariable("f2", nf)
frope = prb.createInputVariable("frope", nf)

# Node times
dt = prb.createVariable("dt", 1, nodes=list(range(0, ns)))

# Creates double integrator
xdot = utils.double_integrator_with_floating_base(q, qdot, qddot)
prb.setDynamics(xdot)
prb.setDt(dt)

# Formulate discrete time dynamics
L = 0.5*cs.dot(qdot, qdot)  # Objective term
dae = {'x': x, 'p': qddot, 'ode': xdot, 'quad': L}
F_integrator = integrators.RK4(dae, opts=None, casadi_type=cs.SX)

# Set bounds and initial guess to variables
foot_z_offset = 0.5
q_min = [-10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0,  # Floating base
         -0.3, -0.1, -0.1,#-foot_z_offset,  # Contact 1
         -0.3, -0.05, -0.1,#-foot_z_offset,  # Contact 2
         -1.57, -1.57, -3.1415,  # rope_anchor
          0.3]  # rope
q_max = [10.0,  10.0,  10.0,  1.0,  1.0,  1.0,  1.0,  # Floating base
          0.3, 0.05, 0.1+foot_z_offset,  # Contact 1
          0.3, 0.1, 0.1+foot_z_offset,  # Contact 2
          1.57, 1.57, 3.1415,  # rope_anchor
          0.3]  # rope
q.setBounds(q_min, q_max)
alpha = 0.3
rope_lenght = 0.3
x_foot = rope_lenght * np.sin(alpha)
q_init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
           x_foot, 0., 0.,
           x_foot, 0., 0.,
           0., alpha, 0.,
           rope_lenght]
q.setInitialGuess(q_init)
q.setBounds(q_init, q_init, 0)


qdot.setBounds(-100.*np.ones(nv), 100.*np.ones(nv))
qdot.setInitialGuess(np.zeros(nv))
qdot.setBounds(np.zeros(nv), np.zeros(nv), 0)
qdot.setBounds(np.zeros(nv), np.zeros(nv), ns+1)

qddot.setBounds(-100.*np.ones(nv), 100.*np.ones(nv))
qddot.setInitialGuess(np.zeros(nv))
f1.setBounds(-10000.*np.ones(nf), 10000.*np.ones(nf))
f1.setInitialGuess(np.zeros(nf))
f2.setBounds(-10000.*np.ones(nf), 10000.*np.ones(nf))
f2.setInitialGuess(np.zeros(nf))
frope.setBounds(-10000.*np.ones(nf), 10000.*np.ones(nf))
frope.setInitialGuess(np.zeros(nf))

dt.setBounds([0.01], [0.08])
dt.setInitialGuess([0.01])


# Cost function
lift_node = 3 #20
touch_down_node = 60

x_distance = -0.4
prb.createCost("wall_distance", 100.*cs.sumsqr(q[0] - x_distance), nodes=range(lift_node, ns+1))
prb.createCost("min_qdot", cs.sumsqr(qdot))
#prb.createCost("min_legs_vel", cs.sumsqr(qdot[6:12]), nodes=list(range(lift_node, touch_down_node)))
#prb.createCost("min_base_vel", cs.sumsqr(qdot[0:6]), nodes=list(range(lift_node, ns+1)))
f1_prev = f1.getVarOffset(-1)
f2_prev = f2.getVarOffset(-1)
prb.createCost("min_df", 0.0001*cs.sumsqr(f1-f1_prev + f2-f2_prev), nodes=range(1, ns))

qddot_prev = qddot.getVarOffset(-1)
#prb.createCost("min_jerk", 0.0003*cs.sumsqr(qddot-qddot_prev), nodes=list(range(1, ns)))

# Constraints
state = prb.getState()
state_prev = state.getVarOffset(-1)
x_prev, _ = utils.double_integrator_with_floating_base(state_prev[0], state_prev[1], qddot_prev)
dt_prev = dt.getVarOffset(-1)
x_int = F_integrator(x0=x_prev, p=qddot_prev, time=dt_prev)
prb.createConstraint("multiple_shooting", x_int["xf"] - x, nodes=range(1, ns+1))


tau_min = np.array([0., 0., 0., 0., 0., 0.,  # Floating base
            -1000., -1000., -1000.,  # Contact 1
            -1000., -1000., -1000.,  # Contact 2
            0., 0., 0.,  # rope_anchor
            -10000.])  # rope

tau_max = - tau_min

dd = {'Contact1': f1, 'Contact2': f2, 'rope_anchor2': frope}
tau = kin_dyn.InverseDynamics(kindyn, dd.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED).call(q, qdot, qddot, dd)
prb.createConstraint("inverse_dynamics", tau, nodes=range(0, ns), bounds=dict(lb=tau_min, ub=tau_max))

FKRope = cs.Function.deserialize(kindyn.fk('rope_anchor2'))
p_rope_init = FKRope(q=q_init)['ee_pos']
p_rope = FKRope(q=q)['ee_pos']
prb.createConstraint("rope_anchor_point", p_rope-p_rope_init)

# WALL
mu = 0.5
R_wall = np.zeros([3, 3])
rot = -np.pi/2.
R_wall[0, 0] = np.cos(rot)
R_wall[0, 2] = np.sin(rot)
R_wall[1, 1] = 1.
R_wall[2, 0] = -np.sin(rot)
R_wall[2, 2] = np.cos(rot)


contact_names = ['Contact1', 'Contact2']
forces = [f1, f2]
for frame, f in zip(contact_names, forces):
    FK = cs.Function.deserialize(kindyn.fk(frame))
    p = FK(q=q)['ee_pos']
    pd = FK(q=q_init)['ee_pos']

    # STANCE PHASE
    prb.createConstraint(f"{frame}_before_jump", p - pd, nodes=range(0, lift_node))

    fc, fc_lb, fc_ub = kin_dyn.linearized_friction_cone(f, mu, R_wall)
    prb.createConstraint(f"{frame}_friction_cone_before_jump", fc, nodes=range(0, lift_node), bounds=dict(lb=fc_lb, ub=fc_ub))
    

    # FLIGHT PHASE
    prb.createConstraint(f"{frame}_no_force_during_jump", f, nodes=range(lift_node, touch_down_node))
    # TOUCH DOWN PHASE
    prb.createConstraint(f"{frame}_friction_cone_after_jump", fc, nodes=range(touch_down_node, ns),
                         bounds=dict(lb=fc_lb, ub=fc_ub))

    DFK = cs.Function.deserialize(kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))
    v = DFK(q=q, qdot=qdot)['ee_vel_linear']
    prb.createConstraint(f"zero_{frame}_vel_after_jump", v, nodes=range(touch_down_node, ns + 1))

    surface_dict = {'a': 1., 'd': -x_foot}
    c, lb, ub = kin_dyn.surface_point_contact(surface_dict, q, kindyn, frame)
    prb.createConstraint(f"{frame}_on_wall", c, nodes=range(touch_down_node, ns + 1), bounds=dict(lb=lb, ub=ub))

# Creates problem
opts = {'ipopt.tol': 0.01,
        'ipopt.constr_viol_tol': 0.01,
        'ipopt.max_iter': 5000,
        'ipopt.linear_solver': 'ma57'}

solver = solver.Solver.make_solver('ipopt', prb, opts)
solver.solve()

solution = solver.getSolutionDict()


if plot_sol:
    # plots raw solution

    hplt = plotter.PlotterHorizon(prb, solution)
    hplt.plotVariables(['f1', 'f2', 'frope'])
    # hplt.plotFunctions()
    plt.show()


q_hist = solution["q"]
qdot_hist = solution["qdot"]
qddot_hist = solution["qddot"]
f1_hist = solution["f1"]
f2_hist = solution["f2"]
frope_hist = solution["frope"]
dt_hist = solution["dt"]

tau_hist = np.zeros(qddot_hist.shape)
ID = kin_dyn.InverseDynamics(kindyn, ['Contact1', 'Contact2', 'rope_anchor2'], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
for i in range(ns):
    frame_force_mapping_i = {'Contact1': f1_hist[:, i], 'Contact2': f2_hist[:, i], 'rope_anchor2': frope_hist[:, i]}
    tau_hist[:, i] = ID.call(q_hist[:, i], qdot_hist[:, i], qddot_hist[:, i], frame_force_mapping_i).toarray().flatten()


# resampling
dt = 0.001
frame_force_hist_mapping = {'Contact1': f1_hist, 'Contact2': f2_hist, 'rope_anchor2': frope_hist}
q_res, qdot_res, qddot_res, frame_force_res_mapping, tau_res = resampler_trajectory.resample_torques(q_hist, qdot_hist, qddot_hist,
                                                                                                     dt_hist.flatten(), dt, dae, frame_force_hist_mapping,
                                                                                                     kindyn, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)




# REPLAY TRAJECTORY
joint_list = ['Contact1_x', 'Contact1_y', 'Contact1_z',
              'Contact2_x', 'Contact2_y', 'Contact2_z',
              'rope_anchor1_1_x', 'rope_anchor1_2_y', 'rope_anchor1_3_z',
              'rope_joint']

repl = replay_trajectory(dt, joint_list, q_res, frame_force_res_mapping)
repl.sleep(1.)
repl.replay()










#!/usr/bin/env python
import logging

import os
from horizon import problem
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter
from horizon.transcriptions import integrators
from horizon.solvers import solver
from horizon.ros.replay_trajectory import *
import matplotlib.pyplot as plt
from horizon.ros import utils as horizon_ros_utils
import rospkg

horizon_ros_utils.roslaunch("horizon_examples", "roped_template.launch")
time.sleep(3.)

r = rospkg.RosPack()
path_to_examples = r.get_path('horizon_examples')
urdffile = os.path.join(path_to_examples, 'urdf', 'roped_template.urdf')
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)
FKRope = cs.Function.deserialize(kindyn.fk('rope_anchor2'))

# OPTIMIZATION PARAMETERS
ns = 75  # number of shooting nodes
nq = kindyn.nq()  # number of DoFs - NB: 7 DoFs floating base (quaternions)
DoF = nq - 7  # Contacts + anchor_rope + rope
nv = kindyn.nv()  # Velocity DoFs
nf = 3  # 2 feet contacts + rope contact with wall, Force DOfs

# Create horizon problem
prb = problem.Problem(ns, logging_level=logging.DEBUG)

# Creates problem STATE variables
q = prb.createStateVariable("q", nq)
qdot = prb.createStateVariable("qdot", nv)

# Creates problem CONTROL variables
qddot = prb.createInputVariable("qddot", nv)
f1 = prb.createInputVariable("f1", nf)
f2 = prb.createInputVariable("f2", nf)
frope = prb.createInputVariable("frope", nf)

# Creates double integrator
xdot = utils.double_integrator_with_floating_base(q, qdot, qddot)

prb.setDynamics(xdot)
# Formulate discrete time dynamics
tf = 2.0  # [s]
dt = tf / ns
prb.setDt(dt)

L = 0.5 * cs.sumsqr(qdot)  # Objective term
dae = {'x': x, 'p': qddot, 'ode': xdot, 'quad': L}
F_integrator_LEAPFROG = integrators.LEAPFROG(dae)
F_integrator = integrators.RK4(dae)

# Bounds and initial guess
q_min = [-10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0,  # Floating base
         0.0, 0.0, 0.0,  # Contact 1
         0.0, 0.0, 0.0,  # Contact 2
         -1.57, -1.57, -3.1415,  # rope_anchor
         0.3]  # rope
q_max = [10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0,  # Floating base
         0.0, 0.0, 0.0,  # Contact 1
         0.0, 0.0, 0.0,  # Contact 2
         1.57, 1.57, 3.1415,  # rope_anchor
         0.3]  # rope
q_init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
          0., 0., 0.,
          0., 0., 0.,
          0., 0.3, 0.,
          0.3]

q.setBounds(q_min, q_max)
q.setInitialGuess(q_init)

qdot_min = (-1000. * np.ones(nv)).tolist()
qdot_max = (1000. * np.ones(nv)).tolist()
qdot.setBounds(qdot_min, qdot_max)
qdot_init = np.zeros(nv).tolist()
qdot.setInitialGuess(qdot_init)

qddot_min = (-1000. * np.ones(nv)).tolist()
qddot_max = (1000. * np.ones(nv)).tolist()
qddot.setBounds(qddot_min, qddot_max)
qddot_init = np.zeros(nv).tolist()
qddot.setInitialGuess(qdot_init)

f_min = np.zeros(nf).tolist()
f_max = np.zeros(nf).tolist()
f1.setBounds(f_min, f_max)
f2.setBounds(f_min, f_max)
f_init = np.zeros(nf).tolist()
f1.setInitialGuess(f_init)
f2.setInitialGuess(f_init)

f_minRope = (-10000. * np.ones(nf)).tolist()
f_maxRope = (10000. * np.ones(nf)).tolist()
frope.setBounds(f_minRope, f_maxRope)
frope.setInitialGuess(np.zeros(nf).tolist())

# Cost function
prb.createCost("min_joint_vel", 1. * cs.dot(qdot, qdot))

# Constraints
# Initial State
prb.createConstraint("q_init", q - q_init, nodes=0)
prb.createConstraint("qdot_init", qdot - qdot_init, nodes=0)

# MULTIPLE SHOOTING
q_prev = q.getVarOffset(-1)
qdot_prev = qdot.getVarOffset(-1)
qddot_prev = qddot.getVarOffset(-1)
x_prev, _ = utils.double_integrator_with_floating_base(q_prev, qdot_prev, qddot_prev)
x_int = F_integrator(x0=x_prev, p=qddot_prev, time=dt)

prb.createConstraint("multiple_shooting", x_int["xf"] - x, nodes=list(range(1, 2)))

q_pprev = q.getVarOffset(-2)
qdot_pprev = qdot.getVarOffset(-2)
qddot_pprev = qddot.getVarOffset(-2)
x_pprev, _ = utils.double_integrator_with_floating_base(q_pprev, qdot_pprev, qddot_pprev)
x_int2 = F_integrator_LEAPFROG(x0=x_prev, x0_prev=x_pprev, p=qddot_prev, time=dt)
prb.createConstraint("multiple_shooting2", x_int2["xf"] - x, nodes=list(range(2, ns + 1)))

# INVERSE DYNAMICS
tau_min = [0., 0., 0., 0., 0., 0.,  # Floating base
           -1000., -1000., -1000.,  # Contact 1
           -1000., -1000., -1000.,  # Contact 2
           0., 0., 0.,  # rope_anchor
           -10000.]  # rope

tau_max = [0., 0., 0., 0., 0., 0.,  # Floating base
           1000., 1000., 1000.,  # Contact 1
           1000., 1000., 1000.,  # Contact 2
           0., 0., 0.,  # rope_anchor
           0.0]  # rope

frame_force_mapping = {'rope_anchor2': frope}
tau = kin_dyn.InverseDynamics(kindyn, ['rope_anchor2'], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED).call(q,
                                                                                                           qdot,
                                                                                                           qddot,
                                                                                                           frame_force_mapping)
prb.createConstraint("inverse_dynamics", tau, nodes=list(range(0, ns)), bounds=dict(lb=tau_min, ub=tau_max))

# ROPE CONTACT CONSTRAINT
p_rope_init = FKRope(q=q_init)['ee_pos']
p_rope = FKRope(q=q)['ee_pos']
prb.createConstraint("rope_anchor_point", p_rope - p_rope_init)

# SETUP SOLVER
opts = {
    'ipopt.tol': 1e-3,
    'ipopt.constr_viol_tol': 1e-3,
    'ipopt.max_iter': 4000}

solver = solver.Solver.make_solver('ipopt', prb, opts)
solver.solve()
solution = solver.getSolutionDict()

q_hist = solution["q"]
qdot_hist = solution["qdot"]
qddot_hist = solution["qddot"]
f1_hist = solution["f1"]
f2_hist = solution["f2"]
frope_hist = solution["frope"]

tau_hist = np.zeros(qddot_hist.shape)
ID = kin_dyn.InverseDynamics(kindyn, ['rope_anchor2'], cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
for i in range(ns):
    frame_force_mapping_i = {'rope_anchor2': frope_hist[:, i]}
    tau_hist[:, i] = ID.call(q_hist[:, i], qdot_hist[:, i], qddot_hist[:, i], frame_force_mapping_i).toarray().flatten()

PRINT = True
if PRINT:
    # plots raw solution
    time = np.arange(0.0, tf + 1e-6, tf / ns)

    plt.figure()
    for i in range(0, 6):
        plt.plot(time[:-1], tau_hist[i, :], '-x')
    plt.suptitle('$\mathrm{base \ force}$', size=20)
    plt.xlabel('$\mathrm{[sec]}$', size=20)
    plt.ylabel('$\mathrm{ [N] }} $', size=20)

    plt.figure()
    plt.plot(time[:-1], frope_hist[0, :])
    plt.plot(time[:-1], frope_hist[1, :])
    plt.plot(time[:-1], frope_hist[2, :])
    plt.suptitle('$\mathrm{rope \ force}$', size=20)
    plt.xlabel('$\mathrm{[sec]}$', size=20)
    plt.ylabel('$\mathrm{ [N] }} $', size=20)

    plt.rcParams.update({
        "text.usetex": True,
        "mathtext.fontset": "cm",
        "legend.fontsize": 20
    })

    plt.figure()
    KE = cs.Function.deserialize(kindyn.kineticEnergy())
    PE = cs.Function.deserialize(kindyn.potentialEnergy())
    ke_hist = np.zeros((q_hist.shape[1]))
    pe_hist = np.zeros((q_hist.shape[1]))
    tote_hist = np.zeros((q_hist.shape[1]))
    for i in range(q_hist.shape[1]):
        ke_hist[i] = KE(q=q_hist[:, i], v=qdot_hist[:, i])["DT"]
        pe_hist[i] = PE(q=q_hist[:, i])["DU"]
        tote_hist[i] = ke_hist[i] + pe_hist[i]
    plt.suptitle('$\mathrm{Energy}$', size=20)
    plt.plot(time, ke_hist, linewidth=2.0, label='$\mathrm{Kinetic}$')
    plt.plot(time, pe_hist, linewidth=2.0, label='$\mathrm{Potential}$')
    plt.plot(time, tote_hist, linewidth=2.0, label='$\mathrm{Total}$')
    plt.legend(loc='upper center', fancybox=True, framealpha=0.5, ncol=3)
    axes = plt.gca()
    axes.set_ylim([-110.0, 40.])
    plt.grid()
    plt.xlabel('$\mathrm{[sec]}$', size=20)
    plt.ylabel('$\mathrm{[J]}$', size=20)

    plt.figure()
    MP = cs.Function.deserialize(kindyn.fk('rope_anchor1_3'))
    COM = cs.Function.deserialize(kindyn.centerOfMass())
    mp_hist = np.zeros((3, q_hist.shape[1]))
    com_hist = np.zeros((3, q_hist.shape[1]))
    for i in range(q_hist.shape[1]):
        mp_hist[:, i] = (MP(q=q_hist[:, i])['ee_pos']).T
        com_hist[:, i] = (COM(q=q_hist[:, i], v=qdot_hist[:, i], a=qdot_hist[:, i])['com']).T
    plt.suptitle('$\mathrm{Base \ and \ COM  \ trajectories}$', size=20)
    plt.plot(time, mp_hist[0, :], linewidth=2.0, label='$\mathrm{Base \ x}$ ', linestyle='--')
    plt.plot(time, mp_hist[2, :], linewidth=2.0, label='$\mathrm{Base \ z}$', linestyle='--')
    plt.plot(time, com_hist[0, :], linewidth=2.0, label='$\mathrm{COM \ x}$')
    plt.plot(time, com_hist[2, :], linewidth=2.0, label='$\mathrm{COM \ z}$')
    plt.legend(loc='upper center', fancybox=True, framealpha=0.5, ncol=2)
    axes = plt.gca()
    axes.set_ylim([-0.2, 0.5])
    plt.grid()
    plt.xlabel('$\mathrm{[m]}$', size=20)
    plt.ylabel('$\mathrm{[sec]}$', size=20)

    plt.show()

inbuild_plot = False
if inbuild_plot:
    hplt = plotter.PlotterHorizon(prb, solution)
    hplt.plotVariables()
    hplt.plotFunctions()
    plt.show()

# REPLAY TRAJECTORY
# resampling
dt = 0.001
frame_force_hist_mapping = {'Contact1': f1_hist, 'Contact2': f2_hist, 'rope_anchor2': frope_hist}
q_res, qdot_res, qddot_res, frame_force_res_mapping, tau_res = resampler_trajectory.resample_torques(q_hist, qdot_hist,
                                                                                                     qddot_hist,
                                                                                                     tf / ns, dt, dae,
                                                                                                     frame_force_hist_mapping,
                                                                                                     kindyn,
                                                                                                     cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

joint_list = ['Contact1_x', 'Contact1_y', 'Contact1_z',
              'Contact2_x', 'Contact2_y', 'Contact2_z',
              'rope_anchor1_1_x', 'rope_anchor1_2_y', 'rope_anchor1_3_z',
              'rope_joint']

replay_trajectory(dt, joint_list, q_res, frame_force_res_mapping, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn).replay()

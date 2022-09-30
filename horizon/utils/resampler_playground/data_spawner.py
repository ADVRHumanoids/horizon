#!/usr/bin/env python3
from horizon import problem
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter, mat_storer
from horizon.transcriptions.transcriptor import Transcriptor
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from horizon.solvers import solver
import os, argparse
from itertools import filterfalse
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from matplotlib import gridspec

def _trj(tau):
    return 64. * tau ** 3 * (1 - tau) ** 3

def compute_polynomial_trajectory(nodes, p_start, p_goal, clearance, dim=None):
    if dim is None:
        dim = [0, 1, 2]

    # todo check dimension of parameter before assigning it

    traj_array = np.zeros(len(nodes))

    start = p_start[dim]
    goal = p_goal[dim]

    index = 0
    for k in range(len(nodes)):
        tau = k / len(nodes)
        trj = _trj(tau) * clearance
        trj += (1 - tau) * start + tau * goal
        traj_array[index] = trj
        index = index + 1

    return np.array(traj_array)


path_to_examples = os.path.abspath(__file__ + "/../../examples")
ms = mat_storer.matStorer(f'{os.path.splitext(os.path.basename(__file__))[0]}.mat')
# options
transcription_method = 'multiple_shooting'
transcription_opts = dict(integrator='RK4')

tf = 2.5
n_nodes = 50

disp = [0., 0., 0., 0., 0., 0., 1.]

# load urdf
urdffile = os.path.join(path_to_examples, 'urdf', 'spot.urdf')
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

# joint names
joint_names = kindyn.joint_names()
if 'universe' in joint_names: joint_names.remove('universe')
if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')

contacts_name = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']

# parameters
n_c = 4
n_q = kindyn.nq()
n_v = kindyn.nv()
n_f = 3
dt = tf / n_nodes

# define dynamics
prb = problem.Problem(n_nodes, casadi_type=cs.MX, abstract_casadi_type=cs.SX)
q = prb.createStateVariable('q', n_q)
q_dot = prb.createStateVariable('q_dot', n_v)
q_ddot = prb.createInputVariable('q_ddot', n_v)
f_list = [prb.createInputVariable(f'force_{i}', n_f) for i in contacts_name]

x_dot = utils.double_integrator(q, q_dot, q_ddot, kindyn)
prb.setDynamics(x_dot)
prb.setDt(dt)
# contact map
contact_map = dict(zip(contacts_name, f_list))

# initial state and initial guess
q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0.0, 0.9, -1.5238505,
                   0.0, 0.9, -1.5202315,
                   0.0, 0.9, -1.5300265,
                   0.0, 0.9, -1.5253125])

q.setBounds(q_init, q_init, 0)
q_dot.setBounds(np.zeros(n_v), np.zeros(n_v), 0)

q.setInitialGuess(q_init)

[f.setInitialGuess([0, 0, 55]) for f in f_list]

# transcription
th = Transcriptor.make_method(transcription_method, prb, opts=transcription_opts)

# dynamic feasibility
id_fn = kin_dyn.InverseDynamics(kindyn, contact_map.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
tau = id_fn.call(q, q_dot, q_ddot, contact_map)
prb.createIntermediateConstraint("dynamic_feasibility", tau[:6])

# final velocity is zero
prb.createFinalConstraint('final_velocity', q_dot)

# contact handling

# list of lifted legs
k_all = range(1, n_nodes + 1)

lifted_legs = {'lf_foot': list(range(10, 25)),
               'rf_foot': list(range(25, 40)),
               'rh_foot': [],
               'lh_foot': []}

q_final = q_init

z_param = prb.createParameter('z_param', 1)


for frame, f in contact_map.items():

    nodes_stance = list(filterfalse(lambda k: k in lifted_legs[frame], k_all))
    nodes_swing = lifted_legs[frame]

    FK = kindyn.fk(frame)
    DFK = kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
    DDFK = kindyn.frameAcceleration(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

    p = FK(q=q)['ee_pos']
    p_start = FK(q=q_init)['ee_pos']
    v = DFK(q=q, qdot=q_dot)['ee_vel_linear']
    a = DDFK(q=q, qdot=q_dot)['ee_acc_linear']


    prb.createConstraint(f"{frame}_vel", v, nodes=nodes_stance)
    # prb.createIntermediateCost(f'{frame}_fn', f[2] - 25.0)

    # when lifted force is zero
    fzero = np.zeros(n_f)
    contact_map[frame].setBounds(fzero, fzero, nodes=nodes_swing)

    # clearance
    z_clear = compute_polynomial_trajectory(nodes_swing, p_start, p_start, 0.1, 2)
    prb.createConstraint(f'{frame}_z_traj', p[2] - z_param, nodes_swing)
    i = 0
    for node in nodes_swing:
        z_param.assign(z_clear[i], node)
        i += 1


prb.createFinalConstraint(f"final_nominal_pos", q - q_final)
prb.createResidual("min_q_dot", q_dot)
for f in f_list:
    prb.createIntermediateResidual(f"min_{f.getName()}", cs.sqrt(3e-3) * f)

# =============
# SOLVE PROBLEM
# =============

opts = dict()

opts['ipopt.tol'] = 0.001
opts['ipopt.constr_viol_tol'] = n_nodes * 1e-3
opts['ipopt.max_iter'] = 2000

solv = solver.Solver.make_solver('ipopt', prb, opts)
solv.solve()
solution = solv.getSolutionDict()

fmap = dict()
for frame, wrench in contact_map.items():
    fmap[frame] = solution[f'{wrench.getName()}']

id_fn = kin_dyn.InverseDynamics(kindyn, fmap.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

tau_sol = np.zeros([tau.shape[0], prb.getNNodes() - 1])
for i in range(tau.shape[1]):
    fmap_i = dict()
    for frame, wrench in fmap.items():
        fmap_i[frame] = wrench[:, i]
    tau_i = id_fn.call(solution['q'][:, i], solution['q_dot'][:, i], solution['q_ddot'][:, i], fmap_i)
    tau_sol[:, i] = tau_i.toarray().flatten()


solution.update(dict(dt=solv.getDt()))
solution.update(dict(n_nodes=n_nodes))
solution.update(dict(tau=tau_sol))
ms.store(solution)

# cumulative_dt = np.zeros(len(dt_sol) + 1)
# for i in range(len(dt_sol)):
#     cumulative_dt[i + 1] = dt_sol[i] + cumulative_dt[i]
#
# solution_constraints_dict = dict()

plot_sol = False
# ========================================================
if plot_sol:
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    hplt = plotter.PlotterHorizon(prb, solution)
    # hplt.plotVariables(show_bounds=True, same_fig=True, legend=False)
    hplt.plotVariables([elem.getName() for elem in f_list], show_bounds=True, gather=2, legend=False)
    # hplt.plotFunctions(show_bounds=True, same_fig=True)
    # hplt.plotFunction('inverse_dynamics', show_bounds=True, legend=True, dim=range(6))

    pos_contact_list = list()
    fig = plt.figure()
    fig.suptitle('Contacts')
    gs = gridspec.GridSpec(2, 2)
    i = 0
    for contact in contacts_name:
        ax = fig.add_subplot(gs[i])
        ax.set_title('{}'.format(contact))
        i += 1
        FK = kindyn.fk(contact)
        pos = FK(q=solution['q'])['ee_pos']
        # for dim in range(n_f):
        #     ax.plot(np.atleast_2d(cumulative_dt), np.array(pos[dim, :]), marker="x", markersize=3,
        #             linestyle='dotted')  # marker="x", markersize=3, linestyle='dotted'

    plt.figure()
    for contact in contacts_name:
        FK = kindyn.fk(contact)
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'feet position - plane_xy')
        plt.scatter(np.array(pos[0, :]), np.array(pos[1, :]), linewidth=0.1)

    plt.figure()
    for contact in contacts_name:
        FK = kindyn.fk(contact)
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'feet position - plane_xz')
        plt.scatter(np.array(pos[0, :]), np.array(pos[2, :]), linewidth=0.1)

    plt.show()
# ======================================================
contact_map = {contacts_name[i]: solution[f_list[i].getName()] for i in range(n_c)}

import subprocess
import rospy
from horizon.ros.replay_trajectory import replay_trajectory

os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=spot'])
rospy.loginfo("'spot' visualization started.")


    # remember to run a robot_state_publisher
repl = replay_trajectory(dt, joint_names, solution['q'], contact_map, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)

repl.sleep(1.)
repl.replay(is_floating_base=True)
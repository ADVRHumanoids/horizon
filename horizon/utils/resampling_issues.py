import numpy as np

from horizon import problem
from horizon.variables import Variable, SingleVariable, Parameter, SingleParameter
from horizon.utils import utils, kin_dyn, resampler_trajectory, mat_storer
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.ros.replay_trajectory import *
from horizon.solvers import solver
import horizon.variables as sv
import horizon.functions as fn
import horizon.misc_function as misc
import matplotlib.pyplot as plt
from horizon.solvers import Solver
from itertools import groupby
from operator import itemgetter
from typing import List

def resample(solution, prb, dt_res, contacts_name):

    n_nodes = prb.getNNodes()

    prev_f_list = [prev_solution[f'force_{c_name}'] for c_name in contacts_name]

    solution_res = dict()
    u_res = resampler_trajectory.resample_input(
        solution['u_opt'],
        solution['dt'].flatten(),
        dt_res)

    x_res = resampler_trajectory.resampler(
        solution['x_opt'],
        solution['u_opt'],
        solution['dt'].flatten(),
        dt_res,
        dae=None,
        f_int=prb.getIntegrator())

    for s in prb.getState():
        sname = s.getName()
        off, dim = prb.getState().getVarIndex(sname)
        solution_res[f'{sname}_res'] = x_res[off:off + dim, :]

    for s in prb.getInput():
        sname = s.getName()
        off, dim = prb.getInput().getVarIndex(sname)
        solution_res[f'{sname}_res'] = u_res[off:off + dim, :]

    solution_res['dt_res'] = dt_res
    solution_res['x_opt_res'] = x_res
    solution_res['u_opt_res'] = u_res

    fmap = dict()
    for frame, wrench in contact_map.items():
        fmap[frame] = solution[f'{wrench.getName()}']

    fmap_res = dict()
    for frame, wrench in contact_map.items():
        fmap_res[frame] = solution_res[f'{wrench.getName()}_res']

    f_res_list = list()
    for f in prev_f_list:
        f_res_list.append(resampler_trajectory.resample_input(f, solution['dt'].flatten(), dt_res))

    tau = solution['inverse_dynamics']
    tau_res = np.zeros([tau.shape[0], u_res.shape[1]])

    id = kin_dyn.InverseDynamics(kindyn, fmap_res.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

    for i in range(tau.shape[1]):
        fmap_i = dict()
        for frame, wrench in fmap.items():
            fmap_i[frame] = wrench[:, i]
        tau_i = id.call(solution['q'][:, i], solution['q_dot'][:, i], solution['q_ddot'][:, i], fmap_i)
        tau[:, i] = tau_i.toarray().flatten()

    for i in range(tau_res.shape[1]):
        fmap_res_i = dict()
        for frame, wrench in fmap_res.items():
            fmap_res_i[frame] = wrench[:, i]
        tau_res_i = id.call(solution_res['q_res'][:, i], solution_res['q_dot_res'][:, i],
                            solution_res['q_ddot_res'][:, i],
                            fmap_res_i)
        tau_res[:, i] = tau_res_i.toarray().flatten()

    solution_res['tau'] = tau
    solution_res['tau_res'] = tau_res

    num_samples = tau_res.shape[1]

    nodes_vec = np.zeros([n_nodes])
    for i in range(1, n_nodes):
        nodes_vec[i] = nodes_vec[i - 1] + solution['dt'].flatten()[i - 1]

    nodes_vec_res = np.zeros([num_samples + 1])
    for i in range(1, num_samples + 1):
        nodes_vec_res[i] = nodes_vec_res[i - 1] + dt_res

    return solution_res, nodes_vec, nodes_vec_res, num_samples
# =========================================
transcription_method = 'multiple_shooting'  # direct_collocation # multiple_shooting
transcription_opts = dict(integrator='RK4')

# rospack = rospkg.RosPack()
# rospack.get_path('spot_urdf')
urdffile = '../examples/urdf/spot.urdf'
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

# joint names
joint_names = kindyn.joint_names()
if 'universe' in joint_names: joint_names.remove('universe')
if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')

res_n = 2
n_nodes = res_n * 100
print(f'n nodes: {n_nodes}')
start_move = 0.4
end_move = 0.8
peak_move = 0.6

node_start_step = int(n_nodes * start_move)
node_end_step = int(n_nodes * end_move)
node_peak = int(n_nodes * peak_move)

print(f'taking off at node: {node_start_step}')
print(f'landing at node: {node_end_step}')

jump_height = 0.2

n_c = 4
n_q = kindyn.nq()
n_v = kindyn.nv()
n_f = 3

# SET PROBLEM STATE AND INPUT VARIABLES
prb = problem.Problem(n_nodes, receding=True)
q = prb.createStateVariable('q', n_q)
q_dot = prb.createStateVariable('q_dot', n_v)
q_ddot = prb.createInputVariable('q_ddot', n_v)

contacts_name = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
f_list = [prb.createInputVariable(f'force_{i}', n_f) for i in contacts_name]

# SET CONTACTS MAP
contact_map = dict(zip(contacts_name, f_list))

# SET DYNAMICS
dt = prb.createInputVariable("dt", 1)  # variable dt as input
# dt = 0.01
# Computing dynamics
x_dot = utils.double_integrator_with_floating_base(q, q_dot, q_ddot)
prb.setDynamics(x_dot)
prb.setDt(dt)

# SET BOUNDS
# q bounds
q_min = [-10., -10., -10., -1., -1., -1., -1.]  # floating base
q_min.extend(kindyn.q_min()[7:])
q_min = np.array(q_min)

q_max = [10., 10., 10., 1., 1., 1., 1.]  # floating base
q_max.extend(kindyn.q_max()[7:])
q_max = np.array(q_max)

q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0.0, 0.9, -1.5238505,
                   0.0, 0.9, -1.5202315,
                   0.0, 0.9, -1.5300265,
                   0.0, 0.9, -1.5253125])

# q_dot bounds
q_dot_lim = 100. * np.ones(n_v)
# q_ddot bounds
q_ddot_lim = 100. * np.ones(n_v)
# f bounds
f_lim = 10000. * np.ones(n_f)

dt_min = 0.01  # [s]
dt_max = 0.1  # [s]

# set bounds and of q
q.setBounds(q_min, q_max)
q.setBounds(q_init, q_init, 0)
# set bounds of q_dot
q_dot_init = np.zeros(n_v)
q_dot.setBounds(-q_dot_lim, q_dot_lim)
q_dot.setBounds(q_dot_init, q_dot_init, 0)
# set bounds of q_ddot
q_ddot.setBounds(-q_ddot_lim, q_ddot_lim)
# set bounds of f
# for f in f_list:
#     f.setBounds(-f_lim, f_lim)

f_min = [-10000., -10000., -10.]
f_max = [10000., 10000., 10000.]
for f in f_list:
    f.setBounds(f_min, f_max)
# set bounds of dt
if isinstance(dt, cs.SX):
    dt.setBounds(dt_min, dt_max)

# SET INITIAL GUESS
load_initial_guess = True
# import initial guess if present
if load_initial_guess:
    ms = mat_storer.matStorer('resampling_issues_100.mat')
    prev_solution = ms.load()
    q_ig = np.zeros([q.getDim(), prb.getNNodes()])
    q_ig[:, ::res_n] = prev_solution['q']
    q.setInitialGuess(q_ig)

    q_dot_ig = np.zeros([q_dot.getDim(), prb.getNNodes()])
    q_dot_ig[:, ::res_n] = prev_solution['q_dot']
    q_dot.setInitialGuess(q_dot_ig)

    q_ddot_ig = np.zeros([q_ddot.getDim(), prb.getNNodes()-1])
    q_ddot_ig[:, ::res_n] = prev_solution['q_ddot']
    q_ddot.setInitialGuess(q_ddot_ig)

    f_ig_list = list()
    for f in f_list:
        f_ig = np.zeros([f.getDim(), prb.getNNodes()-1])
        f_ig[:, ::res_n] = prev_solution[f.getName()]
        f.setInitialGuess(f_ig)

    if isinstance(dt, cs.SX):
        dt_ig = np.zeros([dt.getDim(), prb.getNNodes()-1])
        dt_ig[:, ::res_n] = prev_solution['dt']
        dt.setInitialGuess(dt_ig)

else:
    q.setInitialGuess(q_init)
    if isinstance(dt, cs.SX):
        dt.setInitialGuess(dt_min)

# SET TRANSCRIPTION METHOD
th = Transcriptor.make_method(transcription_method, prb, opts=transcription_opts)

# SET INVERSE DYNAMICS CONSTRAINTS
tau_lim = np.array([0., 0., 0., 0., 0., 0.,  # Floating base
                    1000., 1000., 1000.,  # Contact 1
                    1000., 1000., 1000.,  # Contact 2
                    1000., 1000., 1000.,  # Contact 3
                    1000., 1000., 1000.])  # Contact 4

tau = kin_dyn.InverseDynamics(kindyn, contact_map.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED).call(q,
                                                                                                             q_dot,
                                                                                                             q_ddot,
                                                                                                             contact_map)
prb.createIntermediateConstraint("inverse_dynamics", tau, bounds=dict(lb=-tau_lim, ub=tau_lim))

# SET FINAL VELOCITY CONSTRAINT
prb.createFinalConstraint('final_velocity', q_dot)

# SET CONTACT POSITION CONSTRAINTS
active_leg = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']

mu = 1
R = np.identity(3, dtype=float)  # environment rotation wrt inertial frame

fb_during_jump = np.array([q_init[0], q_init[1], q_init[2] + jump_height, 0.0, 0.0, 0.0, 1.0])
q_final = q_init

for frame, f in contact_map.items():
    # 2. velocity of each end effector must be zero
    FK = kindyn.fk(frame)
    p = FK(q=q)['ee_pos']
    p_start = FK(q=q_init)['ee_pos']
    p_goal = p_start + [0., 0., jump_height]
    DFK = kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
    v = DFK(q=q, qdot=q_dot)['ee_vel_linear']
    DDFK = kindyn.frameAcceleration(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
    a = DDFK(q=q, qdot=q_dot)['ee_acc_linear']

    prb.createConstraint(f"{frame}_vel_before_lift", v, nodes=range(0, node_start_step))
    c = prb.createConstraint(f"{frame}_vel_after_lift", v, nodes=range(node_end_step, n_nodes + 1))

    # friction cones must be satisfied
    fc, fc_lb, fc_ub = kin_dyn.linearized_friction_cone(f, mu, R)
    prb.createIntermediateConstraint(f"{frame}_fc_before_lift", fc, nodes=range(0, node_start_step),
                                     bounds=dict(lb=fc_lb, ub=fc_ub))
    cfc = prb.createIntermediateConstraint(f"{frame}_fc_after_lift", fc, nodes=range(node_end_step, n_nodes),
                                           bounds=dict(lb=fc_lb, ub=fc_ub))

    prb.createConstraint(f"{frame}_no_force_during_lift", f, nodes=range(node_start_step, node_end_step))

    prb.createConstraint(f"start_{frame}_leg", p - p_start, nodes=node_start_step)
    prb.createConstraint(f"lift_{frame}_leg", p - p_goal, nodes=node_peak)
    prb.createConstraint(f"land_{frame}_leg", p - p_start, nodes=node_end_step)

# SET COST FUNCTIONS
# prb.createCostFunction(f"jump_fb", 10000 * cs.sumsqr(q[2] - fb_during_jump[2]), nodes=node_start_step)
prb.createCost("min_q_dot", 1. * cs.sumsqr(q_dot))
prb.createFinalCost(f"final_nominal_pos", 1000 * cs.sumsqr(q - q_init))
for f in f_list:
    prb.createIntermediateCost(f"min_{f.getName()}", 0.01 * cs.sumsqr(f))

# prb.createIntermediateCost('min_dt', 100 * cs.sumsqr(dt))

# =============
# SOLVE PROBLEM
# =============
#
opts = {'ipopt.tol': 0.001,
        'ipopt.constr_viol_tol': 0.001,
        'ipopt.max_iter': 2000,
        'ipopt.linear_solver': 'ma57'}

solver = solver.Solver.make_solver('ipopt', prb, opts)

ms = mat_storer.matStorer('resampling_issues.mat')
# ========================================== direct solve ==========================================================
solver.solve()
prev_solution = solver.getSolutionDict()
prev_solution.update(solver.getConstraintSolutionDict())
n_nodes = prb.getNNodes() - 1
prev_dt = solver.getDt().flatten()

dt_res = 0.001
prev_solution_res, nodes_vec, nodes_vec_res, num_samples = resample(prev_solution, prb, dt_res, contacts_name)

info_dict = dict(n_nodes=prb.getNNodes(), times=nodes_vec, times_res=nodes_vec_res, dt=prev_dt, dt_res=dt_res)
ms.store({**prev_solution, **prev_solution_res, **info_dict})

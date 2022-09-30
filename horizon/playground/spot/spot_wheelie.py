import time

from horizon import problem
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter, mat_storer
from horizon.transcriptions import integrators
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.ros.replay_trajectory import *
from horizon.solvers import solver
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from logging import DEBUG

ms = mat_storer.matStorer(f'{os.path.splitext(os.path.basename(__file__))[0]}.mat')

transcription_method = 'direct_collocation' # 'multiple_shooting'  # direct_collocation
transcription_opts = dict(integrator='RK4')

path_to_examples = os.path.abspath(os.path.dirname(__file__) + "/../../examples")
urdffile = os.path.join(path_to_examples, 'urdf', 'spot.urdf')
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

# joint names
joint_names = kindyn.joint_names()
if 'universe' in joint_names: joint_names.remove('universe')
if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')


n_nodes = 200
node_start_wheelie = 100

n_c = 4
n_q = kindyn.nq()
n_v = kindyn.nv()
n_f = 3

load_initial_guess = False

# SET PROBLEM STATE AND INPUT VARIABLES
prb = problem.Problem(n_nodes)  # , logging_level=DEBUG
q = prb.createStateVariable('q', n_q)
q_dot = prb.createStateVariable('q_dot', n_v)
q_ddot = prb.createInputVariable('q_ddot', n_v)

f_list = list()
for i in range(n_c):
    f_list.append(prb.createInputVariable(f'f{i}', n_f))

# SET CONTACTS MAP
contacts_name = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
contact_map = dict(zip(contacts_name, f_list))


# import initial guess if present
if load_initial_guess:
    prev_solution = ms.load()
    q_ig = prev_solution['q']
    q_dot_ig = prev_solution['q_dot']
    q_ddot_ig = prev_solution['q_ddot']
    f_ig_list = list()
    for f in f_list:
        f_ig_list.append(prev_solution[f'f{i}'])

# SET DYNAMICS
# dt = prb.createInputVariable("dt", 1)  # variable dt as input
dt = 0.01
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
q_dot_lim = 200. * np.ones(n_v)
# q_ddot bounds
q_ddot_lim = 200. * np.ones(n_v)
# f bounds
f_lim = 10000. * np.ones(n_f)

dt_min = 0.01  # [s]
dt_max = 0.15  # [s]

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
for f in f_list:
    f.setBounds(-f_lim, f_lim)

# set bounds of dt
# dt.setBounds(dt_min, dt_max)

# SET INITIAL GUESS
if load_initial_guess:
    for node in range(q_ig.shape[1]):
        q.setInitialGuess(q_ig[:, node], node)

    for node in range(q_dot_ig.shape[1]):
        q_dot.setInitialGuess(q_dot_ig[:, node], node)

    for node in range(q_ddot_ig.shape[1]):
        q_ddot.setInitialGuess(q_ddot_ig[:, node], node)

    for f, f_ig in zip(f_list, f_ig_list):
        for node in range(f_ig.shape[1]):
            f.setInitialGuess(f_ig[:, node], node)
else:
    q.setInitialGuess(q_init)

# dt.setInitialGuess(dt_min)
# SET TRANSCRIPTION METHOD
th = Transcriptor.make_method(transcription_method, prb, opts=transcription_opts)

# SET INVERSE DYNAMICS CONSTRAINTS
tau_lim = np.array([0., 0., 0., 0., 0., 0.,  # Floating base
                    1000., 1000., 1000.,  # Contact 1
                    1000., 1000., 1000.,  # Contact 2
                    1000., 1000., 1000.,  # Contact 3
                    1000., 1000., 1000.])  # Contact 4

tau = kin_dyn.InverseDynamics(kindyn, contact_map.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED).call(q, q_dot,
                                                                                                             q_ddot,
                                                                                                             contact_map)
prb.createIntermediateConstraint("inverse_dynamics", tau, bounds=dict(lb=-tau_lim, ub=tau_lim))

# SET FINAL VELOCITY CONSTRAINT
prb.createFinalConstraint('final_velocity', q_dot)
# SET FINAL ACCELERATION CONSTRAINT
prb.createConstraint('final_acceleration', q_ddot, nodes=range(n_nodes-10, n_nodes))
# q_ddot.setBounds(np.zeros(n_v), np.zeros(n_v), nodes=range(n_nodes-10, n_nodes))

# SET CONTACT POSITION CONSTRAINTS
active_leg = ['lf_foot', 'rf_foot']
mu = 1
R = np.identity(3, dtype=float)  # environment rotation wrt inertial frame

# FK = cs.Function.deserialize(kindyn.fk('lf_foot'))
# intial_lf = FK(q=q_init)['ee_pos']
#
# FK = cs.Function.deserialize(kindyn.fk('rf_foot'))
# intial_rf = FK(q=q_init)['ee_pos']

for frame, f in contact_map.items():
    FK = kindyn.fk(frame)
    p = FK(q=q)['ee_pos']
    p_start = FK(q=q_init)['ee_pos']
    p_goal = p_start + [0., 0., 0.3]

    DFK = kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
    v = DFK(q=q, qdot=q_dot)['ee_vel_linear']
    # velocity of active end effector must be zero
    if frame not in active_leg:
        prb.createConstraint(f"{frame}_vel_wheelie", v)
    else:
        prb.createConstraint(f"{frame}_vel_before_wheelie", v, nodes=range(0, node_start_wheelie))

    # friction cones must be satisfied
    fc, fc_lb, fc_ub = kin_dyn.linearized_friction_cone(f, mu, R)
    if frame not in active_leg:
        prb.createIntermediateConstraint(f"{frame}_fc", fc, bounds=dict(lb=fc_lb, ub=fc_ub))
    else:
        prb.createIntermediateConstraint(f"{frame}_fc_before_wheelie", fc, nodes=range(0, node_start_wheelie), bounds=dict(lb=fc_lb, ub=fc_ub))

    if frame in active_leg:
        # zero force for lifted legs
        prb.createConstraint(f"{frame}_no_force_after_lift", f, nodes=range(node_start_wheelie, n_nodes))
        # terrain constraint
        prb.createConstraint(f"subterrain_{frame}_leg", p[2] - p_start[2], bounds=dict(lb=0., ub=100000))
        # suggestion to lift leg
        # prb.createCost(f"lift_{frame}_leg", 10000 * cs.sumsqr(p[2] - p_goal[2]), nodes=node_start_wheelie)

# SET COST FUNCTIONS

# prb.createCost("min_q_ddot", 10000. * cs.sumsqr(q_ddot), nodes=range(node_start_wheelie, n_nodes))

# minimize the joint velocities
prb.createCost("min_q_dot", 1. * cs.sumsqr(q_dot))

# minimize the forces
for f in f_list:
    prb.createIntermediateCost(f"min_{f.getName()}", 0.01 * cs.sumsqr(f))

# desired pose at the end of the wheelie
# fb_rot_wheelie = np.array([0, -0.5, 0, 0.8660254]) # pi/3 pitch
fb_rot_wheelie = np.array([0, -0.8509035, 0, 0.525322])  # pi/2 pitch
# prb.createFinalCost(f"final_nominal_pos", 10000 * cs.sumsqr(q[3:7] - fb_rot_wheelie))
prb.createCost(f"wheelie_fb", 10000 * cs.sumsqr(q[3:7] - fb_rot_wheelie), nodes=node_start_wheelie)
# prb.createFinalCost(f"final_nominal_pos", 10 * cs.sumsqr(q[3:7] - fb_rot_wheelie))
prb.createFinalCost(f"final_nominal_pos", 10000. * cs.sumsqr(q[4] - fb_rot_wheelie[2]))
# prb.createFinalCost(f"final_nominal_pos", 10000. * cs.sumsqr(q[4] - fb_rot_wheelie[2]))

# =============
# SOLVE PROBLEM
# =============
opts = {'ipopt.tol': 0.001,
        'ipopt.constr_viol_tol': 0.001,
        'ipopt.max_iter': 2000,
        'ipopt.linear_solver': 'ma57'}

solver = solver.Solver.make_solver('ipopt', prb, opts)
solver.solve()

solution = solver.getSolutionDict()
solution_constraints = solver.getConstraintSolutionDict()

solution_constraints_dict = dict()
for name, item in prb.getConstraints().items():
    lb, ub = item.getBounds()
    lb_mat = np.reshape(lb, (item.getDim(), len(item.getNodes())), order='F')
    ub_mat = np.reshape(ub, (item.getDim(), len(item.getNodes())), order='F')
    solution_constraints_dict[name] = dict(val=solution_constraints[name], lb=lb_mat, ub=ub_mat, nodes=item.getNodes())

if isinstance(dt, cs.SX):
    ms.store({**solution, **solution_constraints_dict})
else:
    dt_dict = dict(constant_dt=dt)
    ms.store({**solution, **solution_constraints_dict, **dt_dict})

# ========================================================
contact_map = dict(zip(contacts_name, [solution['f0'], solution['f1'], solution['f2'], solution['f3']]))

# resampling
resampling = True
if resampling:

    if isinstance(dt, cs.SX):
        dt_before_res = solution['dt'].flatten()
    else:
        dt_before_res = dt

    dt_res = 0.001
    dae = {'x': x, 'p': q_ddot, 'ode': x_dot, 'quad': 1}
    q_res, qdot_res, qddot_res, contact_map_res, tau_res = resampler_trajectory.resample_torques(
        solution["q"], solution["q_dot"], solution["q_ddot"], dt_before_res, dt_res, dae, contact_map,
        kindyn,
        cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)

    repl = replay_trajectory(dt_res, joint_names, q_res, contact_map_res, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)
else:
    # remember to run a robot_state_publisher
    repl = replay_trajectory(dt, joint_names, solution['q'], contact_map, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)

repl.sleep(1.)
repl.replay(is_floating_base=True)


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

transcription_method = 'direct_collocation'  # direct_collocation #multiple_shooting
transcription_opts = dict(integrator='RK4')

urdffile = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../urdf', 'spot.urdf')
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

# joint names
joint_names = kindyn.joint_names()
if 'universe' in joint_names: joint_names.remove('universe')
if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')


n_nodes = 50

node_start_step = 15
n_nodes_step = 30
node_end_step = node_start_step + n_nodes_step

n_c = 4
n_q = kindyn.nq()
n_v = kindyn.nv()
n_f = 3

load_initial_guess = False

# SET PROBLEM STATE AND INPUT VARIABLES
prb = problem.Problem(n_nodes) # , logging_level=DEBUG

# state variables
q = prb.createStateVariable('q', n_q)
q_dot = prb.createStateVariable('q_dot', n_v)

u = prb.createInputVariable("actuated_torques", n_v - 6)
tau = cs.vertcat(cs.SX.zeros(6, 1), u)

# forces
f_list = list()
for i in range(n_c):
    f_list.append(prb.createInputVariable(f'f{i}', n_f))

# SET CONTACTS MAP
contacts_name = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
contact_map = dict(zip(contacts_name, f_list))
fd = kin_dyn.ForwardDynamics(kindyn, contacts_name, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)


# jac = fd.fd.jacobian()
# print(jac)
# print(jac.jacobian())
# exit()
q_ddot = fd.call(q, q_dot, tau, contact_map)

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
x, x_dot = utils.double_integrator_with_floating_base(q, q_dot, q_ddot)
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

# f bounds
f_lim = 10000. * np.ones(n_f)

# dt bounds
dt_min = 0.01  # [s]
dt_max = 0.15  # [s]

# torques bounds
u_lim = 1000.*np.ones(n_v-6)

#  ==========================================
# set bounds of q
q.setBounds(q_min, q_max)
q.setBounds(q_init, q_init, 0)
# set bounds of q_dot
q_dot_init = np.zeros(n_v)
q_dot.setBounds(-q_dot_lim, q_dot_lim)
q_dot.setBounds(q_dot_init, q_dot_init, 0)
# set bounds of f
for f in f_list:
    f.setBounds(-f_lim, f_lim)
# set bounds of u
u.setBounds(-u_lim, u_lim)
# set bounds of dt
# dt.setBounds(dt_min, dt_max)
# set initial guess of u
u.setInitialGuess(np.zeros(n_v-6))
# dt.setInitialGuess(dt_min)
# SET TRANSCRIPTION METHOD
th = Transcriptor.make_method(transcription_method, prb, opts=transcription_opts)

# SET FINAL VELOCITY CONSTRAINT
prb.createFinalConstraint('final_velocity', q_dot)

mu = 0.8
R = np.identity(3, dtype=float)  # environment rotation wrt inertial frame

# base_link stays at the same position
FK = cs.Function.deserialize(kindyn.fk('base_link'))
p_base = FK(q=q)['ee_pos']
p_base_start = FK(q=q_init)['ee_pos']

prb.createCost(f"base_link_pos", 10 * cs.sumsqr(p_base - p_base_start))

# without this, variable dt sucks
DFK = cs.Function.deserialize(kindyn.frameVelocity('base_link', cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))
v_base_lin = DFK(q=q, qdot=q_dot)['ee_vel_linear']
v_base_rot = DFK(q=q, qdot=q_dot)['ee_vel_angular']

for frame, f in contact_map.items():
    # velocity of each end effector must be zero
    DFK = cs.Function.deserialize(kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))
    v = DFK(q=q, qdot=q_dot)['ee_vel_linear']
    prb.createConstraint(f"{frame}_vel", v)

    # friction cones must be satisfied
    fc, fc_lb, fc_ub = kin_dyn.linearized_friction_cone(f, mu, R)
    prb.createIntermediateConstraint(f"{frame}_fc", fc, bounds=dict(lb=fc_lb, ub=fc_ub))


# SET COST FUNCTIONS
# prb.createCost("min_q_dot", 1. * cs.sumsqr(q_dot))
prb.createFinalCost(f"final_nominal_pos", 1000 * cs.sumsqr(q - q_init))
# =============
# SOLVE PROBLEM
# =============
opts = {'ipopt.tol': 0.001,
        'ipopt.constr_viol_tol': 0.001,
        'ipopt.max_iter': 2000,
        'ipopt.linear_solver': 'ma57'}

solver = solver.Solver.make_solver('ipopt', prb, opts)
exit()
solver.solve()


solution = solver.getSolutionDict()
solution_constraints = solver.getConstraintSolutionDict()

# print(solution['f0'])
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


contacts_name = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
contact_map = dict(zip(contacts_name, [solution['f0'], solution['f1'], solution['f2'], solution['f3']]))

# resampling
resampling = False
if resampling:

    if isinstance(dt, cs.SX):
        dt_before_res = solution['dt'].flatten()
    else:
        dt_before_res = dt

    u_hist = solution["actuated_torques"]
    tau = cs.vertcat(cs.SX.zeros(6, 1), u)
    tau_hist = np.zeros((solution["q_dot"].shape[0], solution["q_dot"].shape[1] - 1))
    for i in range(tau_hist.shape[1]):
        tau_hist[6:, i] = solution["actuated_torques"][:, i]

    f0_hist = solution["f0"]
    f1_hist = solution["f1"]
    f2_hist = solution["f2"]
    f3_hist = solution["f3"]

    qddot_hist = np.zeros(tau_hist.shape)
    FD = kin_dyn.ForwardDynamics(kindyn, contacts_name, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
    for i in range(n_nodes):
        contact_map_i = dict(zip(contacts_name, [solution['f0'][:, i], solution['f1'][:, i], solution['f2'][:, i], solution['f3'][:, i]]))
        qddot_hist[:, i] = FD.call(solution["q"][:, i], solution["q_dot"][:, i], tau_hist[:, i], contact_map_i).toarray().flatten()

    # resampling
    dt_res = 0.001
    dae = {'x': prb.getState().getVars(), 'p': cs.vertcat(u, f_list[0], f_list[1], f_list[2], f_list[3]), 'ode': x_dot, 'quad': 1}

    input = np.array(cs.vertcat(u_hist, solution["f0"], solution["f1"], solution["f2"], solution["f3"]))
    q_res, qdot_res, input_res = resampler_trajectory.second_order_resample_integrator(solution["q"], solution["q_dot"], input, dt_before_res, dt_res, dae)

    tau_res = resampler_trajectory.resample_input(tau_hist, dt_before_res, dt_res)
    f1_res = resampler_trajectory.resample_input(f0_hist, dt_before_res, dt_res)
    f2_res = resampler_trajectory.resample_input(f1_hist, dt_before_res, dt_res)
    f3_res = resampler_trajectory.resample_input(f2_hist, dt_before_res, dt_res)
    f4_res = resampler_trajectory.resample_input(f3_hist, dt_before_res, dt_res)

    contact_map = dict(zip(contacts_name, [f1_res, f2_res, f3_res, f4_res]))

    repl = replay_trajectory(dt_res, joint_names, q_res, contact_map, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)
else:
    # remember to run a robot_state_publisher
    repl = replay_trajectory(dt, joint_names, solution['q'], contact_map, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)

repl.sleep(1.)
repl.replay(is_floating_base=True)


import casadi as cs
import kiwisolver

from horizon.utils import utils, kin_dyn, mat_storer
import numpy as np
import time
import os
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from itertools import filterfalse
import math

def RK4(dae, opts=None, casadi_type=cs.SX):
    if opts is None:
        opts = dict()
    x = dae['x']
    qddot = dae['p']
    xdot = dae['ode']
    L = dae['quad']

    f_RK = cs.Function('f_RK', [x, qddot], [xdot, L])

    nx = x.size1()
    nv = qddot.size1()

    X0_RK = casadi_type.sym('X0_RK', nx)
    U_RK = casadi_type.sym('U_RK', nv)

    if 'tf' in opts:
        DT_RK = opts['tf']
    else:
        DT_RK = casadi_type.sym('DT_RK', 1)

    X_RK = X0_RK
    Q_RK = 0

    k1, k1_q = f_RK(X_RK, U_RK)
    k2, k2_q = f_RK(X_RK + DT_RK / 2. * k1, U_RK)
    k3, k3_q = f_RK(X_RK + DT_RK / 2. * k2, U_RK)
    k4, k4_q = f_RK(X_RK + DT_RK * k3, U_RK)

    X_RK = X_RK + DT_RK / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    Q_RK = Q_RK + DT_RK / 6. * (k1_q + 2. * k2_q + 2. * k3_q + k4_q)

    if 'tf' in opts:
        f = cs.Function('F_RK', [X0_RK, U_RK], [X_RK, Q_RK], ['x0', 'p'], ['xf', 'qf'])
    else:
        f = cs.Function('F_RK', [X0_RK, U_RK, DT_RK], [X_RK, Q_RK], ['x0', 'p', 'time'], ['xf', 'qf'])
    return f

path_to_examples = os.path.abspath(__file__ + "/../../../examples/")
urdffile = os.path.join(path_to_examples, 'urdf', 'spot.urdf')
urdf = open(urdffile, 'r').read()
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

tf = 2.0
n_nodes = 100
t_jump = (1.0, 1.5)
dt = tf / n_nodes

n_c = 4
n_q = kindyn.nq()
n_v = kindyn.nv()
n_f = 3

# ====================
# unraveled dimensions
# ====================
N_states = n_nodes + 1
N_control = n_nodes
q_dim = N_states * n_q
q_dot_dim = N_states * n_v
q_ddot_dim = N_control * n_v
f_dim = N_control * n_c * n_f

q = cs.SX.sym("q", n_q)
q_dot = cs.SX.sym("q_dot", n_v)
q_ddot = cs.SX.sym("q_ddot", n_v)

# forces
f_list = [cs.SX.sym(f"f{i}", n_f) for i in range(n_c)]

# SET CONTACTS MAP
contacts_name = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
contact_map = dict(zip(contacts_name, f_list))


g_tau = kin_dyn.InverseDynamics(kindyn,
                              contact_map.keys(),
                              cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED).call(q, q_dot, q_ddot, contact_map)

fs = cs.vertcat(*f_list)
state = cs.vertcat(q, q_dot)
input = cs.vertcat(q_ddot, fs)

x_dot = utils.double_integrator_with_floating_base(q, q_dot, q_ddot)
dae = {'x': state, 'p': input, 'ode': x_dot, 'quad': 1}
integrator = RK4(dae, opts=dict(tf=dt))

# technically should be faster
# integrator = integrator.expand()
int_map = integrator.map(N_control, 'thread', 15)

opti = cs.Opti()

q_i = opti.variable(n_q, N_states)
q_dot_i = opti.variable(n_v, N_states)

q_ddot_i = opti.variable(n_v, N_control)
f_i = opti.variable(n_f * n_c, N_control)


X = cs.vertcat(q_i, q_dot_i)
U = cs.vertcat(q_ddot_i, f_i)
X_int = int_map(X[:, :n_nodes], U) # because it's N+1
# starting from node 1
g_multi_shoot = X_int[0] - X[:, 1:]

mu = 1
R = np.identity(3, dtype=float)  # environment rotation wrt inertial frame

contact_map_i = dict(lf_foot=f_i[0:3, :],
                     rf_foot=f_i[3:6, :],
                     lh_foot=f_i[6:9, :],
                     rh_foot=f_i[9:12, :])

g_tau_i = kin_dyn.InverseDynamicsMap(n_nodes, kindyn,
                                     contact_map.keys(),
                                     cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED).call(q_i[:, :n_nodes], q_dot_i[:, :n_nodes], q_ddot_i, contact_map_i)


opti.subject_to(g_tau_i[:6, :] == 0.)

opti.subject_to(g_multi_shoot == 0.)

k_all = range(1, n_nodes + 1)
k_swing = list(range(*[int(t / dt) for t in t_jump]))
k_stance = list(filterfalse(lambda k: k in k_swing, k_all))

q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0.0, 0.9, -1.5238505,
                   0.0, 0.9, -1.5202315,
                   0.0, 0.9, -1.5300265,
                   0.0, 0.9, -1.5253125])

q_tgt = q_init.copy()
q_tgt[0] = 0
q_tgt[5] = math.sin(math.pi / 4)
opti.subject_to(q_i[:6, n_nodes] - q_tgt[:6] == 0)


opti.minimize(cs.sumsqr(10 * (q_i[7:, n_nodes] - q_init[7:])))

lifted_legs = contacts_name.copy()

for frame, f in contact_map.items():
    nodes = k_stance if frame in lifted_legs else k_all

    FK = cs.Function.deserialize(kindyn.fk(frame))
    DFK = cs.Function.deserialize(kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))
    DDFK = cs.Function.deserialize(kindyn.frameAcceleration(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))

    v_i = DFK(q=q_i, qdot=q_dot_i)['ee_vel_linear']

    opti.subject_to(v_i[:, nodes] == 0)

# for leg in lifted_legs:
opti.subject_to(f_i[0:3, k_swing] == 0)
opti.subject_to(f_i[3:6, k_swing] == 0)
opti.subject_to(f_i[6:9, k_swing] == 0)
opti.subject_to(f_i[9:12, k_swing] == 0)

for i in range(n_nodes):
    opti.minimize(cs.sumsqr(1e-1 * (q_i[7:, i] - q_init[7:])))

opti.minimize(cs.sumsqr(1e-3 * f_i[0:3, :]))
opti.minimize(cs.sumsqr(1e-3 * f_i[3:6, :]))
opti.minimize(cs.sumsqr(1e-3 * f_i[6:9, :]))
opti.minimize(cs.sumsqr(1e-3 * f_i[9:12, :]))

opti.minimize(cs.sumsqr(1e-3 * q_ddot_i))
# ===============================
# ==== BOUNDS INITIALIZATION ====
# ===============================
# initial guess q
opti.set_initial(q_i[:, 0], q_init)
opti.subject_to(q_i[:, 0] == q_init)
# opti.subject_to(opti.bounded(q_min, q_i, q_max))

# q_dot bounds
opti.subject_to(q_dot_i[:, 0] == 0)
opti.subject_to(q_dot_i[:, n_nodes] == 0)


# f bounds
# f_lim = 1000. * np.ones(n_c * n_f)
# opti.subject_to(opti.bounded(-f_lim, f_i, f_lim))
# opti.subject_to(opti.bounded(0, f_i[2, :], 1000))
# opti.subject_to(opti.bounded(0, f_i[5, :], 1000))
# opti.subject_to(opti.bounded(0, f_i[8, :], 1000))
# opti.subject_to(opti.bounded(0, f_i[11, :], 1000))

s_opts = {
           'ipopt.tol': 0.001,
           'ipopt.constr_viol_tol': 0.001,
           'ipopt.max_iter': 2000,
           'ipopt.linear_solver': 'ma57'}
        # 'verbose': True,
        # banned options:
        # 'ipopt.hessian_approximation': 'limited-memory',
        # 'expand': True,


opti.solver('ipopt', s_opts)

tic = time.time()
sol = opti.solve()
toc = time.time()
print('time elapsed solving:', toc - tic)


solution = dict()
solution['q_i'] = sol.value(q_i)
solution['q_dot_i'] = sol.value(q_dot_i)
solution['q_ddot_i'] = sol.value(q_ddot_i)
solution['f_i'] = sol.value(f_i)


ms = mat_storer.matStorer(f'{os.path.splitext(os.path.basename(__file__))[0]}.mat')

ms.store(dict(a=solution))

from ctypes import util
from threading import Thread
from horizon.problem import Problem
from casadi_kin_dyn import pycasadi_kin_dyn
from horizon.solvers import Solver
from horizon.transcriptions.transcriptor import Transcriptor
import os
import numpy as np
from horizon.ros import replay_trajectory
from horizon.utils import utils, kin_dyn
import casadi as cs
import rospy
import logging


# set up model
path_to_examples = os.path.dirname(os.path.abspath(__file__)) + "/../examples"
os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples


urdffile = os.path.join(path_to_examples, 'urdf', 'spot.urdf')
print(path_to_examples, urdffile, os.path.dirname(__file__))
urdf = open(urdffile, 'r').read()
rospy.set_param('/robot_description', urdf)

base_init = np.array([0., 0., 0.96, 0., 0.0, 0.0, 1.])

q_init = {
}


contact_dict = {
    'l_sole': {
        'type': 'vertex',
        'vertex_frames': [
            'l_foot_lower_left_link',
            'l_foot_upper_left_link',
            'l_foot_lower_right_link',
            'l_foot_upper_right_link',
        ]
    },

    'r_sole': {
        'type': 'vertex',
        'vertex_frames': [
            'r_foot_lower_left_link',
            'r_foot_upper_left_link',
            'r_foot_lower_right_link',
            'r_foot_upper_right_link',
        ]
    }
}


# set up model
urdf = urdf.replace('continuous', 'revolute')

kd = pycasadi_kin_dyn.CasadiKinDyn(urdf)

# definition of the problem --> required by the ti
N = 60
tf = 1.5
dt = tf / N

prb = Problem(N, receding=True)

prb.setDt(dt)

q = prb.createStateVariable('q', kd.nq())
v_var = prb.createStateVariable('v', kd.nv())
a_var = prb.createStateVariable('a', kd.nv())
j_var = prb.createInputVariable('j', kd.nv())

dt_scale = dt
v = v_var/dt_scale
a = a_var/dt_scale**2
j = j_var/dt_scale**3

contacts = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
forces = [prb.createStateVariable(f'force_{c}', 3) for c in contacts]
fdot_var = [prb.createInputVariable(f'fdot_{c}', 3) for c in contacts]
fdot = fdot_var

xdot = cs.vertcat(
    kd.qdot()(q, v_var)/dt_scale,
    a_var/dt_scale,
    j_var/dt_scale,
    *fdot
)

prb.setDynamics(xdot)

# initial cond
q_init = np.array([0.0, 0.0, 0.50, 0.0, 0.0, 0.0, 1.0,
                       0.0, 0.9, -1.52,
                       0.0, 0.9, -1.52,
                       0.0, 0.9, -1.52,
                       0.0, 0.9, -1.52])
vzero = np.zeros(kd.nv())
fzero = np.zeros(3)

q.setBounds(q_init, q_init, 0)
v_var.setBounds(vzero, vzero, 0)
a_var.setBounds(vzero, vzero, 0)

q.setInitialGuess(q_init)
[f.setInitialGuess(np.array([0, 0, 50])) for f in forces]


# dynamic feasibility
kdframe = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
id_fn = kin_dyn.InverseDynamics(kd, contacts, kdframe)
tau = id_fn.call(q, v, a, {c: f for c, f in zip(contacts, forces)})
prb.createConstraint("dynamic_feasibility", tau[:6]*dt_scale**2, nodes=range(1, N+1))



# regularization
prb.createFinalConstraint('final_q', q[7:] - q_init[7:])
prb.createFinalConstraint('final_b', q[:6] - q_init[:6])

prb.createResidual('reg_q', 1e2*(q[7:] - q_init[7:]))
prb.createResidual('reg_v', 1e-3*v)
prb.createResidual('reg_a', 1e-3*a)
prb.createIntermediateResidual('reg_j', 1e-2*j)
[prb.createResidual(f'reg_{f.getName()}', 1e-3*f) for f in forces]
[prb.createIntermediateResidual(f'reg_{f.getName()}', 1e-1*f) for f in fdot]

# final v
# prb.createFinalConstraint('final_v', v_var)

# contacts
steps = [
    (0, 20, 40),
    (1, 20, 40),
    (2, 20, 40),
    (3, 20, 40),
]

for l, kstart, kgoal in steps:
    swing = list(range(kstart, kgoal))
    stance = list(range(kstart)) + list(range(kgoal, N+1))
    fullstance = list(range(kstart-10)) + list(range(kgoal+10, N+1))

    fk_fn = kd.fk(contacts[l])
    dfk_fn = kd.frameVelocity(contacts[l], kdframe)
    pos, rot = fk_fn(q)
    vel, ome = dfk_fn(q, v)

    prb.createConstraint(f'contact_{l}', vel*dt_scale, nodes=stance)
    prb.createConstraint(f'zero_f_{l}', forces[l], nodes=swing)
    fn = forces[l][2]
    prb.createResidual(f'fn1_{l}', 10*cs.if_else(fn > 20., 0, fn - 20), nodes=fullstance)
    prb.createResidual(f'fn2_{l}', 10*cs.if_else(fn > 0., 0, fn), nodes=stance)


solver_type = 'ilqr'

if solver_type != 'ilqr':
    Transcriptor.make_method('multiple_shooting', prb, opts={})

sol = Solver.make_solver(solver_type, prb, opts={
    'ilqr.alpha_min': 0.001,
    'ilqr.max_iter': 100,
    'ilqr.verbose': True,
    # 'ilqr.log': True,
    # 'ilqr.svd_threshold': 1e-6,
    'ilqr.kkt_decomp_type': 'ldlt',
    'ilqr.codegen_enabled': True,
    'ilqr.codegen_working_dir': './codegen',
    'gnsqp.constraint_violation_tolerance': N*1e-6,
    'gnsqp.merit_derivative_tolerance': N*1e-3
})

sol.solve()

from matplotlib import pyplot as plt

for k, v in sol.getSolutionDict().items():
    plt.figure()
    plt.plot(v.T)
    plt.title(k)
    plt.legend([f'{i}' for i in range(v.shape[1])])

# plt.show()

replay_trajectory.replay_trajectory(dt, 
            kd.joint_names(), 
            sol.getSolutionDict()['q'], 
            {c: sol.getSolutionDict()[f'force_{c}'] for c in contacts},
            kdframe,
            kd).replay()
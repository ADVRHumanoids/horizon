import logging
import os
import numpy as np
from horizon.rhc.taskInterface import TaskInterface
from horizon.utils.actionManager import ActionManager
from horizon.problem import Problem
from horizon.rhc.PhaseManager import PhaseManager, Phase
from horizon.solvers import Solver
from horizon.rhc.model_description import FullModelInverseDynamics
from casadi_kin_dyn import pycasadi_kin_dyn
from horizon.transcriptions.transcriptor import Transcriptor
import casadi as cs
import time
import phase_manager.pyphase as pyphase
import phase_manager.pymanager as pymanager
import math
import rospkg, rospy, yaml

from cartesian_interface.pyci_all import *
from xbot_interface import xbot_interface as xbot

# 1. rviz / roscore
# 2. rosrun cartesian_interface marker_spawner
class CartesioSolver:

    def __init__(self, urdf, srdf, pb, kd, solver) -> None:
        kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
        self.base_fk = kd.fk('base_link')
        self.base_dfk = kd.frameVelocity('base_link', kd_frame)

        self.model = xbot.ModelInterface(get_xbot_config(urdf=urdf, srdf=srdf))

        self.solver_rti = solver
        # self.ee_pos = ee_pos

        self.set_model_from_solution()

        self.ci = pyci.CartesianInterface.MakeInstance(
            solver='',
            problem=pb,
            model=self.model,
            dt=0.01)

        roscpp.init('ciao', [])
        self.ciros = pyci.RosServerClass(self.ci)

        self.task = self.ci.getTask('core_link')

    def set_model_from_solution(self):
        q = self.solver_rti.getSolutionDict()['q'][:, 1]
        dq = self.solver_rti.getSolutionDict()['v'][:, 1]
        base_pos = self.base_fk(q=q)['ee_pos'].toarray()
        base_rot = self.base_fk(q=q)['ee_rot'].toarray()

        qmdl = np.zeros(self.model.getJointNum())

        qmdl[6:] = q[7:]
        base_pose = Affine3(pos=base_pos)
        base_pose.linear = base_rot
        self.model.setJointPosition(qmdl)
        self.model.setFloatingBasePose(base_pose)
        self.model.update()

    def update_ci(self):
        self.ciros.run()
        Tref = self.task.getPoseReference()[0]
        print(Tref)
        ptgt.assign(Tref.translation.tolist()) #+ [0, 0, 0, 1]

def barrier(x):
    return cs.sum1(cs.if_else(x > 0, 0, x ** 2))


def _trj(tau):
    return 64. * tau ** 3 * (1 - tau) ** 3

def compute_polynomial_trajectory(k_start, nodes, nodes_duration, p_start, p_goal, clearance, dim=None):

    if dim is None:
        dim = [0, 1, 2]

    # todo check dimension of parameter before assigning it

    traj_array = np.zeros(len(nodes))

    start = p_start[dim]
    goal = p_goal[dim]

    index = 0
    for k in nodes:
        tau = (k - k_start) / nodes_duration
        trj = _trj(tau) * clearance
        trj += (1 - tau) * start + tau * goal
        traj_array[index] = trj
        index = index + 1

    return np.array(traj_array)

# set up problem
ns = 50
tf = 5.0  # 10s
dt = tf / ns

# set up solver
solver_type = 'ilqr'

transcription_method = 'multiple_shooting'
transcription_opts = dict(integrator='RK4')


# set up model
urdf_path = rospkg.RosPack().get_path('mirror_urdf') + '/urdf/mirror.urdf'
urdf = open(urdf_path, 'r').read()
srdf_path = rospkg.RosPack().get_path('mirror_srdf') + '/srdf/mirror.srdf'
srdf = open(srdf_path, 'r').read()
rospy.set_param('/robot_description', urdf)

# contact frames
contacts = [f'arm_{i+1}_TCP' for i in range(3)]

# initial config
q_init = {}

for i in range(3):
    q_init[f'arm_{i+1}_joint_2'] = -1.9
    q_init[f'arm_{i+1}_joint_3'] = -2.30
    q_init[f'arm_{i+1}_joint_5'] = -0.4

base_init = np.array([0, 0, 0.72, 0, 0, 0, 1])


# set up model description
urdf = urdf.replace('continuous', 'revolute')

kd = pycasadi_kin_dyn.CasadiKinDyn(urdf)
kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
q_init = {k: v for k, v in q_init.items()}

# set up model
prb = Problem(ns, receding=True)  # logging_level=logging.DEBUG
prb.setDt(dt)

model = FullModelInverseDynamics(problem=prb,
                                 kd=kd,
                                 q_init=q_init,
                                 base_init=base_init)

for contact in contacts:
    model.setContactFrame(contact, 'vertex', dict(vertex_frames=[contact]))

model.q.setBounds(model.q0, model.q0, 0)
model.v.setBounds(model.v0, model.v0, 0)

model.q.setInitialGuess(model.q0)

f0 = [0, 0, kd.mass()/3 * 9.8]
for f_name, f_var in model.fmap.items():
    f_var.setInitialGuess(f0)

ptgt = prb.createParameter('ptgt', 3)
goalx = prb.createFinalResidual("final_x", 3 * (model.q[0] - ptgt[0]))
goaly = prb.createFinalResidual("final_y", 3 * (model.q[1] - ptgt[1]))
goalrz = prb.createFinalResidual("final_rz", 3 * (model.q[5] - ptgt[2]))

# final velocity
model.v.setBounds(model.v0, model.v0, nodes=ns)

# base rotation
prb.createResidual("min_rot", 1e-4 * (model.q[3:5] - model.q0[3:5]))

# joint posture
prb.createResidual("min_q", 1e-1 * (model.q[7:] - model.q0[7:]))

# joint velocity
prb.createResidual("min_v", 1e-1 * model.v)

# final posture
prb.createFinalResidual("min_qf", 1e1 * (model.q[7:] - model.q0[7:]))

jlim_cost = barrier(model.q[8:10] - (-2.55)) + \
            barrier(model.q[14:16] - (-2.55)) + \
            barrier(model.q[20:22] - (-2.55))

prb.createCost(f'jlim', 100 * jlim_cost)

# joint acceleration
prb.createIntermediateResidual("min_q_ddot", 1e-2 * model.a)

# contact forces
for f_name, f_var in model.fmap.items():
    prb.createIntermediateResidual(f"min_{f_var.getName()}", 1e-3 * (f_var - f0))

contact_pos = dict()
# contact velocity is zero, and normal force is positive
for i, frame in enumerate(contacts):
    FK = kd.fk(frame)
    DFK = kd.frameVelocity(frame, kd_frame)
    DDFK = kd.frameAcceleration(frame, kd_frame)

    ee_p = FK(q=model.q)['ee_pos']
    ee_rot = FK(q=model.q)['ee_rot']
    ee_v = DFK(q=model.q, qdot=model.v)['ee_vel_linear']
    ee_v_ang = DFK(q=model.q, qdot=model.v)['ee_vel_angular']
    a = DDFK(q=model.q, qdot=model.v)['ee_acc_linear']

    # vertical contact frame (always active?)
    rot_err = cs.sumsqr(ee_rot[2, :2])
    prb.createIntermediateCost(f'{frame}_rot', 1e1*rot_err)  # nodes=[]

    # kinematic contact
    contact = prb.createConstraint(f"{frame}_vel", ee_v, nodes=[])  # ee_v # cs.vertcat(ee_v, ee_v_ang)
    contact_rot = prb.createResidual(f"{frame}_rot_vel", 1 * ee_v_ang, nodes=[])
    # unilateral forces
    fcost = barrier(model.fmap[frame][2] - 100.0)  # fz > 10
    unil = prb.createIntermediateCost(f'{frame}_unil', 1e-3 * fcost, nodes=[])

    # clearance
    contact_pos[frame] = FK(q=model.q0)['ee_pos']
    z_des = prb.createParameter(f'{frame}_z_des', 1)
    clea = prb.createConstraint(f"{frame}_clea", ee_p[2] - z_des, nodes=[])

    # vertical takeoff and touchdown
    lat_vel = cs.vertcat(ee_v[0:2], ee_v_ang)
    vert = prb.createConstraint(f"{frame}_vert", lat_vel, nodes=[])


# set base goal
# x_goal = 2.
# y_goal = 1.

# base_goal = [x_goal, y_goal, math.atan2(y_goal, x_goal)]

# ptgt.assign(base_goal)


cplusplus = True

opts =dict()
# opts['logging_level']=logging.DEBUG
if cplusplus:
    pm = pymanager.PhaseManager(ns)
else:
    pm = PhaseManager(nodes=ns, opts=opts)

c_phases = dict()
for c in contacts:
     c_phases[c] = pm.addTimeline(f'{c}_timeline')
#
i = 0
for c in contacts:
    # stance phase
    stance_duration = 10
    if cplusplus:
        stance_phase = pyphase.Phase(stance_duration, f"stance_{c}")
    else:
        stance_phase = Phase(f'stance_{c}', stance_duration)

    stance_phase.addConstraint(prb.getConstraints(f'{c}_vel'))
    stance_phase.addCost(prb.getCosts(f'{c}_rot_vel'))
    stance_phase.addCost(prb.getCosts(f'{c}_unil'))
    c_phases[c].registerPhase(stance_phase)
#
#     # flight phase
    flight_duration = 10
    if cplusplus:
        flight_phase = pyphase.Phase(flight_duration, f"flight_{c}")
    else:
        flight_phase = Phase(f'flight_{c}', flight_duration)

    flight_phase.addVariableBounds(prb.getVariables(f'f_{c}'),  np.array([[0, 0, 0]] * flight_duration).T, np.array([[0, 0, 0]] * flight_duration).T)
    flight_phase.addConstraint(prb.getConstraints(f'{c}_clea'))

    # vertical take-off
    flight_phase.addConstraint(prb.getConstraints(f'{c}_vert'), nodes=[0, flight_duration-1])  # nodes=[0, 1, 2]

    z_trj = np.atleast_2d(compute_polynomial_trajectory(0, range(flight_duration), flight_duration, contact_pos[c], contact_pos[c], 0.1, dim=2))

    flight_phase.addParameterValues(prb.getParameters(f'{c}_z_des'), z_trj)
    c_phases[c].registerPhase(flight_phase)


model.setDynamics()

if solver_type != 'ilqr':
    Transcriptor.make_method(transcription_method, prb, transcription_opts)

for c in contacts:
    stance = c_phases[c].getRegisteredPhase(f'stance_{c}')
    flight = c_phases[c].getRegisteredPhase(f'flight_{c}')
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)

# for c in contacts:
#     print(c_phases[c].getActivePhase())

################################################ CRAWLING ##############################################################
n_cycle = 20
initial_phase = 2

for i in range(n_cycle):
    for pos, step in enumerate(contacts):
        c_phases[step].addPhase(c_phases[step].getRegisteredPhase(f'flight_{step}'), initial_phase + pos)
    initial_phase += pos + 1



opts = {'ilqr.max_iter': 300,
        'ilqr.alpha_min': 0.01,
        'ilqr.use_filter': False,
        'ilqr.step_length': 1.0,
        'ilqr.enable_line_search': False,
        'ilqr.hxx_reg': 1.0,
        'ilqr.integrator': 'RK4',
        'ilqr.merit_der_threshold': 1e-6,
        'ilqr.step_length_threshold': 1e-9,
        'ilqr.line_search_accept_ratio': 1e-4,
        'ilqr.kkt_decomp_type': 'qr',
        'ilqr.constr_decomp_type': 'qr',
        'ilqr.codegen_enabled': True,
        'ilqr.codegen_workdir': '/tmp/miaolo',
        # 'ilqr.enable_gn': True,
        # 'ilqr.hxx_reg_base': 0.0,
        # 'ilqr.n_threads': 0
        }

# todo if receding is true ....
solver_bs = Solver.make_solver(solver_type, prb, opts)

try:
    solver_bs.set_iteration_callback()
except:
    pass

scoped_opts_rti = opts.copy()
scoped_opts_rti['ilqr.enable_line_search'] = False
scoped_opts_rti['ilqr.max_iter'] = 1
solver_rti = Solver.make_solver(solver_type, prb, scoped_opts_rti)

t = time.time()
solver_bs.solve()
elapsed = time.time() - t
print(f'bootstrap solved in {elapsed} s')
try:
    solver_rti.print_timings()
except:
    pass

solution = solver_bs.getSolutionDict()

# =========================================================================

from horizon.ros import replay_trajectory

repl = replay_trajectory.replay_trajectory(dt, kd.joint_names(), np.array([]), {k: None for k in contacts}, kd_frame, kd)
iteration = 0

rate = rospy.Rate(1 / dt)
flag_action = 1
forces = [prb.getVariables('f_' + c) for c in contacts]
nc = 4

elapsed_time_list = []
elapsed_time_solution_list = []

pb_dict = {
    'stack': [['ee']],

    'ee': {'type': 'Cartesian',
           'distal_link': 'core_link'}
}

pb = yaml.dump(pb_dict)

solver_rti.solve()
ci = CartesioSolver(urdf, srdf, pb, kd, solver_rti)

while iteration < 350:
    iteration = iteration + 1
    print(iteration)

    ci.update_ci()

    x_opt = solution['x_opt']
    u_opt = solution['u_opt']

    shift_num = -1
    xig = np.roll(x_opt, shift_num, axis=1)

    for i in range(abs(shift_num)):
        xig[:, -1 - i] = x_opt[:, -1]
    prb.getState().setInitialGuess(xig)

    uig = np.roll(u_opt, shift_num, axis=1)

    for i in range(abs(shift_num)):
        uig[:, -1 - i] = u_opt[:, -1]
    prb.getInput().setInitialGuess(uig)

    prb.setInitialState(x0=xig[:, 0])

    tic = time.time()
    pm._shift_phases()
    elapsed_time = time.time() - tic
    print('cycle:', elapsed_time)
    elapsed_time_list.append(elapsed_time)

    tic_solve = time.time()
    solver_rti.solve()
    elapsed_time_solving = time.time() - tic_solve
    print('solve:', elapsed_time_solving)
    elapsed_time_solution_list.append(elapsed_time_solving)
    solution = solver_rti.getSolutionDict()

    ci.set_model_from_solution()

    repl.frame_force_mapping = {cname: solution[f.getName()] for cname, f in model.fmap.items()}
    repl.publish_joints(solution['q'][:, 0])
    repl.publishContactForces(rospy.Time.now(), solution['q'][:, 0], 0)
    rate.sleep()


print("elapsed time resetting nodes:", sum(elapsed_time_list) / len(elapsed_time_list))
print("elapsed time solving:", sum(elapsed_time_solution_list) / len(elapsed_time_solution_list))
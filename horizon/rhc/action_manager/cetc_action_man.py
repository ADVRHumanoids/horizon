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
import time, rospkg, rospy
import phase_manager.pyphase as pyphase
import phase_manager.pymanager as pymanager


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
tf = 2.0  # 10s
dt = tf / ns

# set up solver
solver_type = 'ilqr'

transcription_method = 'multiple_shooting'
transcription_opts = dict(integrator='RK4')

# set up model
urdf_path = rospkg.RosPack().get_path('cetc_sat_urdf') + '/urdf/cetc_sat.urdf'
urdf = open(urdf_path, 'r').read()
rospy.set_param('/robot_description', urdf)

kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
kd = pycasadi_kin_dyn.CasadiKinDyn(urdf)
contacts = ['ball_1', 'ball_2', 'ball_3', 'ball_4']

base_init = np.array([0.0, 0.0, 0.505, 0.0, 0.0, 0.0, 1.0])

q_init = {'shoulder_yaw_1': 0.0,
          'shoulder_pitch_1': 0.93,
          'elbow_pitch_1': 1.68,

          'shoulder_yaw_2': 0.0,
          'shoulder_pitch_2': 0.93,
          'elbow_pitch_2': 1.68,

          'hip_roll_1': 0.0,
          'hip_pitch_1': 0.9,
          'knee_pitch_1': -1.52,

          'hip_roll_2': 0.0,
          'hip_pitch_2': 0.9,
          'knee_pitch_2': -1.52,

          'hip_roll_3': 0.0,
          'hip_pitch_3': 0.9,
          'knee_pitch_3': -1.52,

          'hip_roll_4': 0.0,
          'hip_pitch_4': 0.9,
          'knee_pitch_4': -1.52

          }

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
model.v.setBounds(np.zeros(model.nv), np.zeros(model.nv), 0)

model.q.setInitialGuess(model.q0)

for f_name, f_var in model.fmap.items():
    f_var.setInitialGuess([0, 0, kd.mass()/4 * 9.8])


if solver_type != 'ilqr':
    Transcriptor.make_method(transcription_method, prb, transcription_opts)

contact_pos = dict()
# contact velocity is zero, and normal force is positive
for i, frame in enumerate(contacts):
    FK = kd.fk(frame)
    DFK = kd.frameVelocity(frame, kd_frame)
    DDFK = kd.frameAcceleration(frame, kd_frame)

    p = FK(q=model.q)['ee_pos']
    v = DFK(q=model.q, qdot=model.v)['ee_vel_linear']
    a = DDFK(q=model.q, qdot=model.v)['ee_acc_linear']

    # kinematic contact
    contact = prb.createConstraint(f"{frame}_vel", v, nodes=[])
    # unilateral forces
    fcost = barrier(model.fmap[frame][2] - 10.0)  # fz > 10
    unil = prb.createIntermediateCost(f'{frame}_unil', 1e1 * fcost, nodes=[])

    # clearance
    contact_pos[frame] = FK(q=model.q0)['ee_pos']
    z_des = prb.createParameter(f'{frame}_z_des', 1)

    clea = prb.createConstraint(f"{frame}_clea", p[2] - z_des, nodes=[])

# prb.createFinalConstraint('final_v', model.v)
# final_base_x
# prb.createFinalResidual(f'min_q0', 3 * (model.q[0]))
prb.createFinalResidual(f'min_q0', 3 * (model.q[0] - 1.)) #-0.5
# final_base_y
# prb.createFinalResidual(f'min_q1', 3 * (model.q[1] - model.q0[1]))
# joint posture
prb.createResidual("min_q", 1. * (model.q[7:] - model.q0[7:]))
# joint acceleration
prb.createIntermediateResidual("min_q_ddot", 0.01 * model.a)
# contact forces
for f_name, f_var in model.fmap.items():
    prb.createIntermediateResidual(f"min_{f_var.getName()}", 0.01 * f_var)

cplusplus = True  # without set nodes: 6.5*10-5 (c++) vs 9*10-4 (python)

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
    stance_duration = 5
    if cplusplus:
        stance_phase = pyphase.Phase(stance_duration, f"stance_{c}")
    else:
        stance_phase = Phase(f'stance_{c}', stance_duration)
    stance_phase.addConstraint(prb.getConstraints(f'{c}_vel'))
    stance_phase.addCost(prb.getCosts(f'{c}_unil'))
    c_phases[c].registerPhase(stance_phase)
    #
    #     # flight phase
    flight_duration = 5
    if cplusplus:
        flight_phase = pyphase.Phase(flight_duration, f"flight_{c}")
    else:
        flight_phase = Phase(f'flight_{c}', flight_duration)
    flight_phase.addVariableBounds(prb.getVariables(f'f_{c}'),  np.array([[0, 0, 0]] * flight_duration).T, np.array([[0, 0, 0]] * flight_duration).T)
    flight_phase.addConstraint(prb.getConstraints(f'{c}_clea'))

    z_trj = np.atleast_2d(compute_polynomial_trajectory(0, range(flight_duration), flight_duration, contact_pos[c], contact_pos[c], 0.03, dim=2))
    flight_phase.addParameterValues(prb.getParameters(f'{c}_z_des'), z_trj)
    c_phases[c].registerPhase(flight_phase)

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


n_cycle = 10
initial_phase = 2

# todo
lift_contacts = ['ball_1', 'ball_4', 'ball_2', 'ball_3']

# for i in range(n_cycle):
#     for pos, step in enumerate(lift_contacts):
#         c_phases[step].addPhase(c_phases[step].getRegisteredPhase(f'flight_{step}'), initial_phase + pos)
#     initial_phase += pos + 1

range_trot_1 = range(4, 18, 2)
range_trot_2 = range(5, 19, 2)
#
lift_contact = 'ball_1'
for i in range_trot_1:
    c_phases[lift_contact].addPhase(c_phases[lift_contact].getRegisteredPhase(f'flight_{lift_contact}'), i)
lift_contact = 'ball_3'
for i in range_trot_2:
    c_phases[lift_contact].addPhase(c_phases[lift_contact].getRegisteredPhase(f'flight_{lift_contact}'), i)
lift_contact = 'ball_2'
for i in range_trot_2:
    c_phases[lift_contact].addPhase(c_phases[lift_contact].getRegisteredPhase(f'flight_{lift_contact}'), i)
lift_contact = 'ball_4'
for i in range_trot_1:
    c_phases[lift_contact].addPhase(c_phases[lift_contact].getRegisteredPhase(f'flight_{lift_contact}'), i)

model.setDynamics()

opts = {'ilqr.max_iter': 200,
        'ilqr.alpha_min': 0.01,
        'ilqr.step_length_threshold': 1e-9,
        'ilqr.line_search_accept_ratio': 1e-4,
        }

# todo if receding is true ....
solver_bs = Solver.make_solver('ilqr', prb, opts)

try:
    solver_bs.set_iteration_callback()
except:
    pass

scoped_opts_rti = opts.copy()
scoped_opts_rti['ilqr.enable_line_search'] = False
scoped_opts_rti['ilqr.max_iter'] = 1
solver_rti = Solver.make_solver('ilqr', prb, scoped_opts_rti)

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

import subprocess, rospy
from horizon.ros import replay_trajectory

# os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
# subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=spot'])
# rospy.loginfo("'spot' visualization started.")

repl = replay_trajectory.replay_trajectory(dt, kd.joint_names(), np.array([]), {k: None for k in contacts}, kd_frame,
                                           kd)
iteration = 0

rate = rospy.Rate(1 / dt)
flag_action = 1
forces = [prb.getVariables('f_' + c) for c in contacts]
nc = 4

elapsed_time_list = []
elapsed_time_solution_list = []

while iteration < 200:
    iteration = iteration + 1
    print(iteration)

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

    repl.frame_force_mapping = {cname: solution[f.getName()] for cname, f in model.fmap.items()}
    repl.publish_joints(solution['q'][:, 0])
    repl.publishContactForces(rospy.Time.now(), solution['q'][:, 0], 0)
    rate.sleep()


print("elapsed time resetting nodes:", sum(elapsed_time_list) / len(elapsed_time_list))
print("elapsed time solving:", sum(elapsed_time_solution_list) / len(elapsed_time_solution_list))




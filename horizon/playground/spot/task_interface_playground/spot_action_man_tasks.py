import os
import time

import numpy as np
from horizon.rhc.taskInterface import TaskInterface
from horizon.utils.actionManager import ActionManager, Step
from horizon.problem import Problem
from horizon.rhc.model_description import FullModelInverseDynamics
from casadi_kin_dyn import pycasadi_kin_dyn
from matlogger2 import matlogger
import phase_manager.pyphase as pyphase
import phase_manager.pymanager as pymanager
from horizon.utils import trajectoryGenerator as tg

# set up problem
ns = 50
tf = 2.0  # 10s
dt = tf / ns

# set up solver
solver_type = 'ilqr'

# set up model
path_to_examples = os.path.dirname('../../../examples/')
urdffile = os.path.join(path_to_examples, 'urdf', 'spot.urdf')
urdf = open(urdffile, 'r').read()
kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
kd = pycasadi_kin_dyn.CasadiKinDyn(urdf)
contacts = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']

base_init = np.array([0.0, 0.0, 0.505, 0.0, 0.0, 0.0, 1.0])
q_init = {'lf_haa_joint': 0.0,
          'lf_hfe_joint': 0.9,
          'lf_kfe_joint': -1.52,

          'lh_haa_joint': 0.0,
          'lh_hfe_joint': 0.9,
          'lh_kfe_joint': -1.52,

          'rf_haa_joint': 0.0,
          'rf_hfe_joint': 0.9,
          'rf_kfe_joint': -1.52,

          'rh_haa_joint': 0.0,
          'rh_hfe_joint': 0.9,
          'rh_kfe_joint': -1.52}


# set up model
prb = Problem(ns, receding=True)  # logging_level=logging.DEBUG
prb.setDt(dt)

model = FullModelInverseDynamics(problem=prb,
                                 kd=kd,
                                 q_init=q_init,
                                 base_init=base_init)


ti = TaskInterface(prb, model)

# ti.loadPlugins(['horizon.rhc.plugins.contactTaskSpot'])

ti.setTaskFromYaml('config_walk.yaml')

final_base_x = ti.getTask('final_base_x')
final_base_x.setRef([model.q0[0], 0, 0, 0, 0, 0, 1])

final_base_y = ti.getTask('final_base_y')
final_base_y.setRef([0, model.q0[1], 0, 0, 0, 0, 1])

# min_rot = ti.getTask('min_rot')
# min_rot.setRef(ti.q0[3:5])

# final_x = ti.getTask('final_x')
# final_x.setRef([model.q0[0], 0, 0, 0, 0, 0, 1])
#
# final_y = ti.getTask('final_y')
# final_y.setRef([0, model.q0[1], 0, 0, 0, 0, 1])

f0 = np.array([0, 0, kd.mass()/4 * 9.8])
reg = ti.getTask('regularization')
reg.setRef(1, [0, 0, 0]) #f0
reg.setRef(2, [0, 0, 0]) #f0
reg.setRef(3, [0, 0, 0]) #f0
reg.setRef(4, [0, 0, 0]) #f0

opts = dict()

# c_0 = ti.getTask('contact_lf_foot')
# c_1 = ti.getTask('contact_rf_foot')
# c_2 = ti.getTask('contact_lh_foot')
# c_3 = ti.getTask('contact_rh_foot')

# step = range(10, 30)
# contact_c_0 = [c_n for c_n in list(range(ns + 1)) if c_n not in step]
# c_0.setNodes(contact_c_0)
# c_1.setNodes(range(ns + 1))
# c_2.setNodes(range(ns + 1))
# c_3.setNodes(range(ns + 1))

# ===============================================================
# ===============================================================
# ===============================================================
q = ti.prb.getVariables('q')
v = ti.prb.getVariables('v')

q.setBounds(model.q0, model.q0, nodes=0)
v.setBounds(model.v0, model.v0, nodes=0)

q.setInitialGuess(model.q0)

forces = [prb.getVariables('f_' + c) for c in contacts]

for f in forces:
    f.setInitialGuess(f0)


# manual
# for c in contacts:
#     c_task = ti.getTask('foot_contact_' + c)
#     c_task.setNodes(range(ns + 1))

stance_duration = 5
flight_duration = 5

pm = pymanager.PhaseManager(ns)

tg = tg.TrajectoryGenerator()
c_phases = dict()
for c in contacts:
     c_phases[c] = pm.addTimeline(f'{c}_timeline')


for c in contacts:
    # stance phase
    stance_phase = pyphase.Phase(stance_duration, f"stance_{c}")
    contact_task = ti.getTask('foot_contact_' + c)
#
    stance_phase.addItem(contact_task)
    c_phases[c].registerPhase(stance_phase)

    # flight phase
    flight_phase = pyphase.Phase(flight_duration, f"flight_{c}")
    clearance_task = ti.getTask('foot_z_' + c)
    stance_phase.addItem(clearance_task)

    FK = kd.fk(c)
    initial_contact_pos = FK(q=model.q0)['ee_pos'].elements()[0]

    z_trj = tg.from_derivatives(flight_duration, initial_contact_pos, initial_contact_pos, 0.02, [0, 0, 0])

    cart_ref = np.zeros([7, stance_duration])
    cart_ref[2, :] = z_trj

    print(cart_ref)
    clearance_task.setRef(cart_ref)

    # clearance_task.
    c_phases[c].registerPhase(flight_phase)



    clearance_task.setRef()


# clearance_task.setRef()

for c in contacts:
    stance = c_phases[c].getRegisteredPhase(f'stance_{c}')
    flight = c_phases[c].getRegisteredPhase(f'flight_{c}')
#
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(flight)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)
    c_phases[c].addPhase(stance)



# final_base_x.setRef([0.5, 0, 0, 0, 0, 0, 1])
# ptgt.assign(ptgt_final, nodes=ns)

#
# for c_name, c_item in prb.getConstraints().items():
#     print(c_item.getName())
#     print(c_item.getNodes())
# print("=========================")
# for c_name, c_item in prb.getCosts().items():
#     print(c_item.getName())
#     print(c_item.getNodes())
# for c_name, c_item in prb.getVariables().items():
#     print(c_item.getName())
#     print(c_item.getNodes())
#     print(c_item.getLowerBounds())
#
# exit()
ti.finalize()
ti.bootstrap()
solution = ti.solution

import subprocess, rospy
from horizon.ros import replay_trajectory

rospy.set_param('/robot_description', urdf)
bashCommand = 'rosrun robot_state_publisher robot_state_publisher'
subprocess.Popen(bashCommand.split(), start_new_session=True)

# os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
# subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=spot'])
# rospy.loginfo("'spot' visualization started.")

ml = matlogger.MatLogger2('ciao')

for name, element in solution.items():
    ml.create(name, element.shape[0])

for contact in contacts:
    FK = kd.fk(contact)
    pos = FK(q=solution['q'])['ee_pos']
    ml.create(contact + "pos", pos.shape[0])


for i, contact in enumerate(contacts):
    ml.create(contact, i)

# os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
# subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=spot'])
# rospy.loginfo("'spot' visualization started.")

# single replay
# q_sol = solution['q']
# frame_force_mapping = {contacts[i]: solution[forces[i].getName()] for i in range(nc)}
# repl = replay_trajectory.replay_trajectory(dt, kd.joint_names()[2:], q_sol, frame_force_mapping, kd_frame, kd)
# repl.sleep(1.)
# repl.replay(is_floating_base=True)
# exit()
# =========================================================================
repl = replay_trajectory.replay_trajectory(dt, kd.joint_names(), np.array([]), {k: None for k in contacts}, kd_frame,
                                           kd, trajectory_markers=contacts)
iteration = 0

# solver_rti.solution_dict['x_opt'] = solver_bs.getSolutionState()
# solver_rti.solution_dict['u_opt'] = solver_bs.getSolutionInput()
rate = rospy.Rate(1 / dt)
flag_action = 1
forces = [ti.prb.getVariables('f_' + c) for c in contacts]
nc = 4

elapsed_time_list = []
elapsed_time_solution_list = []
elapsed_time_iter_list = []

while iteration < 100:
    tic_iter = time.time()
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

    # tic = time.time()
    # elapsed_time = time.time() - tic
    # print('cycle:', elapsed_time)
    # elapsed_time_list.append(elapsed_time)
    pm._shift_phases()

    tic_solve = time.time()
    ti.solver_rti.solve()
    elapsed_time_solving = time.time() - tic_solve
    print('solve:', elapsed_time_solving)
    elapsed_time_solution_list.append(elapsed_time_solving)
    solution = ti.solver_rti.getSolutionDict()

    for name, element in solution.items():
        ml.add(name, element[:, 0])

    for contact in contacts:
        FK = kd.fk(contact)
        contact_pos = FK(q=solution['q'][:, 0])['ee_pos']
        ml.add(contact + "pos", contact_pos)


    repl.frame_force_mapping = {cname: solution[f.getName()] for cname, f in model.fmap.items()}
    repl.publish_joints(solution['q'][:, 0])
    repl.publishContactForces(rospy.Time.now(), solution['q'][:, 0], 0)
    rate.sleep()

    elapsed_time_iter = time.time() - tic_iter
    print('full_iter:', elapsed_time_iter)
    elapsed_time_iter_list.append(elapsed_time_iter)

# print("elapsed time resetting nodes:", sum(elapsed_time_list) / len(elapsed_time_list))
print("elapsed time solving:", sum(elapsed_time_solution_list) / len(elapsed_time_solution_list))
print("elapsed time iterating:", sum(elapsed_time_iter_list) / len(elapsed_time_iter_list))


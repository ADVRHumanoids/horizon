import os
import time

import numpy as np
from horizon.rhc.taskInterface import TaskInterface
from horizon.utils.actionManager import ActionManager, Step
from horizon.problem import Problem
from horizon.rhc.model_description import FullModelInverseDynamics
from casadi_kin_dyn import pycasadi_kin_dyn

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

f0 = np.array([0, 0, 55])
reg = ti.getTask('regularization')
reg.setRef(1, f0)
reg.setRef(2, f0)
reg.setRef(3, f0)
reg.setRef(4, f0)

opts = dict()

am = ActionManager(ti, opts)

# step_frame = 'lf_foot'
# FK = kd.fk(step_frame)
# p0 = FK(q=model.q0)['ee_pos']
# one_step = Step(step_frame, 10, 20, p0, p0, 0.1)
# am.setStep(one_step)
am._trot([10, 200])
# am._walk([10, 200], [0, 2, 1, 3])


# ===============================================================
# ===============================================================
# ===============================================================
q = ti.prb.getVariables('q')
v = ti.prb.getVariables('v')

q.setBounds(model.q0, model.q0, nodes=0)
v.setBounds(model.v0, model.v0, nodes=0)

q.setInitialGuess(model.q0)

forces = [prb.getVariables('f_' + c) for c in contacts]
f0 = np.array([0, 0, 55])
for f in forces:
    f.setInitialGuess(f0)


# ================================================================
# ================================================================
# ===================== stuff to wrap ============================
# ================================================================
# ================================================================
import subprocess, rospy
from horizon.ros import replay_trajectory

# todo

# final_base_x.setRef([0.5, 0, 0, 0, 0, 0, 1])
# ptgt.assign(ptgt_final, nodes=ns)

ti.finalize()
ti.bootstrap()
solution = ti.solution


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
repl = replay_trajectory.replay_trajectory(dt, kd.joint_names(), np.array([]), {k: None for k in contacts}, kd_frame, kd)
iteration = 0

# solver_rti.solution_dict['x_opt'] = solver_bs.getSolutionState()
# solver_rti.solution_dict['u_opt'] = solver_bs.getSolutionInput()
rate = rospy.Rate(1 / dt)
flag_action = 1
forces = [ti.prb.getVariables('f_' + c) for c in contacts]
nc = 4

while True:
    iteration = iteration + 1
    print(iteration)

    am.execute(solution)
    tic = time.time()
    ti.solver_rti.solve()
    elapsed_time_solving = time.time() - tic
    print('solve:', elapsed_time_solving)
    solution = ti.solver_rti.getSolutionDict()
    repl.frame_force_mapping = {cname: solution[f.getName()] for cname, f in ti.model.fmap.items()}
    repl.publish_joints(solution['q'][:, 0])
    repl.publishContactForces(rospy.Time.now(), solution['q'][:, 0], 0)
    rate.sleep()


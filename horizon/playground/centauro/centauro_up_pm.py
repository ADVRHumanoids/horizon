import copy
import logging
import os
import numpy as np
from horizon.rhc.taskInterface import TaskInterface
from horizon.problem import Problem
from horizon.rhc.model_description import *
from phase_manager import pymanager, pyphase
from horizon.rhc.tasks.interactionTask import InteractionTask
from horizon.utils.actionManager import ActionManager, Step
from casadi_kin_dyn import py3casadi_kin_dyn
import casadi as cs
from horizon.utils import trajectoryGenerator
import rospy
from horizon.utils.analyzer import ProblemAnalyzer
rospy.init_node('centauro_up_node')
# set up model
path_to_examples = os.path.abspath(os.path.dirname(__file__) + "/../../examples")
os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples

# urdffile = os.path.join(path_to_examples, 'urdf', 'centauro.urdf')
urdffile = os.path.join(path_to_examples, 'urdf', 'centauro_big_wheels.urdf')
urdf = open(urdffile, 'r').read()
rospy.set_param('/robot_description', urdf)

fixed_joint_map = {'torso_yaw': 0.00,   # 0.00,

                    'j_arm1_1': 1.50,   # 1.60,
                    'j_arm1_2': 0.1,    # 0.,
                    'j_arm1_3': 0.2,   # 1.5,
                    'j_arm1_4': -2.2,  # 0.3,
                    'j_arm1_5': 0.00,   # 0.00,
                    'j_arm1_6': -1.3,   # 0.,
                    # 'j_arm1_7': 0.0,    # 0.0,

                    'j_arm2_1': 1.50,   # 1.60,
                    'j_arm2_2': 0.1,    # 0.,
                    'j_arm2_3': -0.2,   # 1.5,
                    'j_arm2_4': -2.2,   #-0.3,
                    'j_arm2_5': 0.0,    # 0.0,
                    'j_arm2_6': -1.3,   # 0.,
                    # 'j_arm2_7': 0.0,    # 0.0,
                    'd435_head_joint': 0.0,
                    'velodyne_joint': 0.0,

                    # 'hip_yaw_1': -0.746,
                    # 'hip_pitch_1': -1.254,
                    # 'knee_pitch_1': -1.555,
                    # 'ankle_pitch_1': -0.3,
                    #
                    # 'hip_yaw_2': 0.746,
                    # 'hip_pitch_2': 1.254,
                    # 'knee_pitch_2': 1.555,
                    # 'ankle_pitch_2': 0.3,
                    #
                    # 'hip_yaw_3': 0.746,
                    # 'hip_pitch_3': 1.254,
                    # 'knee_pitch_3': 1.555,
                    # 'ankle_pitch_3': 0.3,
                    #
                    # 'hip_yaw_4': -0.746,
                    # 'hip_pitch_4': -1.254,
                    # 'knee_pitch_4': -1.555,
                    # 'ankle_pitch_4': -0.3,
                    }

# initial config
q_init = {
    'hip_yaw_1': -0.746,
    'hip_pitch_1': -1.254,
    'knee_pitch_1': -1.555,
    'ankle_pitch_1': -0.3,

    'hip_yaw_2': 0.746,
    'hip_pitch_2': 1.254,
    'knee_pitch_2': 1.555,
    'ankle_pitch_2': 0.3,

    'hip_yaw_3': 0.746,
    'hip_pitch_3': 1.254,
    'knee_pitch_3': 1.555,
    'ankle_pitch_3': 0.3,

    'hip_yaw_4': -0.746,
    'hip_pitch_4': -1.254,
    'knee_pitch_4': -1.555,
    'ankle_pitch_4': -0.3,
}

wheels = [f'j_wheel_{i + 1}' for i in range(4)]
q_init.update(zip(wheels, 4 * [0.]))

ankle_yaws = [f'ankle_yaw_{i + 1}' for i in range(4)]
# q_init.update(zip(ankle_yaws, 4 * [0.]))
q_init.update(dict(ankle_yaw_1=np.pi/4))
q_init.update(dict(ankle_yaw_2=-np.pi/4))
q_init.update(dict(ankle_yaw_3=-np.pi/4))
q_init.update(dict(ankle_yaw_4=np.pi/4))

q_init.update(fixed_joint_map)

base_init = np.array([0, 0, 0.718565, 0, 0, 0, 1])

# set up model description
urdf = urdf.replace('continuous', 'revolute')

kd = pycasadi_kin_dyn.CasadiKinDyn(urdf, fixed_joints=fixed_joint_map)
kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
q_init = {k: v for k, v in q_init.items() if k not in fixed_joint_map.keys()}

# set up problem
N = 50
tf = 5.0
dt = tf / N

prb = Problem(N, receding=True)  # logging_level=logging.DEBUG
prb.setDt(dt)

# set up model
model = FullModelInverseDynamics(problem=prb,
                                 kd=kd,
                                 q_init=q_init,
                                 base_init=base_init,
                                 fixed_joint_map=fixed_joint_map)
# else:
#     model = SingleRigidBodyDynamicsModel(problem=prb,
#                                          kd=kd,
#                                          q_init=q_init,
#                                          base_init=base_init,
#                                          contact_dict=contact_dict)


ti = TaskInterface(prb=prb,
                   model=model)

ti.setTaskFromYaml(os.path.dirname(__file__) + '/centauro_config_pm.yaml')


f0 = np.array([0, 0, kd.mass()/4])
f0_flight = np.array([0, 0, 0])
init_force = ti.getTask('joint_regularization')
# init_force.setRef(1, f0)
# init_force.setRef(2, f0)
# init_force.setRef(3, f0)
# init_force.setRef(4, f0)
#
# init_force.setRef(1, f0_flight, nodes=range(20, 31))
# init_force.setRef(2, f0_flight, nodes=range(20, 31))


final_base_x = ti.getTask('final_base_xy')
final_base_x.setRef(np.array([[2., 0, 0, 0, 0, 0, 1]]).T)


# base_posture = ti.getTask('base_posture')
# base_posture.setRef(np.array([[0, 0, 0.718565, 0, 0, 0, 1]]).T)
tg = trajectoryGenerator.TrajectoryGenerator()

pm = pymanager.PhaseManager(N)

c_phases = dict()
for c_name, forces in model.cmap.items():
    print(f'adding phase for ee: {c_name}')
    c_phases[c_name] = pm.addTimeline(f'{c_name}_timeline')

for c_name, forces in model.cmap.items():

    # stance phase
    stance_duration = 5
    stance_phase = pyphase.Phase(stance_duration, f'stance_{c_name}')
    stance_phase.addItem(ti.getTask(f'foot_{c_name}'))
    c_phases[c_name].registerPhase(stance_phase)

    # flight phase
    flight_duration = 3
    flight_phase = pyphase.Phase(flight_duration, f'flight_{c_name}')
    init_z_foot = model.kd.fk(c_name)(q=model.q0)['ee_pos'].elements()[2]
    final_z_foot = init_z_foot + 0.03
    ref_trj = np.zeros(shape=[7, flight_duration])
    ref_trj[2, :] = np.atleast_2d(
        tg.from_derivatives(flight_duration, init_z_foot, final_z_foot, 0.05, [None, 0, None]))
    flight_phase.addItemReference(ti.getTask(f'foot_z_{c_name}'), ref_trj)
    c_phases[c_name].registerPhase(flight_phase)

for c_name, forces in model.cmap.items():
    stance = c_phases[c_name].getRegisteredPhase(f'stance_{c_name}')
    c_phases[c_name].addPhase(stance)
    c_phases[c_name].addPhase(stance)
    c_phases[c_name].addPhase(stance)
    c_phases[c_name].addPhase(stance)
    c_phases[c_name].addPhase(stance)
    c_phases[c_name].addPhase(stance)
    c_phases[c_name].addPhase(stance)
    c_phases[c_name].addPhase(stance)
    c_phases[c_name].addPhase(stance)
    c_phases[c_name].addPhase(stance)

flight_contact = ['contact_1'] # 'contact_2'

initial_phase_action = 6
for flight_c_name in flight_contact:
    flight = c_phases[flight_c_name].getRegisteredPhase(f'flight_{flight_c_name}')
    c_phases[flight_c_name].addPhase(flight, initial_phase_action)
    # initial_phase_action += 1

# ===============================================================
# ===============================================================
# ===============================================================
q = ti.prb.getVariables('q')
v = ti.prb.getVariables('v')
a = ti.prb.getVariables('a')

# ti.prb.createFinalResidual("min_qf", 1e2 * (q[7:] - ti.model.q0[7:]))
# ti.prb.createFinalCost("min_rot", 1e-1 * (q[3:4] - ti.model.q0[3:4]))
# ti.prb.createFinalCost("min_rot", 1e1 * (q[3:5] - ti.model.q0[3:5]))
# adding minimization of angular momentum
# cd_fun = ti.model.kd.computeCentroidalDynamics()
# h_lin, h_ang, dh_lin, dh_ang = cd_fun(q, v, a)
# ti.prb.createIntermediateResidual('min_angular_mom', 0.1 * dh_ang)

# continuity of forces
# continuity_force_w = 5e-2
# for f_name, f_var in model.fmap.items():
#     f_var_prev = f_var.getVarOffset(-1)
#     prb.createIntermediateResidual(f'continuity_force_{f_name}', continuity_force_w * (f_var - f_var_prev), range(1, prb.getNNodes() - 2))

q.setBounds(ti.model.q0, ti.model.q0, nodes=0)
v.setBounds(ti.model.v0, ti.model.v0, nodes=0)

q.setInitialGuess(ti.model.q0)

for cname, cforces in ti.model.cmap.items():
    for c in cforces:
        c.setInitialGuess(f0[:c.size1()])

        if cname == 'contact_1':
            c.setInitialGuess(f0_flight, nodes=range(20, 23))

        # if cname == 'contact_2':
        #     c.setInitialGuess(f0_flight, nodes=range(20, 23))



replay_motion = True
plot_sol = not replay_motion

# todo how to add?
ti.prb.createFinalConstraint('final_v', v)
# ================================================================
# ================================================================
# ===================== stuff to wrap ============================
# ================================================================
# ================================================================
from horizon.ros import replay_trajectory

ti.finalize()

pa = ProblemAnalyzer(prb)
pa.print()
# exit()

ti.bootstrap()
solution = ti.solution
ti.resample(0.001)
# ti.save_solution('centauro_up_pm.mat')
ti.replay_trajectory()












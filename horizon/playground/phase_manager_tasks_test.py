from cartesian_interface.pyci_all import *
from xbot_interface import config_options as co
from xbot_interface import xbot_interface as xbot
from horizon.problem import Problem
from horizon.rhc.model_description import FullModelInverseDynamics
from horizon.rhc.taskInterface import TaskInterface
from horizon.utils import trajectoryGenerator
from horizon.ros import replay_trajectory
import casadi_kin_dyn.py3casadi_kin_dyn as casadi_kin_dyn
import phase_manager.pymanager as pymanager
import phase_manager.pyphase as pyphase

from sensor_msgs.msg import Joy
from geometry_msgs.msg import Pose, Twist, Vector3
from walk_me_maybe.msg import SRBDTrajectory

import casadi as cs
import rospy
import rospkg
import numpy as np
import subprocess
import os


'''
Load urdf and srdf
'''
cogimon_urdf_folder = rospkg.RosPack().get_path('cogimon_urdf')
urdf = open(cogimon_urdf_folder + '/urdf/cogimon.urdf', 'r').read()
file_dir = os.getcwd()
ns = 20
T = 1.
dt = T / ns

prb = Problem(ns, receding=True, casadi_type=cs.SX)
prb.setDt(dt)

base_init = np.atleast_2d(np.array([0.03, 0., 0.962, 0., -0.029995, 0.0, 0.99955]))
# base_init = np.array([0., 0., 0.96, 0., 0.0, 0.0, 1.])

q_init = {"LHipLat":       -0.0,
          "LHipSag":       -0.363826,
          "LHipYaw":       0.0,
          "LKneePitch":    0.731245,
          "LAnklePitch":   -0.307420,
          "LAnkleRoll":    0.0,
          "RHipLat":       0.0,
          "RHipSag":       -0.363826,
          "RHipYaw":       0.0,
          "RKneePitch":    0.731245,
          "RAnklePitch":   -0.307420,
          "RAnkleRoll":    -0.0,
          "WaistLat":      0.0,
          "WaistYaw":      0.0,
          "LShSag":        1.1717860,
          "LShLat":        -0.059091562,
          "LShYaw":        -5.18150657e-02,
          "LElbj":         -1.85118,
          "LForearmPlate": 0.0,
          "LWrj1":         -0.523599,
          "LWrj2":         -0.0,
          "RShSag":        1.17128697,
          "RShLat":        6.01664139e-02,
          "RShYaw":        0.052782481,
          "RElbj":         -1.8513760,
          "RForearmPlate": 0.0,
          "RWrj1":         -0.523599,
          "RWrj2":         -0.0}

# contact_dict = {
#     'l_sole': {'type': 'surface'},
#     'r_sole': {'type': 'surface'}
# }

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

kin_dyn = casadi_kin_dyn.CasadiKinDyn(urdf)

model = FullModelInverseDynamics(problem=prb,
                                     kd=kin_dyn,
                                     q_init=q_init,
                                     base_init=base_init,
                                     contact_dict=contact_dict)


rospy.set_param('/robot_description', urdf)
bashCommand = 'rosrun robot_state_publisher robot_state_publisher'
process = subprocess.Popen(bashCommand.split(), start_new_session=True)

ti = TaskInterface(prb=prb, model=model)
ti.setTaskFromYaml(file_dir + '/config_phase_manager_test.yaml')



cd_fun = ti.model.kd.computeCentroidalDynamics()

# ======================================
# for name, force in model.fmap.items():
#     f_prev = force.getVarOffset(-1)
#     prb.createResidual(name + '_dot_reg', 1e-2 * (force - f_prev), nodes=15)
#
# exit()
# adding minimization of angular momentum
# h_lin, h_ang, dh_lin, dh_ang = cd_fun(model.q, model.v, model.a)
# ti.prb.createIntermediateResidual('min_angular_mom', 1e-1 * dh_ang)


tg = trajectoryGenerator.TrajectoryGenerator()

pm = pymanager.PhaseManager(ns)
# phase manager handling
c_phases = dict()
for c in contact_dict:
     c_phases[c] = pm.addTimeline(f'{c}_timeline')

for c in contact_dict:
    # stance phase
    stance_duration = 10
    stance_phase = pyphase.Phase(stance_duration, f'stance_{c}')
    stance_phase.addItem(ti.getTask(f'foot_contact_{c}'))
    c_phases[c].registerPhase(stance_phase)

    # short_stance_duration = 5
    # short_stance_phase = pyphase.Phase(short_stance_duration, f'short_stance_{c}')
    # short_stance_phase.addItem(ti.getTask(f'foot_contact_{c}'))
    # c_phases[c].registerPhase(short_stance_phase)

    flight_duration = 10
    flight_phase = pyphase.Phase(flight_duration, f'flight_{c}')

    init_z_foot = model.kd.fk(c)(q=model.q0)['ee_pos'].elements()[2]

    ref_trj = np.zeros(shape=[7, flight_duration])
    ref_trj[2, :] = np.atleast_2d(tg.from_derivatives(flight_duration, init_z_foot, init_z_foot, 0.05, [None, 0, None]))

    flight_phase.addItemReference(ti.getTask(f'foot_z_{c}'), ref_trj)
    # flight_phase.addItem(ti.getTask(f'foot_contact_{c}'), nodes=[flight_duration-1]) #, nodes=[flight_duration-1]

    # v_contact = model.kd.frameVelocity(c, model.kd_frame)(q=model.q, qdot=model.v)['ee_vel_linear']
    # p_contact = model.kd.fk(c)(q=model.q)['ee_pos'].elements()
    # last_swing_vel = prb.createConstraint(f'{c}last_swing_vel', v_contact, [])
    # last_swing_zero = prb.createConstraint(f'{c}_last_swing_zero', p_contact[2] - init_z_foot, [])
    # flight_phase.addConstraint(last_swing_vel, nodes=[flight_duration-1])
    # for contact in contact_dict[c]['vertex_frames']:
    #     flight_phase.addVariableBounds(prb.getVariables(f'f_{contact}'), np.array([[0, 0, 0]]).T, np.array([[np.inf, np.inf, np.inf]]).T, nodes=[flight_duration-1])

    c_phases[c].registerPhase(flight_phase)

for c in contact_dict:
    stance = c_phases[c].getRegisteredPhase(f'stance_{c}')
    flight = c_phases[c].getRegisteredPhase(f'flight_{c}')
    # short_stance = c_phases[c].getRegisteredPhase(f'short_stance_{c}')
    c_phases[c].addPhase(stance)
    # c_phases[c].addPhase(stance)


    # print(f'active phases in timeline {c}:')
    # for active_phase in c_phases[c].getActivePhases():
    #     print(f'{active_phase.getName()}: {active_phase.getActiveNodes()}')
    #
    # print(c_phases[c].getEmptyNodes())

ti.model.q.setBounds(ti.model.q0, ti.model.q0, nodes=0)
ti.model.v.setBounds(ti.model.v0, ti.model.v0, nodes=0)
ti.model.q.setInitialGuess(ti.model.q0)

for name, constraint in prb.getConstraints().items():
    if name == 'zero_velocity_l_foot_l_sole_vel_cartesian_task':
        print('foot:', name)
        print(constraint.getNodes())

for name, variable in prb.getVariables().items():
    if name == 'f_r_foot_upper_right_link':
    # if name in ['f_' + elem for elem in contact_dict['l_sole']['vertex_frames']]:
        print(f'{name} lb:\n', variable.getLowerBounds())

f0 = [0, 0, kin_dyn.mass()/8 * 9.8] #, 0, 0, 0]
for cname, cforces in ti.model.cmap.items():
    for c in cforces:
        c.setInitialGuess(f0)

# finalize taskInterface and solve bootstrap problem
exit()
ti.finalize()
ti.bootstrap()
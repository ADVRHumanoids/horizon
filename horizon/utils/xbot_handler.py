# /usr/bin/env python3
import numpy as np
##### for robot and stuff
import xbot_interface.config_options as xbot_opt
from xbot_interface import xbot_interface as xbot
import rospy
from casadi_kin_dyn import py3casadi_kin_dyn
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter, mat_storer
from horizon.ros.replay_trajectory import *
import os

# run ~/iit-centauro-ros-pkg/centauro_gazebo/launch/centauro_world.launch
# run xbot2-core
# run this script
## ==================
## PREPARE TRAJECTORY
## ==================

class XBotHandler:
    def __init__(self):

        self.init_xbot()
        self.robot.sense()



        ## PREPARE ROBOT

    def init_xbot(self, is_floating=True):

        rospy.init_node('xbot_replay')

        opt = xbot_opt.ConfigOptions()

        urdf = rospy.get_param('/xbotcore/robot_description')
        srdf = rospy.get_param('/xbotcore/robot_description_semantic')

        opt.set_urdf(urdf)
        opt.set_srdf(srdf)
        opt.generate_jidmap()
        opt.set_bool_parameter('is_model_floating_base', is_floating)
        opt.set_string_parameter('model_type', 'RBDL')
        opt.set_string_parameter('framework', 'ROS')
        self.model = xbot.ModelInterface(opt)
        self.robot = xbot.RobotInterface(opt)



    def homing(self, q_homing, duration=5., dt = 0.001):

        self.robot.sense()
        init_pose = self.robot.getMotorPosition()

        # print(self.robot.getStiffness())
        # damp = self.robot.getDamping()
        # new_damp = [2 * elem for elem in damp]

        # for i in range(1000):
        #     self.robot.setDamping(new_damp)  # [200, 200, 100]

        # print(self.robot.getDamping())

        final_pose = q_homing
        t = 0
        tau = 0

        rate = rospy.Rate(1. / dt)

        while tau < 1.:
            tau = t / duration
            alpha = ((6 * tau - 15) * tau + 10) * tau * tau * tau
            q_ref = init_pose + alpha * (final_pose - init_pose)
            self.robot.setPositionReference(q_ref)
            self.robot.move()
            rate.sleep()
            t += dt



    def stupid_homing(self, q_homing, duration=5):

        for i in range(100):
            self.robot.setPositionReference(q_homing)

            self.robot.move()


    def replay(self, solution, var_names=None, dt_name=None, homing_duration=5.):

        if var_names is None:
            var_names = ['q']

        if dt_name is None:
            dt_name = ['dt']

        var_q = var_names[0]
        dt = solution[dt_name]

        if not isinstance(solution[dt_name], float):
            dt = solution[dt_name].item(0)

        q = solution[var_q]
        num_samples = solution[var_q].shape[1]

        q_robot = q[7:, :]
        # q_dot_robot = qdot_res[6:, :]
        # tau_robot = tau_res[6:, :]

        # set stiffness and damping
        # for i in range(1):
            # self.robot.setStiffness(3 *[200, 200, 100]) #[200, 200, 100]
            # self.robot.setDamping(3 * [50, 50, 30])  # [200, 200, 100]
            # self.robot.move()

        # exit()
        q_homing = q_robot[:, 0]
        self.robot.sense()
        self.homing(q_homing, duration=homing_duration, dt=dt)

        input('click to replay')
        rate = rospy.Rate(1. / dt)

        for i in range(num_samples):
            self.robot.setPositionReference(q_robot[:, i])
            # robot.setVelocityReference(q_dot_robot[:, i])
            # robot.setEffortReference(tau_robot[:, i])
            self.robot.move()
            rate.sleep()


    def publish(self, q_robot):

        # ===================== XBOT =====================
        # q_dot_robot = qdot_res[6:, :]
        # tau_robot = tau_res[6:, :]

        self.robot.sense()
        # =================================================

        self.robot.setPositionReference(q_robot)
        # robot.dot_robot[:, i])
        # robot.setVelocityReference() ...
        # robot.setEffortReference(tau_robot[:, i])
        self.robot.move()


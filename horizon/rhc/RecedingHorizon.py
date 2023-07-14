import phase_manager.pymanager as pymanager
import phase_manager.pyphase as pyphase
from horizon.problem import Problem
from horizon.rhc.taskInterface import ProblemInterface, TaskInterface
from horizon.rhc.model_description import FullModelInverseDynamics, SingleRigidBodyDynamicsModel
from horizon.rhc.action_manager.ActionManager import ActionManager
import numpy as np
import horizon.utils.kin_dyn as kd
import matplotlib.pyplot as plt
from horizon.ros import replay_trajectory
from typing import Union
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
# from geometry_msgs.msg import PointStamped

# '''
# Load urdf and srdf
# '''
#
# kyon_urdf_folder = rospkg.RosPack().get_path('kyon_urdf')
# kyon_srdf_folder = rospkg.RosPack().get_path('kyon_srdf')
#
# urdf = subprocess.check_output(["xacro",
#                                 kyon_urdf_folder + "/urdf/kyon.urdf.xacro",
#                                 "sensors:=false",
#                                 "upper_body:=false",
#                                 "wheels:=true",
#                                 "payload:=false"])
#
# srdf = subprocess.check_output(["xacro",
#                                 kyon_srdf_folder + "/srdf/kyon.srdf.xacro",
#                                 "sensors:=false",
#                                 "upper_body:=false",
#                                 "wheels:=true",
#                                 "payload:=false"])
# urdf = urdf.decode('utf-8')
# srdf = srdf.decode('utf-8')
#
# file_dir = os.getcwd()

#
# '''
# Initialize Horizon problem
# '''
# ns = 30
# T = 3
# dt = T / ns
#
# prb = Problem(ns, receding=True, casadi_type=cs.SX)
# prb.setDt(dt)
#
# urdf = urdf.replace('continuous', 'revolute')
# kin_dyn = casadi_kin_dyn.CasadiKinDyn(urdf)
#
# q_init = {'hip_roll_1': 0.0,
#           'hip_pitch_1': 0.7,
#           'knee_pitch_1': -1.4,
#           'hip_roll_2': 0.0,
#           'hip_pitch_2': 0.7,
#           'knee_pitch_2': -1.4,
#           'hip_roll_3': 0.0,
#           'hip_pitch_3': -0.7,
#           'knee_pitch_3': 1.4,
#           'hip_roll_4': 0.0,
#           'hip_pitch_4': -0.7,
#           'knee_pitch_4': 1.4,
#           'wheel_joint_1': 0.0,
#           'wheel_joint_2': 0.0,
#           'wheel_joint_3': 0.0,
#           'wheel_joint_4': 0.0}
#
# # q_init = {'hip_roll_1': 0.0,
# #           'hip_pitch_1': -0.3,
# #           'knee_pitch_1': -0.92,
# #           'hip_roll_2': 0.0,
# #           'hip_pitch_2': -0.3,
# #           'knee_pitch_2': -0.92,
# #           'hip_roll_3': 0.0,
# #           'hip_pitch_3': -0.6,
# #           'knee_pitch_3': 1.26,
# #           'hip_roll_4': 0.0,
# #           'hip_pitch_4': -0.6,
# #           'knee_pitch_4': 1.26,
# #           'wheel_joint_1': 0.0,
# #           'wheel_joint_2': 0.0,
# #           'wheel_joint_3': 0.0,
# #           'wheel_joint_4': 0.0}
#
# # 'shoulder_yaw_1': 0.0,
# # 'shoulder_pitch_1': 0.9,
# # 'elbow_pitch_1': 1.68,
# # 'wrist_pitch_1': 0.,
# # 'wrist_yaw_1': 0.,
# # 'shoulder_yaw_2': 0.0,
# # 'shoulder_pitch_2': 0.9,
# # 'elbow_pitch_2': 1.68,
# # 'wrist_pitch_2': 0.,
# # 'wrist_yaw_2': 0.}
#
# base_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
# # base_init = np.array([0.0, 0.0, 0.0, 0.014192, 0.0297842, -0.0230466, 0.9991898])
#
# FK = kin_dyn.fk('ball_1')
# init = base_init.tolist() + list(q_init.values())
# init_pos_foot = FK(q=init)['ee_pos']
# base_init[2] = -init_pos_foot[2]
#
# model = FullModelInverseDynamics(problem=prb,
#                                  kd=kin_dyn,
#                                  q_init=q_init,
#                                  base_init=base_init,
#                                  )
#
# rospy.set_param('mpc/robot_description', urdf)
# bashCommand = 'rosrun robot_state_publisher robot_state_publisher robot_description:=mpc/robot_description'
# process = subprocess.Popen(bashCommand.split(), start_new_session=True)
#
# ti = TaskInterface(prb=prb, model=model)
# ti.setTaskFromYaml(file_dir + '/../config/kyon_to_config.yaml')

# def __init_ros(self):
#     ''' ros implementation, should be done outside'''
#
#     ros_name = f'rhc_{self.name}'
#     rospy.init_node(ros_name)
#     roscpp.init(ros_name, [])
#
#     solution_publisher = rospy.Publisher('/mpc_solution', JointTrajectory, queue_size=10)
#     rospy.sleep(1.)



# stance_duration = 5
# flight_duration = 5
# for c in model.cmap.keys():
#     # stance phase normal
#     stance_phase = pyphase.Phase(stance_duration, f'stance_{c}')
#     if ti.getTask(f'{c}_contact') is not None:
#         stance_phase.addItem(ti.getTask(f'{c}_contact'))
#     else:
#         raise Exception('task not found')
#
#     c_phases[c].registerPhase(stance_phase)
#
#     # flight phase normal
#     flight_phase = pyphase.Phase(flight_duration, f'flight_{c}')
#     init_z_foot = model.kd.fk(c)(q=model.q0)['ee_pos'].elements()[2]
#     ref_trj = np.zeros(shape=[7, flight_duration])
#     ref_trj[2, :] = np.atleast_2d(
#         tg.from_derivatives(flight_duration, init_z_foot, init_z_foot, 0.05, [None, 0, None]))
#     if ti.getTask(f'z_{c}') is not None:
#         flight_phase.addItemReference(ti.getTask(f'z_{c}'), ref_trj)
#     else:
#         raise Exception('task not found')
#     # flight_phase.addConstraint(prb.getConstraints(f'{c}_vert'), nodes=[0 ,flight_duration-1])  # nodes=[0, 1, 2]
#     c_phases[c].registerPhase(flight_phase)

# TODO: people don't like TaskInterface.

# TODO: Maybe create a RecedingHorizonROS

class RecedingHorizon:
    def __init__(self,
                 prb: Problem,
                 model: Union[FullModelInverseDynamics, SingleRigidBodyDynamicsModel],
                 config_file=None,
                 opts=None):

        #

        if opts is None:
            opts = {}

        self.opts = opts

        self.problem = prb
        self.model = model

        # variables for loop
        self.dt = self.problem.getDt()
        self.rate = 1 / self.dt
        self.iteration = 0

        # if config file is available, create taskInterface, otherwise create basic problemInterface
        if config_file:
            self.interface = TaskInterface(prb=self.problem, model=self.model)
            self.interface.setTaskFromYaml(config_file)
        else:
            self.interface = ProblemInterface(prb=self.problem, model=self.model)

        # self.__set_initial_conditions()
        self.__init_phase_manager()

        # flags
        self.__ros_flag = False
        self.__replay_flag = False
        self.__plot_flag = False
        self.__boostrap_solved = False

        if "input" in self.opts:
            if "joystick" in self.opts["input"] and self.opts["input"]["joystick"] is True:
                self.__init_joy()

        if "ros" in self.opts and self.opts["ros"] is True:
            self.__ros_flag = self.__init_ros()

        if "replay" in self.opts and self.opts["replay"] is True:
            self.__replay_flag = self.__init_replayer()

        self.__figs = dict()

    def __init_ros(self):
        import subprocess
        import rospy

        name = 'fungo'
        rospy.set_param('/robot_description', self.model.kd.urdf())
        bashCommand = 'rosrun robot_state_publisher robot_state_publisher'
        subprocess.Popen(bashCommand.split(), start_new_session=True)

        ros_name = f'rhc_{name}'
        rospy.init_node(ros_name)

        solution_publisher = rospy.Publisher('/mpc_solution', JointTrajectory, queue_size=10)
        rospy.sleep(1.)

        return True


    def __init_phase_manager(self):

        # create phaseManager for receding horizon
        self.__pm = pymanager.PhaseManager(self.problem.getNNodes() - 1)

        # default create a phase for each contact specified in the model
        contact_phases = dict()
        for c in self.model.cmap.keys():
            contact_phases[c] = self.__pm.addTimeline(f'{c}_timeline')

    def __init_action_manager(self):

        pass
        # create actionManager for simple horizon
        # self.am = ActionManager(self.interface, self.pm, self.model.cmap.keys())

    def __set_default_action(self):

        pass
        # for c in model.cmap.keys():
        #     stance = c_phases[c].getRegisteredPhase(f'stance_{c}')
        #     flight = c_phases[c].getRegisteredPhase(f'flight_{c}')
        #     c_phases[c].addPhase(stance)
        #     c_phases[c].addPhase(stance)
        #     c_phases[c].addPhase(stance)
        #     c_phases[c].addPhase(stance)
        #     c_phases[c].addPhase(stance)

    def set_initial_conditions(self):

        # todo not very flexible
        # initialize the robot as still, in the initial position, with weight equally distributed on contacts

        # #set initial bounds and initial guess
        for name, elem in self.model.getState().items():

            if self.model.getInitialState(name) is not None:
                elem.setInitialGuess(self.model.getInitialState(name))
                elem.setBounds(self.model.getInitialState(name), self.model.getInitialState(name), nodes=0)

        for name, elem in self.model.getInput().items():

            if self.model.getInitialState(name) is not None:
                elem.setBounds(self.model.getInitialInput(name), self.model.getInitialInput(name), nodes=0)
                elem.setInitialGuess(self.model.getInitialInput(name))

        f0 = [0, 0, self.model.kd.mass() / len(self.model.getForceMap().keys()) * 9.8]
        for cname, cforces in self.model.getContactMap().items():
            for c in cforces:
                c.setInitialGuess(f0)

    def set_solver_options(self, solver_options):
        # todo: keep it here?
        # this raises a question: should I make recedingHorizon a child of Task/Problem Interface
        self.interface.setSolverOptions(solver_options)

    def bootstrap(self):
        # finalize taskInterface and solve bootstrap problem
        self.interface.finalize()
        self.interface.bootstrap()

        self.interface.load_initial_guess()
        self.__solution = self.interface.solution

        self.__boostrap_solved = True

    def __init_replayer(self):

        if self.__ros_flag is False:
            raise Exception("ROS required for replayer.")

        contact_list_repl = list(self.model.cmap.keys())
        self.repl = replay_trajectory.replay_trajectory(self.dt, self.model.kd.joint_names(), np.array([]),
                                                   {k: None for k in self.model.fmap.keys()},
                                                   self.model.kd_frame, self.model.kd,
                                                   trajectory_markers=contact_list_repl)

        # future_trajectory_markers={'base_link': 'world', 'ball_1': 'world'})
        return True


    def __init_joy(self):
        # todo
        from horizon.utils.JoystickInterface import JoyInterface
        self.jc = JoyInterface()


    def __publish_solution(self):

        if not self.__ros_flag:
            raise Exception("ROS required for publishing the solution.")

    # def plot(self, var):
    #
    #     if var in self.__figs[var]:
    #         self.__figs[var] = plt.figure()
    #         ax_plot = plt.subplot()
    #
    #     plt.ion()
    #     plt.show()
    #     x_axis = range(ns + 1)
    #     y_axis = var
    #     ax_plot.scatter(x_axis, y_axis)
    #
    #     plt.pause(0.001)
    #     ax_plot.clear()


    def __init_run(self):

        time_elapsed_shifting_list = list()
        xig = np.empty([self.problem.getState().getVars().shape[0], 1])
        return xig


    def setInput(self):

        # // as a phase
        pass
    def run(self):

        if not self.__boostrap_solved:
            print('\033[1m' + 'Bootstrap not found. Bootstrapping...' + '\033[0m')
            self.bootstrap()
            print('\033[1m' + 'Start looping...' + '\033[0m')

        # set initial state and initial guess
        shift_num = -1

        x_opt = self.__solution['x_opt']
        xig = np.roll(x_opt, shift_num, axis=1)
        for i in range(abs(shift_num)):
            xig[:, -1 - i] = x_opt[:, -1]

        self.problem.getState().setInitialGuess(xig)
        self.problem.setInitialState(x0=xig[:, 0])

        # shift phases of phase manager
        # tic = time.time()
        self.__pm._shift_phases()
        # time_elapsed_shifting = time.time() - tic
        # time_elapsed_shifting_list.append(time_elapsed_shifting)

        # self.jc.run(self.__solution)

        self.iteration = self.iteration + 1

        self.interface.rti()
        self.__solution = self.interface.solution
        dt_res = 0.01
        self.interface.resample(dt_res=dt_res, nodes=[0, 1], resample_tau=False)

        tau = list()

        for i in range(self.__solution['q_res'].shape[1] - 1):
            tau.append(self.interface.model.computeTorqueValues(self.__solution['q_res'][:, i], self.__solution['v_res'][:, i],
                                                    self.__solution['a_res'][:, i],
                                                    {name: self.__solution['f_' + name][:, i] for name in self.model.fmap}))

        # jt = JointTrajectory()
        # for i in range(self.__solution['q_res'].shape[1]):
        #     jtp = JointTrajectoryPoint()
        #     jtp.positions = self.__solution['q_res'][:, i].tolist()
        #     jtp.velocities = self.__solution['v_res'][:, i].tolist()
        #     if i < len(tau):
        #         jtp.accelerations = self.__solution['a_res'][:, i].tolist()
        #         jtp.effort = tau[i].elements()
        #     else:
        #         jtp.accelerations = solution['a_res'][:, -1].tolist()
        #         jtp.effort = tau[-1].elements()
        #
        #     jt.points.append(jtp)
        #
        # jt.joint_names = [elem for elem in kin_dyn.joint_names() if elem not in ['universe', 'reference']]
        # jt.header.stamp = rospy.Time.now()
        #
        # solution_publisher.publish(jt)

        if self.__replay_flag:

            # replay stuff
            self.repl.frame_force_mapping = {cname: self.__solution[f.getName()] for cname, f in self.model.getForceMap().items()}
            self.repl.publish_joints(self.__solution['q'][:, 0])
            self.repl.publishContactForces(rospy.Time.now(), self.__solution['q'][:, 0], 0)
        # repl.publish_future_trajectory_marker('base_link', self.__solution['q'][0:3, :])
        # repl.publish_future_trajectory_marker('ball_1', self.__solution['q'][8:11, :])

if __name__ == '__main__':

    import rospkg
    import casadi_kin_dyn.py3casadi_kin_dyn as casadi_kin_dyn
    import casadi as cs
    import rospy

    np.set_printoptions(suppress=False,  # suppress small results
                        linewidth=np.inf,
                        precision=3,
                        threshold=np.inf,  # number of displayed elements for array
                        formatter=None
                        )
    

    cogimon_urdf_folder = rospkg.RosPack().get_path('cogimon_urdf')
    cogimon_srdf_folder = rospkg.RosPack().get_path('cogimon_srdf')

    urdf = open(cogimon_urdf_folder + '/urdf/cogimon.urdf', 'r').read()


    ns = 20
    T = 1.
    dt = T / ns

    prb = Problem(ns, receding=True, casadi_type=cs.SX)
    prb.setDt(dt)

    base_init = np.atleast_2d(np.array([0.03, 0., 0.962, 0., -0.029995, 0.0, 0.99955]))
    # base_init = np.array([0., 0., 0.96, 0., 0.0, 0.0, 1.])

    q_init = {"LHipLat": -0.0,
              "LHipSag": -0.363826,
              "LHipYaw": 0.0,
              "LKneePitch": 0.731245,
              "LAnklePitch": -0.307420,
              "LAnkleRoll": 0.0,
              "RHipLat": 0.0,
              "RHipSag": -0.363826,
              "RHipYaw": 0.0,
              "RKneePitch": 0.731245,
              "RAnklePitch": -0.307420,
              "RAnkleRoll": -0.0,
              "WaistLat": 0.0,
              "WaistYaw": 0.0,
              "LShSag": 1.1717860,
              "LShLat": -0.059091562,
              "LShYaw": -5.18150657e-02,
              "LElbj": -1.85118,
              "LForearmPlate": 0.0,
              "LWrj1": -0.523599,
              "LWrj2": -0.0,
              "RShSag": 1.17128697,
              "RShLat": 6.01664139e-02,
              "RShYaw": 0.052782481,
              "RElbj": -1.8513760,
              "RForearmPlate": 0.0,
              "RWrj1": -0.523599,
              "RWrj2": -0.0}

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

    model_kin_dyn = casadi_kin_dyn.CasadiKinDyn(urdf)

    model = FullModelInverseDynamics(problem=prb,
                                     kd=model_kin_dyn,
                                     q_init=q_init,
                                     base_init=base_init,
                                     contact_dict=contact_dict,
                                     sys_order_degree=3)

    model._make_vertex_contact('r_sole', dict(vertex_frames=['r_foot_lower_left_link',
                                                             'r_foot_upper_left_link',
                                                             'r_foot_lower_right_link',
                                                             'r_foot_upper_right_link']))

    model._make_vertex_contact('l_sole', dict(vertex_frames=['l_foot_lower_left_link',
                                                             'l_foot_upper_left_link',
                                                             'l_foot_lower_right_link',
                                                             'l_foot_upper_right_link']))


    opts = dict(
        ros=True,
        replay=True
    )
    mi = RecedingHorizon(prb=prb, model=model, opts=opts)
    mi.set_initial_conditions()

    solver_opt = dict(type="ilqr")
    mi.set_solver_options(solver_opt)


    # for var_name, var in mi.model.getState().items():
    #     print(f'name: {var_name}')
    #     print(f'intial guess: {var.getInitialGuess()}')
    #     print(f'bounds: {var.getBounds()}')

    while not rospy.is_shutdown():
        mi.run()


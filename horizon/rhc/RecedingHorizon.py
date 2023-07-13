import phase_manager.pymanager as pymanager
import phase_manager.pyphase as pyphase
from horizon.problem import Problem
from horizon.rhc.taskInterface import ProblemInterface, TaskInterface
from horizon.rhc.model_description import FullModelInverseDynamics, SingleRigidBodyDynamicsModel
from horizon.rhc.action_manager.ActionManager import ActionManager
import numpy as np
import horizon.utils.kin_dyn as kd
import rospy, roscpp
from joy_commands import JoyCommands
from horizon.ros import replay_trajectory
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
# maybe a taskInterface bare? without tasks, only with stuff to compute, solve, ...
class RecedingHorizon:
    def __init__(self,
                 problem: Problem,
                 model: FullModelInverseDynamics | SingleRigidBodyDynamicsModel,
                 config_file=None,
                 opts=None):

        self.problem = problem
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

    def __init_phase_manager(self):

        # create phaseManager for receding horizon
        self.pm = pymanager.PhaseManager(self.problem.getNNodes() - 1)

        # default create a phase for each contact specified in the model
        contact_phases = dict()
        for c in self.model.cmap.keys():
            contact_phases[c] = self.pm.addTimeline(f'{c}_timeline')

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

    def __set_initial_conditions(self):

        self.problem.setInitialState(self.model.getInitialValue())
        #set initial bounds
        self.interface.model.getVariable['q'].setBounds(self.interface.model.q0, self.interface.model.q0, nodes=0)
        self.interface.model.v.setBounds(self.interface.model.v0, self.interface.model.v0, nodes=0)
        self.interface.model.a.setBounds(np.zeros([self.model.a.shape[0], 1]), np.zeros([self.model.a.shape[0], 1]), nodes=0)

        # set initial guess
        self.interface.model.q.setInitialGuess(self.interface.model.q0)
        self.interface.model.v.setInitialGuess(self.interface.model.v0)


        f0 = [0, 0, self.model.kd.mass() / len(self.model.cmap.keys()) * 9.8]
        for cname, cforces in self.interface.model.cmap.items():
            for c in cforces:
                c.setInitialGuess(f0)

    def bootstrap(self):
        # finalize taskInterface and solve bootstrap problem
        self.interface.finalize()
        self.interface.bootstrap()

        self.interface.load_initial_guess()
        self.__solution = self.interface.solution

    def __init_replayer(self):

        contact_list_repl = list(self.model.cmap.keys())
        self.repl = replay_trajectory.replay_trajectory(self.dt, self.model.kd.joint_names(), np.array([]),
                                                   {k: None for k in self.model.fmap.keys()},
                                                   self.model.kd_frame, self.model.kd,
                                                   trajectory_markers=contact_list_repl)

        # future_trajectory_markers={'base_link': 'world', 'ball_1': 'world'})

    def __init_joy(self):

        self.jc = JoyCommands(self.gm)



    def __init_run(self):

        time_elapsed_shifting_list = list()
        xig = np.empty([self.prb.getState().getVars().shape[0], 1])


    def setInput(self):

        # // as a phase
        pass
    def run(self):

        # set initial state and initial guess
        shift_num = -1

        x_opt = self.__solution['x_opt']
        xig = np.roll(x_opt, shift_num, axis=1)
        for i in range(abs(shift_num)):
            xig[:, -1 - i] = x_opt[:, -1]

        self.prb.getState().setInitialGuess(xig)
        self.prb.setInitialState(x0=xig[:, 0])

        # shift phases of phase manager
        # tic = time.time()
        self.pm._shift_phases()
        # time_elapsed_shifting = time.time() - tic
        # time_elapsed_shifting_list.append(time_elapsed_shifting)

        self.jc.run(self.__solution)

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



        # replay stuff
        self.repl.frame_force_mapping = {cname: self.__solution[f.getName()] for cname, f in ti.model.fmap.items()}
        self.repl.publish_joints(self.__solution['q'][:, 0])
        self.repl.publishContactForces(rospy.Time.now(), self.__solution['q'][:, 0], 0)
        # repl.publish_future_trajectory_marker('base_link', self.__solution['q'][0:3, :])
        # repl.publish_future_trajectory_marker('ball_1', self.__solution['q'][8:11, :])


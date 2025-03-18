import rospy
from Cython.Compiler.TreePath import operations
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool, SetBoolRequest, Trigger, TriggerResponse
from std_msgs.msg import String
from horizon_msgs.msg import OperationMode as OperationModeMsg
from horizon.rhc.gait_manager import GaitManager, PhaseGaitWrapper
import numpy as np
from enum import IntEnum, Enum
from horizon.utils.logger import Logger
from typing import Callable, Union
from functools import partial

# marker = Marker()
# marker.header.frame_id = 'world'
# marker.header.stamp = rospy.Time.now()
# marker.id = 1
# marker.action = Marker.ADD
# marker.scale.x = 0.05
# marker.scale.y = 0.05
# marker.scale.z = 0.05
# marker.color.r = 1
# marker.color.g = 0
# marker.color.b = 0
# marker.color.a = 1
# marker.type = Marker.SPHERE
# marker.pose.position.x = reference[0]
# marker.pose.position.y = reference[1]
# marker.pose.position.z = reference[2]
# marker.pose.orientation.x = reference[3]
# marker.pose.orientation.y = reference[4]
# marker.pose.orientation.z = reference[5]
# marker.pose.orientation.w = reference[6]
# self.__pub.publish(marker)

# todo: this should go in the gait_manager and should be extracted here, automatically creating the binding between action and topic/buttons
class OperationMode(Enum):
    STAND = 0
    CRAWL = 1
    TROT = 2
    STEP = 3
    DRAG = 4
    WALK = 5
    IDLE = 6

class GaitManagerROS:
    def __init__(self, gm: Union[GaitManager, PhaseGaitWrapper], opt : dict = None):

        self.__opt = opt
        self.__logger = Logger(self)

        self.__gait_manager = gm
        self.__ti = self.__gait_manager.getTaskInterface()

        self.__operation_mode_locked = False

        # horizon duration
        self.__T = self.__ti.getProblem().getDt() * (self.__ti.getProblem().getNNodes() - 1)

        # this should be wrapped
        self.__base_pose_weight = 1. # should be one weight for rot and pos
        self.__base_rot_weight = 1.

        # this version receives commands as base velocity
        # open ros topic
        self.__base_vel_sub = rospy.Subscriber('/horizon/base_velocity/reference', Twist, self.__base_vel_cb)

        # init tasks connection
        self.__init_options()
        self.__base_pose_xy_task = self.__ti.getTask(self.__base_pose_xy_task_name)
        self.__base_pose_z_task = self.__ti.getTask(self.__base_pose_z_task_name)
        self.__base_yaw_ori_task = self.__ti.getTask(self.__base_yaw_orientation_task_name)

        self.__base_vel_ref = np.zeros(6)

        self.__operation_mode_pub = rospy.Publisher('/horizon/operation_mode', OperationModeMsg, queue_size=10)
        # self.__action_switch_srv_dict = dict()
        # for action_name in self.__gait_manager.getActionList():
            # self.__action_switch_srv_dict.update({action_name, rospy.Service(f'/horizon/{action_name}/switch', SetBool, partial(self.__switch_action_cb, action_name))})
            # open ros services
        self.__switch_crawl_srv = rospy.Service('/horizon/crawl/switch', SetBool, self.__switch_crawl_cb)
        self.__switch_trot_srv = rospy.Service('/horizon/trot/switch', SetBool, self.__switch_trot_cb)
        self.__switch_step_srv = rospy.Service('/horizon/step/switch', SetBool, self.__switch_step_cb)
        self.__switch_drag_srv = rospy.Service('/horizon/drag/switch', SetBool, self.__switch_drag_cb)
        self.__switch_walk_srv = rospy.Service('/horizon/walk/switch', SetBool, self.__switch_walk_cb)
        self.__switch_idle_srv = rospy.Service('/horizon/idle/switch', SetBool, self.__switch_idle_cb)
        self.__switch_stand_srv = rospy.Service('/horizon/stand/switch', SetBool, self.__switch_stand_cb)
        # self.__switch_postural_srv = rospy.Service('/horizon/postural/switch', SetBool, self.__switch_postural_cb)

        # param

        self.__param_action = dict()
        self.__param_action['walk'] = {'step_duration': 10, 'step_height': 0.05, 'double_stance': 3}
        self.__param_action['stand'] = {'duration': 1}
        self.__param_action['crawl'] = {'step_duration': 10, 'step_height': 0.05, 'double_stance': 3}
        self.__param_action['trot'] = {'step_duration': 10, 'step_height': 0.1, 'double_stance': 3}
        # self.__param_action['postural'] = {'type': 'None'}

        self.__walk_params_ros = dict()

        for action_name, param_actions in self.__param_action.items():
            self.__walk_params_ros[action_name] = dict()
            for param_name, param_value in param_actions.items():
                self.__walk_params_ros[action_name][param_name] = rospy.set_param(f'/horizon/{action_name}/{param_name}', param_value)

        # self.__contact_params_srv = dict()
        # self.__contact_params = dict()

        # contact_param_dict = {'duration':10, 'height':0.05}
        #
        # for contact_name in self.__gait_manager.getContacts():
        #
        #     self.__contact_params[contact_name] = dict()
        #     self.__contact_params_srv[contact_name] = dict()

            # for param_name, param_value in contact_param_dict.items():
            #     self.__contact_params[contact_name][param_name] = param_value
            #     self.__contact_params_srv[contact_name][param_name] = rospy.set_param(f'~{contact_name}/{param_name}', self.__contact_params[contact_name][param_name])
        #
        self.__current_solution = None

        # initialize initial operation mode
        self.__operation_mode = OperationMode.IDLE

        self.__init_actions()
        self.__init_plugins()
        # get one random contact phase to check when to add new phases
        self.__one_random_contact_timeline = next(iter(self.__gait_manager.getContactTimelines().values()))

        self.__logger.log(f'Timeline used to check if horizon tail is empty: {self.__one_random_contact_timeline.getName()}')

    def __init_actions(self):

        self.__action_dict = {
                                OperationMode.IDLE: partial(self.__gait_manager.action, 'stand'),
                                OperationMode.STAND: partial(self.__gait_manager.action, 'stand'),
                                OperationMode.TROT:  lambda: self.__gait_manager.action('trot', **self.__get_params('trot')),
                                OperationMode.WALK: lambda: self.__gait_manager.action('walk', **self.__get_params('walk')),
                                OperationMode.CRAWL: lambda: self.__gait_manager.action('crawl', **self.__get_params('crawl')),
                                # OperationMode.POSTURAL: lambda: self.__gait_manager.action('postural', **self.__get_params('postural'))
                                # OperationMode.DRAG: self.__gm.drag,
                                # OperationMode.STEP: lambda: self.__gm.step(swing_contact='ball_1')
        }

    def __init_plugins(self):

        self.__plugin_dict = dict()

        # self.__actions_manager = dict()

        for plugin_name, plugin in self.__gait_manager.getPluginDict().items():

            for action_name, action_callable in plugin.get_actions().items():

                # self.__actions_manager[action_name] = {'action_callable': action_callable, 'status':'Started'}

                if callable(action_callable):
                    service_name = f'/horizon/gait_manager/{plugin_name}/{action_name}/start'
                    rospy.Service(service_name, Trigger, self.__create_service_wrapper(plugin_name, action_name))
                    self.__logger.log(f"Registered '{plugin_name}:{action_name}' as ROS service: {service_name}")

    def __create_service_wrapper(self, plugin_name, action_name):
        """Wrap the method to return a TriggerResponse for ROS service."""
        def wrapper(req):

            if self.__operation_mode != OperationMode.IDLE:
                return TriggerResponse(success=False, message=f"{action_name} cannot start.")

                # Call the actual method
            self.__gait_manager.getPluginDict()[plugin_name].setStatus(action_name, 'Started')

            # Return a generic TriggerResponse for ROS service
            return TriggerResponse(success=True, message=f"{action_name} started.")

        return wrapper

    def __get_params(self, action_name) -> dict :

        for param_name, ros_param in self.__walk_params_ros[action_name].items():
            self.__param_action[action_name][param_name] = rospy.get_param(f'/horizon/{action_name}/{param_name}')

        return self.__param_action[action_name]

    def setBasePoseWeight(self, w):
        self.__base_pose_weight = w

    def getBasePoseWeight(self):
        return self.__base_pose_weight

    def setBaseRotWeight(self, w):
        self.__base_rot_weight = w

    def __init_options(self):

        if self.__opt is None:
            self.__opt = dict()

        # default values
        self.__base_pose_xy_task_name = 'base_xy'
        self.__base_pose_z_task_name = 'base_z'
        self.__base_yaw_orientation_task_name = 'base_yaw_orientation'

        if 'task_name' in self.__opt:
            task_name_dict = self.__opt['task_name']
            base_pose_xy_key = 'base_pose_xy'
            base_pose_z_key = 'base_pose_z'
            base_orientation_key = 'base_orientation'

            if base_pose_xy_key in task_name_dict:
                self.__base_pose_xy_task_name = task_name_dict[base_pose_xy_key]

            if base_pose_z_key in task_name_dict:
                self.__base_pose_z_task_name = task_name_dict[base_pose_z_key]

            if base_orientation_key in task_name_dict:
                self.__base_orientation_task_name = task_name_dict[base_orientation_key]

    def __update_solution(self):

        self.__current_solution = self.__ti.solution

    def __base_vel_cb(self, msg: Twist):

        if not self.__operation_mode_locked:
            self.__base_vel_ref[0] = msg.linear.x
            self.__base_vel_ref[1] = msg.linear.y
            self.__base_vel_ref[2] = msg.linear.z
            self.__base_vel_ref[3] = msg.angular.x
            self.__base_vel_ref[4] = msg.angular.y
            self.__base_vel_ref[5] = msg.angular.z

    # def __switch_action_cb(self, req: SetBoolRequest, action_name):

    def __switch_idle_cb(self, req: SetBoolRequest):

        if req.data:

            if self.__operation_mode_locked:
                return {'success': False}

            self.__operation_mode = OperationMode.IDLE
        else:
            if self.__operation_mode == OperationMode.IDLE:
                self.__operation_mode = OperationMode.STAND

        return {'success': True}

    def __switch_stand_cb(self, req: SetBoolRequest):

        if req.data:

            if self.__operation_mode_locked:
                return {'success': False}

            self.__operation_mode = OperationMode.STAND
        else:
            if self.__operation_mode == OperationMode.STAND:
                self.__operation_mode = OperationMode.STAND

        return {'success': True}

    def __switch_crawl_cb(self, req: SetBoolRequest):

        if req.data:

            if self.__operation_mode_locked:
                return {'success': False, 'message': 'OperationMode is locked.'}

            if self.__operation_mode == OperationMode.IDLE:
                return {'success': False, 'message': 'Cannot transition to CRAWL from IDLE.'}

            self.__operation_mode = OperationMode.CRAWL
        else:
            if self.__operation_mode == OperationMode.CRAWL:
                self.__operation_mode = OperationMode.STAND

        return {'success': True}

    def __switch_trot_cb(self, req: SetBoolRequest):

        if req.data:

            if self.__operation_mode_locked:
                return {'success': False, 'message': 'OperationMode is locked.'}

            if self.__operation_mode == OperationMode.IDLE:
                return {'success': False, 'message': 'Cannot transition to TROT from IDLE.'}

            self.__operation_mode = OperationMode.TROT
        else:
            if self.__operation_mode == OperationMode.TROT:
                self.__operation_mode = OperationMode.STAND

        return {'success': True}

    def __switch_step_cb(self, req: SetBoolRequest):

        if req.data:

            if self.__operation_mode_locked:
                return {'success': False, 'message': 'OperationMode is locked.'}

            if self.__operation_mode == OperationMode.IDLE:
                return {'success': False, 'message': 'Cannot transition to STEP from IDLE.'}

            self.__operation_mode = OperationMode.STEP
        else:
            if self.__operation_mode == OperationMode.STEP:
                self.__operation_mode = OperationMode.STAND

        return {'success': True}

    def __switch_drag_cb(self, req: SetBoolRequest):

        if req.data:

            if self.__operation_mode_locked:
                return {'success': False, 'message': 'OperationMode is locked.'}

            if self.__operation_mode == OperationMode.IDLE:
                return {'success': False, 'message': 'Cannot transition to DRAG from IDLE.'}

            self.__operation_mode = OperationMode.DRAG
        else:
            if self.__operation_mode == OperationMode.DRAG:
                self.__operation_mode = OperationMode.STAND

        return {'success': True}

    def __switch_walk_cb(self, req: SetBoolRequest):

        if req.data:

            if self.__operation_mode_locked:
                return {'success': False, 'message': 'OperationMode is locked.'}

            if self.__operation_mode == OperationMode.IDLE:
                return {'success': False, 'message': 'Cannot transition to WALK from IDLE.'}

            self.__operation_mode = OperationMode.WALK
        else:
            if self.__operation_mode == OperationMode.WALK:
                self.__operation_mode = OperationMode.STAND

        return {'success': True}


    def __set_phases(self, *args, **kwargs):

        if self.__one_random_contact_timeline.getEmptyNodes() > 0:
            action: Callable = self.__action_dict.get(self.__operation_mode)

            if action:
                action(*args, **kwargs)  # Call the function
            else:
                self.__logger.log("Invalid operation mode")


    def __set_base_commands(self):

        # =========================== X Y  ================================
        if self.__base_pose_xy_task.getCartesianType() == 'position':
            base_reference_xy = np.array([[self.__current_solution['q'][0, 0], # x pos at node 0
                                           self.__current_solution['q'][1, 0], # x pos at node 1
                                           0., 0., 0., 0., 0.]]).T
        else:
            base_reference_xy = np.atleast_2d(self.__base_pose_xy_task.getValues()[:, -1]).T

        # move base on xy-axis in local frame
        linear_velocity_vector = np.array([self.__base_pose_weight * self.__base_vel_ref[0],
                                           self.__base_pose_weight * self.__base_vel_ref[1],
                                           0])

        # rotate in local base
        linear_velocity_vector_rot = self.__rotate_vector(linear_velocity_vector, self.__current_solution['q'][[6, 3, 4, 5], 0])

        if self.__base_pose_xy_task.getCartesianType() == 'velocity':
            base_reference_xy[0] = linear_velocity_vector_rot[0]
            base_reference_xy[1] = linear_velocity_vector_rot[1]
            self.__base_pose_xy_task.setRef(base_reference_xy[0:2])
        else:
            base_reference_xy[0] += linear_velocity_vector_rot[0] * self.__T
            base_reference_xy[1] += linear_velocity_vector_rot[1] * self.__T
            self.__base_pose_xy_task.setRef(base_reference_xy)

        # ============================ YAW  =================================



        if self.__base_yaw_ori_task.getCartesianType() == 'position':
            base_reference = np.array([[0., 0., 0., 0, 0, 0, 0]]).T

            d_angle = np.pi / 2 * self.__base_rot_weight * self.__base_vel_ref[5]
            axis = [0, 0, 1]
            angular_velocity_vector = self.__incremental_rotate(np.atleast_2d(self.__base_yaw_ori_task.getValues()[[6, 3, 4, 5], 0]).T, d_angle, axis)

            base_reference[3] += angular_velocity_vector.x
            base_reference[4] += angular_velocity_vector.y
            base_reference[5] += angular_velocity_vector.z
            base_reference[6] += angular_velocity_vector.w

            self.__base_yaw_ori_task.setRef(base_reference)

        else:
            angular_velocity_vector = self.__base_rot_weight * self.__base_vel_ref[5]
            self.__base_yaw_ori_task.setRef(angular_velocity_vector)


        # ============================= Z =================================

        # can add to xy
        base_reference_z = np.atleast_2d(self.__base_pose_z_task.getValues()[:, 0]).T

        linear_velocity_vector_z = np.array([0.,
                                             0.,
                                             self.__base_pose_weight * self.__base_vel_ref[2]])

        base_reference_z[2] += linear_velocity_vector_z[2] * self.__T

        self.__base_pose_z_task.setRef(base_reference_z)


    def run(self):

        self.__logger.log(f'operation mode: {self.__operation_mode}')
        self.publish_operation_mode()

        self.__update_solution()

        # set phases
        self.__set_phases()

        if self.__operation_mode == OperationMode.IDLE:
            self.__run_plugins()

        # set base_commands
        self.__set_base_commands()


    def publish_operation_mode(self):
        msg = OperationModeMsg()
        msg.mode = self.__operation_mode.value
        msg.mode_name = self.__operation_mode.name
        self.__operation_mode_pub.publish(msg)

    def __run_plugins(self):

        self.__operation_mode_locked = False

        for plugin_name, plugin in self.__gait_manager.getPluginDict().items():

            for action_name, status in plugin.getStatus().items():
                if status == 'Started' or status == 'Running':
                    self.__operation_mode_locked = True
                    plugin.get_actions()[action_name]()


    def __incremental_rotate(self, q_initial: np.quaternion, d_angle, axis) -> np.quaternion:
        # np.quaternion is [w,x,y,z]
        q_incremental = np.array([np.cos(d_angle / 2),
                                  axis[0] * np.sin(d_angle / 2),
                                  axis[1] * np.sin(d_angle / 2),
                                  axis[2] * np.sin(d_angle / 2)
                                  ])

        # normalize the quaternion
        q_incremental /= np.linalg.norm(q_incremental)

        # initial orientation of the base

        # final orientation of the base
        q_result = np.quaternion(*q_incremental) * np.quaternion(*q_initial)

        return q_result

    def __rotate_vector(self, vector, quaternion):

        # normalize the quaternion
        quaternion = quaternion / np.linalg.norm(quaternion)

        # construct a pure quaternion
        v = np.array([0, vector[0], vector[1], vector[2]])

        # rotate the vector p = q* v q
        rotated_v = self.__quaternion_multiply(quaternion,
                                              self.__quaternion_multiply(v, self.__conjugate_quaternion(quaternion)))

        # extract the rotated vector
        rotated_vector = rotated_v[1:]

        return rotated_vector

    def __quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])


    def __conjugate_quaternion(self, q):
        q_conjugate = np.copy(q)
        q_conjugate[1:] *= -1.0
        return q_conjugate
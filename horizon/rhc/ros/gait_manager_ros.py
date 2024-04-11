import rospy
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool, SetBoolRequest
from horizon.rhc.gait_manager import GaitManager
import numpy as np
from enum import Enum

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


class OperationMode(Enum):
    STAND = 0
    CRAWL = 1
    TROT = 2

class GaitManagerROS:
    def __init__(self, gm: GaitManager, opt : dict = None):

        self.__opt = opt

        self.__gm = gm
        self.__ti = self.__gm.getTaskInterface()

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
        self.__base_ori_task = self.__ti.getTask(self.__base_orientation_task_name)

        self.__base_vel_ref = np.zeros(6)

        # open ros services
        self.__switch_walk_srv = rospy.Service('/horizon/walk/switch', SetBool, self.__switch_walk_cb)
        self.__switch_trot_srv = rospy.Service('/horizon/trot/switch', SetBool, self.__switch_trot_cb)

        self.__current_solution = None

        # initialize initial operation mode
        self.__operation_mode = OperationMode.STAND

        # get one random contact phase to check when to add new phases
        self.__one_random_contact_timeline = next(iter(self.__gm.getContactTimelines().values()))

        print('Timeline used to check if horizon tail is empty: ', self.__one_random_contact_timeline.getName())


    def __init_options(self):

        if self.__opt is None:
            self.__opt = dict()

        # default values
        self.__base_pose_xy_task_name = 'base_xy'
        self.__base_pose_z_task_name = 'base_z'
        self.__base_orientation_task_name = 'base_orientation'

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

        self.__base_vel_ref[0] = msg.linear.x
        self.__base_vel_ref[1] = msg.linear.y
        self.__base_vel_ref[2] = msg.linear.z
        self.__base_vel_ref[3] = msg.angular.x
        self.__base_vel_ref[4] = msg.angular.y
        self.__base_vel_ref[5] = msg.angular.z

    def __switch_walk_cb(self, req: SetBoolRequest):

        if req.data:
            self.__operation_mode = OperationMode.CRAWL
        else:
            if self.__operation_mode == OperationMode.CRAWL:
                self.__operation_mode = OperationMode.STAND

        return {'success': True}

    def __switch_trot_cb(self, req: SetBoolRequest):

        if req.data:
            self.__operation_mode = OperationMode.TROT
        else:
            if self.__operation_mode == OperationMode.TROT:
                self.__operation_mode = OperationMode.STAND

        return {'success': True}

    def __set_phases(self):

        if self.__operation_mode == OperationMode.CRAWL:
            if self.__one_random_contact_timeline.getEmptyNodes() > 0:
                self.__gm.crawl(vref=self.__base_vel_ref[[0, 1, 5]])

        if self.__operation_mode == OperationMode.TROT:
            if self.__one_random_contact_timeline.getEmptyNodes() > 0:
                self.__gm.trot()

        if self.__operation_mode == OperationMode.STAND:
            if self.__one_random_contact_timeline.getEmptyNodes() > 0:
                self.__gm.stand()

    def __set_base_commands(self):

        # =========================== X Y  ================================
        base_reference_xy = np.array([[self.__current_solution['q'][0, 0], # x pos at node 0
                                       self.__current_solution['q'][1, 0], # x pos at node 1
                                       0., 0., 0., 0., 0.]]).T

        # move base on xy-axis in local frame
        linear_velocity_vector = np.array([self.__base_pose_weight * self.__base_vel_ref[0],
                                           self.__base_pose_weight * self.__base_vel_ref[1],
                                           0])

        # rotate in local base
        linear_velocity_vector_rot = self.__rotate_vector(linear_velocity_vector, self.__current_solution['q'][[6, 3, 4, 5], 0])

        base_reference_xy[0] += linear_velocity_vector_rot[0] * self.__T
        base_reference_xy[1] += linear_velocity_vector_rot[1] * self.__T

        self.__base_pose_xy_task.setRef(base_reference_xy)

        # ============================ YAW  =================================

        base_reference_yaw = np.array([[0., 0., 0., 0, 0, 0, 0]]).T

        d_angle = np.pi / 2 * self.__base_rot_weight * self.__base_vel_ref[5]
        axis = [0, 0, 1]

        angular_velocity_vector = self.__incremental_rotate(self.__current_solution['q'][[6, 3, 4, 5], 0], d_angle, axis)

        base_reference_yaw[3] = angular_velocity_vector.x
        base_reference_yaw[4] = angular_velocity_vector.y
        base_reference_yaw[5] = angular_velocity_vector.z
        base_reference_yaw[6] = angular_velocity_vector.w

        self.__base_ori_task.setRef(base_reference_yaw)

        # ============================= Z =================================

        # can add to xy
        base_reference_z = np.array([[0., 0, self.__current_solution['q'][2, 0], 0., 0., 0., 0.]]).T

        linear_velocity_vector_z = np.array([0.,
                                             0.,
                                             self.__base_pose_weight * self.__base_vel_ref[2]])

        base_reference_z[2] += linear_velocity_vector_z[2] * self.__T

        self.__base_pose_z_task.setRef(base_reference_z)




    def run(self):

        self.__update_solution()

        # set phases
        self.__set_phases()

        # set base_commands
        self.__set_base_commands()



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
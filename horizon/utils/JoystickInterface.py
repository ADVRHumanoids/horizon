import numpy as np
from horizon.rhc.action_manager import ActionManager
from sensor_msgs.msg import Joy
import rospy
import math

class JoyInterface:
    def __init__(self):
        self.joy_msg = None

        rospy.Subscriber('/joy', Joy, self.joy_callback)
        rospy.wait_for_message('/joy', Joy, timeout=0.5)

    def joy_callback(self, msg):
        self.joy_msg = msg

    def run(self, solution):
        pass


class JoyCommands:
    def __init__(self, action_manager: ActionManager):
        self.action_manager = action_manager
        self.base_weight = 15.
        self.base_rot_weight = 0.5
        self.com_height_w = 0.02

        self.joy_msg = None

        self.final_base_xy = self.action_manager.task_interface.getTask('final_base_xy')
        self.com_height = self.action_manager.task_interface.getTask('com_height')
        self.base_orientation = self.action_manager.task_interface.getTask('base_orientation')

        rospy.Subscriber('/joy', Joy, self.joy_callback)
        rospy.wait_for_message('/joy', Joy, timeout=0.5)

    def joy_callback(self, msg):
        self.joy_msg = msg

    def run(self, solution):

        if self.joy_msg.buttons[4] == 1:
            # step
            if self.action_manager.contact_phases['ball_1'].getEmptyNodes() > 0:
                self.action_manager.trot()
                # self.gait_manager.trot_jumped()
                # self.gait_manager.slide()
                # self.gait_manager.crawl()
                # self.gait_manager.leap()
                # self.gait_manager.walk()
                # self.gait_manager.jump()
                # self.gait_manager.wheelie()
                # self.gait_manager.step('wheel_1')
        else:
            # stand
            if self.action_manager.contact_phases['ball_1'].getEmptyNodes() > 0:
                self.action_manager.stand()

        if np.abs(self.joy_msg.axes[0]) > 0.1 or np.abs(self.joy_msg.axes[1]) > 0.1:
            # move com on x axis w.r.t the base

            vec = np.array([self.base_weight * self.joy_msg.axes[1],
                            self.base_weight * self.joy_msg.axes[0], 0])

            rot_vec = self._rotate_vector(vec, solution['q'][[6, 3, 4, 5], 0])
            reference = np.array([[solution['q'][0, 0] + rot_vec[0], solution['q'][1, 0] + rot_vec[1], 0., 0., 0., 0., 0.]]).T

            self.final_base_xy.setRef(reference)
        else:
            # move it back in the middle
            reference = np.array([[solution['q'][0, 0], solution['q'][1, 0], 0., 0., 0., 0., 0.]]).T
            self.final_base_xy.setRef(reference)

        if np.abs(self.joy_msg.axes[3]) > 0.1:
            # rotate base around z
            d_angle = np.pi / 2 * self.joy_msg.axes[3] * self.base_rot_weight
            axis = [0, 0, 1]
            q_result = self._incremental_rotate(solution['q'][[6, 3, 4, 5], 0], d_angle, axis)

            # set orientation of the quaternion
            reference = np.array([[0., 0., 0., q_result.x, q_result.y, q_result.z, q_result.w]]).T
            self.base_orientation.setRef(reference)

        elif self.joy_msg.axes[7] == 1:
            # rotate base around y
            d_angle = np.pi / 10
            axis = [0, 1, 0]
            rot_vec = self._rotate_vector(axis, solution['q'][[6, 3, 4, 5], 0])
            q_result = self._incremental_rotate(solution['q'][[6, 3, 4, 5], 0], d_angle, rot_vec)

            # set orientation of the quaternion
            reference = np.array([[0., 0., 0., q_result.x, q_result.y, q_result.z, q_result.w]]).T
            self.base_orientation.setRef(reference)

        elif self.joy_msg.axes[7] == -1:
            # rotate base around y
            d_angle = - np.pi / 10
            axis = [0, 1, 0]
            rot_vec = self._rotate_vector(axis, solution['q'][[6, 3, 4, 5], 0])
            q_result = self._incremental_rotate(solution['q'][[6, 3, 4, 5], 0], d_angle, rot_vec)

            # set orientation of the quaternion
            reference = np.array([[0., 0., 0., q_result.x, q_result.y, q_result.z, q_result.w]]).T
            self.base_orientation.setRef(reference)

        elif self.joy_msg.axes[6] == 1:
            # rotate base around x
            d_angle = np.pi / 10
            axis = [1, 0, 0]
            rot_vec = self._rotate_vector(axis, solution['q'][[6, 3, 4, 5], 0])
            q_result = self._incremental_rotate(solution['q'][[6, 3, 4, 5], 0], d_angle, rot_vec)

            # set orientation of the quaternion
            reference = np.array([[0., 0., 0., q_result.x, q_result.y, q_result.z, q_result.w]]).T
            self.base_orientation.setRef(reference)

        elif self.joy_msg.axes[6] == -1:
            # rotate base around x
            d_angle = -np.pi / 10
            axis = [1, 0, 0]
            rot_vec = self._rotate_vector(axis, solution['q'][[6, 3, 4, 5], 0])
            q_result = self._incremental_rotate(solution['q'][[6, 3, 4, 5], 0], d_angle, rot_vec)

            # set orientation of the quaternion
            reference = np.array([[0., 0., 0., q_result.x, q_result.y, q_result.z, q_result.w]]).T
            self.base_orientation.setRef(reference)

        else:
            # set rotation of the base as the current one
            reference = np.array([[0., 0., 0., solution['q'][3, 0], solution['q'][4, 0], solution['q'][5, 0], solution['q'][6, 0]]]).T
            self.base_orientation.setRef(reference)

        if self.joy_msg.buttons[0] == 1:
            # change com height
            reference = np.array([[0., 0, solution['q'][2, 0] + self.com_height_w, 0., 0., 0., 0.]]).T
            self.com_height.setRef(reference)

        if self.joy_msg.buttons[2] == 1:
            # change com height
            reference = np.array([[0., 0, solution['q'][2, 0] - self.com_height_w, 0., 0., 0., 0.]]).T
            self.com_height.setRef(reference)

    def _incremental_rotate(self, q_initial: np.quaternion, d_angle, axis) -> np.quaternion:

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

    def _quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])

    def _conjugate_quaternion(self, q):
        q_conjugate = np.copy(q)
        q_conjugate[1:] *= -1.0
        return q_conjugate

    def _rotate_vector(self, vector, quaternion):

        # normalize the quaternion
        quaternion = quaternion / np.linalg.norm(quaternion)

        # construct a pure quaternion
        v = np.array([0, vector[0], vector[1], vector[2]])

        # rotate the vector p = q* v q
        rotated_v = self._quaternion_multiply(quaternion, self._quaternion_multiply(v, self._conjugate_quaternion(quaternion)))

        # extract the rotated vector
        rotated_vector = rotated_v[1:]

        return rotated_vector

    def _quat_to_eul(self, x_quat, y_quat, z_quat, w_quat):

        # convert quaternion to Euler angles
        roll = math.atan2(2 * (w_quat * x_quat + y_quat * z_quat), 1 - 2 * (x_quat * x_quat + y_quat * y_quat))
        pitch = math.asin(2 * (w_quat * y_quat - z_quat * x_quat))
        yaw = math.atan2(2 * (w_quat * z_quat + x_quat * y_quat), 1 - 2 * (y_quat * y_quat + z_quat * z_quat))

        roll = math.degrees(roll)
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)

        return np.array([roll, pitch, yaw])

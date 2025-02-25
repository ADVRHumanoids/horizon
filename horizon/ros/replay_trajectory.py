import numpy
import numpy as np
import casadi as cs
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
import geometry_msgs.msg
import time
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from copy import deepcopy
from horizon.ros.trajectory_viewer import TrajectoryViewer
from threading import Thread, Lock

try:
    import tf as ros_tf
except ImportError:
    from . import tf_broadcaster_simple as ros_tf
    print('will not use tf publisher')

lock = Lock()


def normalize_quaternion(q):

    def normalize(v):
        return v / np.linalg.norm(v)

    quat = normalize([q[3], q[4], q[5], q[6]])
    q[3:7] = quat[0:4]

    return q


class replay_trajectory:
    def __init__(self,
                 dt,
                 joint_list,
                 q_replay,
                 frame_force_mapping=None,
                 force_reference_frame=cas_kin_dyn.CasadiKinDyn.LOCAL,
                 kindyn=None,
                 fixed_joint_map=None,
                 trajectory_markers=None,
                 trajectory_markers_opts=None,
                 future_trajectory_markers=None,
                 future_trajectory_markers_opts=None):
        """
        Contructor
        Args:
            dt: time of replaying trajectory
            joint_list: list of joints names
            q_replay: joints position to replay
            frame_force_mapping: map between forces and frames where the force is acting
            force_reference_frame: frame w.r.t. the force is expressed. If LOCAL_WORLD_ALIGNED then forces are rotated in LOCAL frame before being published
            kindyn: needed if forces are in LOCAL_WORLD_ALIGNED
        """

        if trajectory_markers is None:
            trajectory_markers = []

        if future_trajectory_markers is None:
            future_trajectory_markers = []

        if frame_force_mapping is None:
            frame_force_mapping = {}

        if future_trajectory_markers_opts is None:
            future_trajectory_markers_opts = {}

        if fixed_joint_map is None:
            fixed_joint_map = {}

        self.dt = dt
        self.joints_1dof = [j for j in joint_list if kindyn.joint_nq(j) == 1]
        self.joints_floating = [j for j in joint_list if kindyn.joint_nq(j) == 7]
        self.iq_1dof = [kindyn.joint_iq(j) for j in self.joints_1dof]
        self.iq_floating = [kindyn.joint_iq(j) for j in self.joints_floating]
        self.parent_child_floating = [(kindyn.parentLink(j), kindyn.childLink(j)) for j in self.joints_floating]

        self.q_replay = q_replay
        self.__sleep = 0.
        self.force_pub = []
        self.frame_force_mapping = {}
        self.slow_down_rate = 1.
        self.frame_fk = dict()
        self.fixed_joint_map = fixed_joint_map

        if frame_force_mapping is not None:
            self.frame_force_mapping = deepcopy(frame_force_mapping)

        if trajectory_markers_opts is None:
            trajectory_markers_opts = {}

        self.tv = dict()
        self.future_tv = dict()

        for frame in trajectory_markers:
            self.tv[frame] = TrajectoryViewer(frame, opts=trajectory_markers_opts)

        for frame in future_trajectory_markers:
            if frame not in self.frame_fk:
                FK = kindyn.fk(frame)
                self.frame_fk[frame] = FK
            self.future_tv[frame] = TrajectoryViewer(frame, opts=future_trajectory_markers_opts)

        # WE CHECK IF WE HAVE TO ROTATE CONTACT FORCES:
        if force_reference_frame is cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED:
            if kindyn is None:
                raise Exception('kindyn input can not be None if force_reference_frame is LOCAL_WORLD_ALIGNED!')
            for frame in self.frame_force_mapping.keys(): # WE LOOP ON FRAMES
                if frame not in self.frame_fk:
                    FK = kindyn.fk(frame)
                    self.frame_fk[frame] = FK

                # rotate frame
                # w_all = self.frame_force_mapping[frame]
                # for k in range(0, w_all.shape[1]):
                #     w_R_f = FK(q=self.q_replay[:, k])['ee_rot']
                #     w = w_all[:, k].reshape(-1, 1)
                #     if w.shape[0] == 3:
                #         self.frame_force_mapping[frame][:, k] = np.dot(w_R_f.T,  w).T
                #     else:
                #         A = np.zeros((6, 6))
                #         A[0:3, 0:3] = A[3:6, 3:6] = w_R_f.T
                #         self.frame_force_mapping[frame][:, k] = np.dot(A,  w).T

        try:
            rospy.init_node('joint_state_publisher')
        except rospy.exceptions.ROSException as e:
            pass
        self.pub = rospy.Publisher('joint_states', JointState, queue_size=10)
        self.br = ros_tf.TransformBroadcaster()

        if self.frame_force_mapping:
            for key in self.frame_force_mapping:
                self.force_pub.append(rospy.Publisher(key+'_forces', geometry_msgs.msg.WrenchStamped, queue_size=10))

    def publish_past_trajectory_marker(self, trajectory_marker_action=None):

        for elem in self.tv.values():
            elem.publish_sphere(action=trajectory_marker_action)

    def publish_future_trajectory_marker(self, solution):

        for frame_name, frame_publisher in self.future_tv.items():
            traj = self.frame_fk[frame_name](q=solution)['ee_pos'].toarray()
            frame_publisher.publish_line(traj)

    def publishContactForces(self, time, qk, k):
        i = 0
        for frame in self.frame_force_mapping:

            f_msg = geometry_msgs.msg.WrenchStamped()
            f_msg.header.stamp = time
            f_msg.header.frame_id = frame

            f = self.frame_force_mapping[frame][:, k]

            w_R_f = self.frame_fk[frame](q=qk)['ee_rot'].toarray()

            if f.shape[0] == 3:
                f = np.dot(w_R_f.T,  f).T
            else:
                A = np.zeros((6, 6))
                A[0:3, 0:3] = A[3:6, 3:6] = w_R_f.T
                f = np.dot(A,  f).T

            f_msg.wrench.force.x = f[0]
            f_msg.wrench.force.y = f[1]
            f_msg.wrench.force.z = f[2]

            if f.shape[0] == 3:
                f_msg.wrench.torque.x = 0.
                f_msg.wrench.torque.y = 0.
                f_msg.wrench.torque.z = 0.
            else:
                f_msg.wrench.torque.x = f[3]
                f_msg.wrench.torque.y = f[4]
                f_msg.wrench.torque.z = f[5]

            self.force_pub[i].publish(f_msg)
            i += 1

    def sleep(self, secs):
        '''
        Set sleep time between trajectory sequences
        Args:
            secs: time to sleep in seconds
        '''
        self.__sleep = secs

    def setSlowDownFactor(self, slow_down_factor):
        '''
        Set a slow down factor for the replay of the trajectory
        Args:
             slow_down_factor: fator to slow down
        '''
        self.slow_down_rate = 1./slow_down_factor

    def publish_joints(self, qk, skip_tf=False, prefix=''):

        joint_state_pub = JointState()
        joint_state_pub.header = Header()
        joint_state_pub.name = self.joints_1dof + list(self.fixed_joint_map.keys())
        t = rospy.Time.now()
        br = self.br
        nq = len(qk)

        for iq, (parent, child) in zip(self.iq_floating, self.parent_child_floating):

            if skip_tf:
                break

            q = normalize_quaternion(qk[iq:iq+7])

            m = geometry_msgs.msg.TransformStamped()
            m.header.frame_id = prefix + '/' + parent
            m.child_frame_id = prefix + '/' + child
            m.transform.translation.x = q[0]
            m.transform.translation.y = q[1]
            m.transform.translation.z = q[2]
            m.transform.rotation.x = q[3]
            m.transform.rotation.y = q[4]
            m.transform.rotation.z = q[5]
            m.transform.rotation.w = q[6]

            br.sendTransform((m.transform.translation.x, m.transform.translation.y, m.transform.translation.z),
                                (m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z,
                                m.transform.rotation.w),
                                t, m.child_frame_id, m.header.frame_id)


        joint_state_pub.header.stamp = t
        joint_state_pub.position = qk[self.iq_1dof].tolist() + list(self.fixed_joint_map.values())
        joint_state_pub.velocity = []
        joint_state_pub.effort = []

        # action = Marker.ADD
        # for elem in self.tv.values():
        #     elem.publish_once(action=trajectory_marker_action)

        self.pub.publish(joint_state_pub)


    def replay(self, prefix=''):

        rate = rospy.Rate(self.slow_down_rate / self.dt)
        nq = np.shape(self.q_replay)[0]
        ns = np.shape(self.q_replay)[1]

        # for elem in self.tv.values():
        #     dt_markers = 0.05
        #     multiplier = self.dt / dt_markers
        #     n_markers_max = int((ns-1) * multiplier)
        #     p = Thread(target=self.publish_frame_trajectories, args=(self.tv['ball_1'], 1/dt_markers, n_markers_max))
        #     p.start()

        while not rospy.is_shutdown():

            k = 0
            for qk in self.q_replay.T:

                t = rospy.Time.now()

                # publish trajectory of frames with markers

                if k == ns - 1:
                    action = Marker.DELETEALL
                else:
                    action = Marker.ADD

                self.publish_joints(qk, prefix=prefix)


                if self.frame_force_mapping:
                    if k != ns-1:
                        self.publishContactForces(t, qk, k)


                rate.sleep()
                k += 1
            if self.__sleep > 0.:
                time.sleep(self.__sleep)
                print('replaying traj ...')

    # def publish_frame_trajectories(self, pub, rate, markers_max=500):
    #
    #     k = 0
    #     rospy_rate = rospy.Rate(rate)
    #     while True:
    #         if k == markers_max:
    #             action = Marker.DELETEALL
    #             k = 0
    #             print('reset')
    #         else:
    #             action = Marker.ADD
    #
    #         k += 1
    #         pub.publish_once(action=action, markers_max=markers_max)
    #
    #         rospy_rate.sleep()


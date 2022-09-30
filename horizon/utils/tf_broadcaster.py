#!/usr/bin/env python3
import rospy
import tf
from multiprocessing import Process


class TFBroadcaster:

    @staticmethod
    def _internal_publish(name, pose_vec, rate):
        br = tf.TransformBroadcaster()
        rospy.init_node(f'{name}_broadcaster')
        ros_rate = rospy.Rate(rate)
        while not rospy.is_shutdown():

            br.sendTransform(pose_vec[:3],
                             pose_vec[3:],
                             rospy.Time.now(),
                             f"{name}_tf",
                             "world")
            ros_rate.sleep()

    @classmethod
    def publish(cls, name, pose_vec, rate=10.0):
        p = Process(target=cls._internal_publish, args=(name, pose_vec, rate))
        p.start()

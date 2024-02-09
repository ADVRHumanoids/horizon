#! /usr/bin/env python
import copy

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TwistStamped, Pose, Point, Vector3, Quaternion
from std_msgs.msg import Header, ColorRGBA, String
from sensor_msgs.msg import JointState
import subprocess
import time
from numpy_ros import to_numpy, to_message


class TrajectoryViewer:

    def __init__(self, frame, parent=None, color=None, pub_name='markers_array/'):

        if parent is None:
            self.parent = frame
        else:
            self.parent = parent

        if color is None:
            self.color = [0.0, 2.0, 0.0, 0.8]
        else:
            self.color = color

        self.frame = frame
        self.count = 0
        self.sphere_publisher = rospy.Publisher(pub_name + self.frame, MarkerArray, queue_size=100)
        self.line_publisher = rospy.Publisher(pub_name + self.frame, MarkerArray, queue_size=100)

        # rospy.Subscriber("/joint_states", JointState, self.event_in_cb)
        self.a = [1, 1, 1]
        self.sphere_array = MarkerArray()
        self.line_array = MarkerArray()
        rospy.sleep(0.5)

    # def event_in_cb(self, msg):
    #     self.waypoints = msg
    #     self.a = [1, 1, 1]
    #
    #     self.publish_once()

    def publish_sphere(self, action=None, markers_max=1000, marker_lifetime=10):

        if action is None:
            action = Marker.ADD

        if action == Marker.DELETEALL:
            self.sphere_array.markers.clear()
            self.count = 0

        self.markers_max = markers_max

        marker = Marker(
                        type=Marker.SPHERE,
                        action=action,
                        lifetime=rospy.Duration(marker_lifetime),
                        pose=Pose(Point(self.a[0] / 10 ** 5, self.a[1] / 10 ** 5, self.a[2] / 10 ** 5), Quaternion(0, 0, 0, 1)),
                        scale=Vector3(0.02, 0.02, 0.02),
                        header=Header(frame_id=self.parent),
                        color=ColorRGBA(*self.color)
                        )

        # self.marker.id = self.count
        marker.header.stamp = rospy.Time.now()

        if (self.count > self.markers_max):
            if self.sphere_array.markers:
                self.sphere_array.markers.pop(0)

        id = 0
        for m in self.sphere_array.markers:
            m.id = id
            id += 1

        self.count += 1

        self.sphere_array.markers.append(marker)
        self.sphere_publisher.publish(self.sphere_array)

    def publish_line(self, points):

        self.line_array.markers.clear()

        marker = Marker(type=Marker.LINE_STRIP,
                             action=Marker.ADD,
                             scale=Vector3(0.02, 0.02, 0.02),
                             header=Header(frame_id=self.parent),
                             color=ColorRGBA(*self.color))

        marker.pose.orientation.w = 1

        for col in range(points.shape[1]):

            point = to_message(Point, points[:3, col])
            marker.points.append(point)

        self.line_array.markers.append(marker)
        self.line_publisher.publish(self.line_array)


if __name__ == '__main__':
    # rospy.init_node("trajectory_interactive_markers_node", anonymous=True)
    # tv = TrajectoryViewer()
    #
    # rate = rospy.Rate(1 / 0.01)
    # while not rospy.is_shutdown():
    #     tv.publish_once('ball_1')
    #     rate.sleep()
    # #
    # # rospy.sleep(0.5)
    # # rospy.spin()
    import numpy as np
    rospy.init_node("cazzo", anonymous=True)
    tv = TrajectoryViewer("com")

    vec = np.array([[1, 1, 1, 0, 0, 0, 1],
                    [2, 2, 1, 0, 0, 0, 1],
                    [3, 1, 3, 0, 0, 0, 1]])

    rate = rospy.Rate(1 / 0.01)
    while not rospy.is_shutdown():
        tv.publish_once_pose(vec)
        rate.sleep()
    #
    # rospy.sleep(0.5)
    # rospy.spin()
#! /usr/bin/env python
import copy
import random
import rclpy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TwistStamped, Pose, Point, Vector3, Quaternion
from std_msgs.msg import Header, ColorRGBA, String
from sensor_msgs.msg import JointState
import subprocess
import time
# from numpy_ros import to_numpy, to_message


class TrajectoryViewer:

    def __init__(self, frame, opts=None, node=None):

        self.__init_opts(opts)

        self.frame = frame
        self.count = 0
        if node is None:
            try:
                self.node = rclpy.create_node('trajectory_viewer')
                rclpy.get_global_executor().add_node(self.node)
            except rclpy.exceptions.ROSException as e:
                pass
        else:
            self.node = node
        self.sphere_publisher = self.node.create_publisher(MarkerArray, self.prefix + self.frame, 100)
        self.line_publisher = self.node.create_publisher(MarkerArray, self.prefix + self.frame, 100)

        # rclpy.Subscriber("/joint_states", JointState, self.event_in_cb)
        self.a = [1, 1, 1]
        self.sphere_array = MarkerArray()
        self.line_array = MarkerArray()
        time.sleep(0.5)

    # def event_in_cb(self, msg):
    #     self.waypoints = msg
    #     self.a = [1, 1, 1]
    #
    #     self.publish_once()
    def __init_opts(self, opts):
        if 'prefix' in opts:
            self.prefix = opts['prefix']
        else:
            self.prefix = "future_marker_array/"

        if 'parent' in opts:
            self.parent = opts['parent']
        else:
            self.parent = 'world'

        if 'colors' in opts:
            self.color = opts['colors']
        else:
            self.color = [random.uniform(0, 1),
                          random.uniform(0, 1),
                          random.uniform(0, 1),
                          1.]

        if 'scale' in opts:
            self.scale = opts['scale']
        else:
            self.scale = Vector3()
            self.scale.x = 0.01
            self.scale.y = 0.01
            self.scale.z = 0.01

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
                        lifetime=rclpy.Duration(marker_lifetime),
                        pose=Pose(Point(self.a[0] / 10 ** 5, self.a[1] / 10 ** 5, self.a[2] / 10 ** 5), Quaternion(0, 0, 0, 1)),
                        scale=self.scale,
                        header=Header(frame_id=self.parent),
                        color=ColorRGBA(*self.color)
                        )

        # self.marker.id = self.count
        marker.header.stamp = rclpy.Time.now()

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
                        scale=self.scale,
                        header=Header(frame_id=self.parent),
                        color=ColorRGBA(*self.color))

        marker.pose.orientation.w = 1

        for col in range(points.shape[1]):

            point = Point(points[:3, col])
            marker.points.append(point)

        self.line_array.markers.append(marker)
        self.line_publisher.publish(self.line_array)


if __name__ == '__main__':
    # rclpy.init_node("trajectory_interactive_markers_node", anonymous=True)
    # tv = TrajectoryViewer()
    #
    # rate = rclpy.Rate(1 / 0.01)
    # while not rclpy.is_shutdown():
    #     tv.publish_once('ball_1')
    #     rate.sleep()
    # #
    # # rclpy.sleep(0.5)
    # # rclpy.spin()
    import numpy as np
    rclpy.init_node("something", anonymous=True)
    tv = TrajectoryViewer("com")

    vec = np.array([[1, 1, 1, 0, 0, 0, 1],
                    [2, 2, 1, 0, 0, 0, 1],
                    [3, 1, 3, 0, 0, 0, 1]])

    rate = rclpy.Rate(1 / 0.01)
    while not rclpy.is_shutdown():
        tv.publish_once_pose(vec)
        rate.sleep()
    #
    # rclpy.sleep(0.5)
    # rclpy.spin()
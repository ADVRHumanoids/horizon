#! /usr/bin/env python
import random
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion
from std_msgs.msg import Header, ColorRGBA

from horizon.ros import ros2


class TrajectoryViewer:

    def __init__(self, frame, opts=None):

        self.__init_opts(opts)

        self.frame = frame
        self.count = 0
        self.sphere_publisher = ros2.create_publisher(MarkerArray, self.prefix + self.frame, 100)
        self.line_publisher = ros2.create_publisher(MarkerArray, self.prefix + self.frame, 100)

        self.a = [1, 1, 1]
        self.sphere_array = MarkerArray()
        self.line_array = MarkerArray()
        ros2.sleep(0.5)

    @staticmethod
    def _to_vector3(value):
        if isinstance(value, Vector3):
            return value

        if isinstance(value, np.ndarray):
            value = value.tolist()

        if isinstance(value, (list, tuple)) and len(value) == 3:
            msg = Vector3()
            msg.x = float(value[0])
            msg.y = float(value[1])
            msg.z = float(value[2])
            return msg

        raise ValueError("scale must be a Vector3 or a sequence of 3 numeric values")

    def to_point_message(self, arr):

        msg = Point()
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()

        msg.x, msg.y, msg.z = arr
        return msg

    # def event_in_cb(self, msg):
    #     self.waypoints = msg
    #     self.a = [1, 1, 1]
    #
    #     self.publish_once()
    def __init_opts(self, opts):
        if opts is None:
            opts = {}

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
            self.scale = self._to_vector3(opts['scale'])
        else:
            self.scale = self._to_vector3((0.01, 0.01, 0.01))

    def publish_sphere(self, action=None, markers_max=1000, marker_lifetime=10):

        if action is None:
            action = Marker.ADD

        if action == Marker.DELETEALL:
            self.sphere_array.markers.clear()
            self.count = 0

        self.markers_max = markers_max

        pose = Pose()
        pose.position.x = self.a[0] / 10 ** 5
        pose.position.y = self.a[1] / 10 ** 5
        pose.position.z = self.a[2] / 10 ** 5
        pose.orientation.w = 1.0

        color = ColorRGBA()
        color.r = self.color[0]
        color.g = self.color[1]
        color.b = self.color[2]
        color.a = self.color[3]

        marker = Marker(
                        type=Marker.SPHERE,
                        action=action,
                        lifetime=ros2.duration(marker_lifetime),
                pose=pose,
                        scale=self.scale,
                        header=Header(frame_id=self.parent),
                color=color
                        )

        # self.marker.id = self.count
        marker.header.stamp = ros2.now()

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

        color = ColorRGBA()
        color.r = self.color[0]
        color.g = self.color[1]
        color.b = self.color[2]
        color.a = self.color[3]

        marker = Marker(type=Marker.LINE_STRIP,
                        action=Marker.ADD,
                        scale=self.scale,
                        header=Header(frame_id=self.parent),
                        color=color)

        marker.pose.orientation.w = 1

        for col in range(points.shape[1]):

            point = self.to_point_message(points[:3, col])
            marker.points.append(point)

        self.line_array.markers.append(marker)
        self.line_publisher.publish(self.line_array)


if __name__ == '__main__':
    # ros2.init_node("trajectory_interactive_markers_node")
    # tv = TrajectoryViewer()
    #
    # rate = ros2.Rate(1 / 0.01)
    # while not ros2.is_shutdown():
    #     tv.publish_once('ball_1')
    #     rate.sleep()
    # #
    # # ros2.sleep(0.5)
    ros2.init_node("something")
    tv = TrajectoryViewer("com")

    vec = np.array([[1, 1, 1, 0, 0, 0, 1],
                    [2, 2, 1, 0, 0, 0, 1],
                    [3, 1, 3, 0, 0, 0, 1]])

    rate = ros2.Rate(1 / 0.01)
    while not ros2.is_shutdown():
        tv.publish_once_pose(vec)
        rate.sleep()
    #
    # ros2.sleep(0.5)

#! /usr/bin/env python

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TwistStamped, Pose, Point, Vector3, Quaternion
from std_msgs.msg import Header, ColorRGBA, String
from sensor_msgs.msg import JointState

class TrajectoryViewer:

    def __init__(self, frame, color=None):

        if color is None:
            self.color = [0.0, 2.0, 0.0, 0.8]
        else:
            self.color = color

        self.frame = frame
        self.count = 0
        self.marker_publisher = rospy.Publisher('visualization_marker_array' + self.frame, MarkerArray, queue_size=100)
        # rospy.Subscriber("/joint_states", JointState, self.event_in_cb)
        self.a = [1, 1, 1]
        self.marker_array = MarkerArray()
        rospy.sleep(0.5)
    def event_in_cb(self, msg):
        self.waypoints = msg
        self.a = [1, 1, 1]

        self.publish_marker_array()

    def publish_marker_array(self, markers_max=100, action=None):

        if action is None:
            action = Marker.ADD

        self.markers_max = markers_max

        self.marker = Marker(
                            type=Marker.SPHERE,
                            action=action,
                            lifetime=rospy.Duration(10),
                            pose=Pose(Point(self.a[0]/10**5, self.a[1]/10**5,self.a[2]/10**5), Quaternion(0, 0, 0, 1)),
                            scale=Vector3(0.02, 0.02, 0.02),
                            header=Header(frame_id=self.frame),
                            color=ColorRGBA(*self.color))

        # self.marker.id = self.count
        self.marker.header.stamp = rospy.Time.now()

        if (self.count > self.markers_max):
            self.marker_array.markers.pop(0)

        id = 0
        for m in self.marker_array.markers:
            m.id = id
            id += 1

        self.count += 1

        self.marker_array.markers.append(self.marker)
        self.marker_publisher.publish(self.marker_array)


if __name__ == '__main__':
    rospy.init_node("trajectory_interactive_markers_node", anonymous=True)
    tv = TrajectoryViewer()

    rate = rospy.Rate(1/0.01)
    while not rospy.is_shutdown():
        tv.publish_marker_array('ball_1')
        rate.sleep()
    #
    # rospy.sleep(0.5)
    # rospy.spin()
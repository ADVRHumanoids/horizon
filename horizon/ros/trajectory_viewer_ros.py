import argparse
from visualization_msgs.msg import Marker

from horizon.ros import ros2
from horizon.ros.trajectory_viewer import TrajectoryViewer

def main(args):

    ros2.init_node("trajectory_viewer")
    frame = args.frame
    rate = float(args.rate)
    markers_max = int(args.markers_max)
    pub = TrajectoryViewer(frame)
    k = 0

    rospy_rate = ros2.Rate(rate)
    while not ros2.is_shutdown():
        if k == markers_max:
            action = Marker.DELETEALL
            k = 0
            print('reset')
        else:
            action = Marker.ADD

        k += 1
        pub.publish_sphere(action=action, markers_max=markers_max)
        rospy_rate.sleep()

if __name__ == '__main__':


    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--frame', '-f', default='world', help='frame')
    parser.add_argument('--rate', '-r', default=10.0, help='rate')
    parser.add_argument('--markers-max', '-m', default=1000, help='maximum markers before reset')

    args = parser.parse_args()
    main(args)

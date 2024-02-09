import argparse
import rospy
from visualization_msgs.msg import Marker

def main(args):

    frame = args.frame
    rate = args.rate
    k = 0

    rospy_rate = rospy.Rate(rate)
    while True:
        if k == markers_max:
            action = Marker.DELETEALL
            k = 0
            print('reset')
        else:
            action = Marker.ADD

        k += 1
        pub.publish_once(action=action, markers_max=markers_max)
        rospy_rate.sleep()

if __name__ == '__main__':


    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--frame', '-f', help='frame')
    parser.add_argument('--rate', '-r', help='rate')

    args = parser.parse_args()
    main(args)
from tf2_msgs.msg import TFMessage

class TransformBroadcaster:
    def __init__(self, node) -> None:
        self.pub = node.create_publisher(TFMessage, '/tf', 10)

    def sendTransform(self, msg):
        tfmsg = TFMessage()
        tfmsg.transforms.append(msg)
        self.pub.publish(tfmsg)
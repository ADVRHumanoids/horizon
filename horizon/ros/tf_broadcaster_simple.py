from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster as Tf2TransformBroadcaster

class TransformBroadcaster:
    def __init__(self) -> None:
        from horizon.ros import ros2
        self._br = Tf2TransformBroadcaster(ros2.get_node())

    def sendTransform(self, pos, rot, time, child_frame_id, frame_id):
        transf = TransformStamped()
        transf.header.stamp = time
        transf.header.frame_id = frame_id
        transf.child_frame_id = child_frame_id
        transf.transform.translation.x = pos[0]
        transf.transform.translation.y = pos[1]
        transf.transform.translation.z = pos[2]
        transf.transform.rotation.x = rot[0]
        transf.transform.rotation.y = rot[1]
        transf.transform.rotation.z = rot[2]
        transf.transform.rotation.w = rot[3]
        self._br.sendTransform(transf)

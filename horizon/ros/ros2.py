import time

import rclpy
from rclpy.parameter import Parameter


_node = None
_node_name = "horizon"
_params = {}


def init_node(name=_node_name, args=None):
    global _node, _node_name

    if _node is not None:
        return _node

    _node_name = name
    try:
        rclpy.init(args=args)
    except RuntimeError:
        pass

    _node = rclpy.create_node(name)
    return _node


def get_node():
    return init_node(_node_name)


def create_publisher(msg_type, topic, queue_size=10):
    return get_node().create_publisher(msg_type, topic, queue_size)


def create_subscription(msg_type, topic, callback, queue_size=10):
    return get_node().create_subscription(msg_type, topic, callback, queue_size)


def create_service(srv_type, topic, callback):
    def service_callback(request, response):
        result = callback(request)
        if isinstance(result, dict):
            for key, value in result.items():
                setattr(response, key, value)
            return response
        if result is not None:
            return result
        return response

    return get_node().create_service(srv_type, topic, service_callback)


def spin_once(timeout_sec=0.0):
    if rclpy.ok():
        rclpy.spin_once(get_node(), timeout_sec=timeout_sec)


def is_shutdown():
    return not rclpy.ok()


def now():
    return get_node().get_clock().now().to_msg()


def duration(seconds=0.0):
    sec = int(seconds)
    nanosec = int((seconds - sec) * 1e9)
    from builtin_interfaces.msg import Duration
    return Duration(sec=sec, nanosec=nanosec)


def sleep(seconds):
    spin_once(0.0)
    time.sleep(seconds)
    spin_once(0.0)


class Rate:
    def __init__(self, hz):
        self._period = 1.0 / float(hz)
        self._next = time.monotonic() + self._period

    def sleep(self):
        spin_once(0.0)
        delay = self._next - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        now_monotonic = time.monotonic()
        while self._next <= now_monotonic:
            self._next += self._period


def _parameter_name(name):
    return name.strip("/~").replace("/", ".")


def set_param(name, value):
    node = get_node()
    ros2_name = _parameter_name(name)
    _params[name] = value

    if not node.has_parameter(ros2_name):
        node.declare_parameter(ros2_name, value)
    else:
        node.set_parameters([Parameter(ros2_name, value=value)])

    return value


def get_param(name, default=None):
    node = get_node()
    ros2_name = _parameter_name(name)

    if node.has_parameter(ros2_name):
        return node.get_parameter(ros2_name).value

    if name in _params:
        return _params[name]

    if default is not None:
        return set_param(name, default)

    raise KeyError(f"ROS parameter '{name}' is not set")


def has_param(name):
    return get_node().has_parameter(_parameter_name(name)) or name in _params


def loginfo(message):
    get_node().get_logger().info(str(message))


def logwarn(message):
    get_node().get_logger().warning(str(message))


def logerr(message):
    get_node().get_logger().error(str(message))


def shutdown():
    global _node

    if _node is not None:
        _node.destroy_node()
        _node = None

    if rclpy.ok():
        rclpy.shutdown()

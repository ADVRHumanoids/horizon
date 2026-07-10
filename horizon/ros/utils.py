import subprocess

def roslaunch(package, launch, *args):
    """
    Run a ROS 2 launch file in a separate process.
    Args:
        package: where the launch file is located
        launch: file
    """
    cmd = ['ros2', 'launch', package, launch, *args]
    subprocess.Popen(cmd)

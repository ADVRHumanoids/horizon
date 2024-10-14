from horizon.rhc import taskInterface
from rosbot_param_server import rosbot_param_server_py
import rospy

class TaskServerClass:
    def __init__(self, ti : taskInterface.TaskInterface):
        self.__ti = ti

        self.__task_map = {task.getName(): task for task in self.__ti.getTasks()}

        self.__pm = rosbot_param_server_py.ParameterManager()

        for name, task in self.__task_map.items():
            self.__pm.createParameter(name, task.setWeight, task.getWeight)

    # def task_weight_callback(self):


if __name__ == '__main__':

    from horizon.problem import Problem
    from horizon.rhc.model_description import FullModelInverseDynamics
    from horizon.rhc.taskInterface import TaskInterface
    from casadi_kin_dyn import py3casadi_kin_dyn as casadi_kin_dyn
    import casadi as cs
    import numpy as np
    import time
    import rospkg
    import subprocess

    rospy.init_node('test_task_server_class')

    rosbot_param_server_py.init('test_task_server_class', [])
    param_manager = rosbot_param_server_py.ParameterManager()

    '''
    Load urdf 
    '''
    kyon_urdf_folder = rospkg.RosPack().get_path('kyon_urdf')
    urdf = subprocess.check_output(["xacro", kyon_urdf_folder + "/urdf/kyon.urdf.xacro",
                                    "sensors:=false",
                                    f"upper_body:=false",
                                    f"payload:=false",
                                    ])
    urdf = urdf.decode('utf-8')

    ns = 30
    T = 1.5
    dt = T / ns

    prb = Problem(ns, receding=True, casadi_type=cs.SX)
    prb.setDt(dt)

    kin_dyn = casadi_kin_dyn.CasadiKinDyn(urdf)

    q_init = {'hip_roll_1': 0.0,
              'hip_pitch_1': 0.7,
              'knee_pitch_1': -1.4,
              'hip_roll_2': 0.0,
              'hip_pitch_2': -0.7,
              'knee_pitch_2': 1.4,
              'hip_roll_3': 0.0,
              'hip_pitch_3': 0.7,
              'knee_pitch_3': -1.4,
              'hip_roll_4': 0.0,
              'hip_pitch_4': -0.7,
              'knee_pitch_4': 1.4,
              }

    base_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    FK = kin_dyn.fk('contact_1')
    init = base_pose.tolist() + list(q_init.values())
    init_pos_foot = FK(q=kin_dyn.mapToQ(q_init))['ee_pos']
    base_pose[2] = -init_pos_foot[2]



    model = FullModelInverseDynamics(problem=prb,
                                     kd=kin_dyn,
                                     q_init=q_init,
                                     base_init=base_pose
                                     )


    ti = TaskInterface(prb=prb, model=model)
    ti.setTaskFromYaml(rospkg.RosPack().get_path('kyon_controller') + '/config/kyon_config.yaml')

    # finalize taskInterface and solve bootstrap problem
    ti.finalize()

    for task in ti.getTasks():
        if isinstance(task.weight_param, dict):
            for name in task.weight_param.keys():
                param_manager.createParameter(name, task.setWeight, task.getWeight()[name][0, 0])
        else:
            param_manager.createParameter(task.weight_param.getName(), task.setWeight, task.getWeight()[0, 0])

    ti.bootstrap()
    ti.load_initial_guess()
    solution = ti.solution

    rate = rospy.Rate(1 / dt)
    while not rospy.is_shutdown():
        # set initial state and initial guess
        shift_num = -1

        x_opt = solution['x_opt']
        xig = np.roll(x_opt, shift_num, axis=1)
        for i in range(abs(shift_num)):
            xig[:, -1 - i] = x_opt[:, -1]

        prb.getState().setInitialGuess(xig)
        prb.setInitialState(x0=xig[:, 0])

        # shift phases of phase manager
        tic = time.time()
        ti.rti()

        solution = ti.solution

        rate.sleep()

        rosbot_param_server_py.ros_update()

    rosbot_param_server_py.shutdown()


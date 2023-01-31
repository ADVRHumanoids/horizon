from horizon.utils import plotter
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.ros import replay_trajectory
from horizon.solvers.solver import Solver
from horizon.rhc.taskInterface import TaskInterface
from horizon.rhc.model_description import *
import numpy as np
import rospkg
import casadi as cs
import rospy
import subprocess

"""
This application is basically what /playground/mirror/mirror_walk_am.py does, but using the tasks dicts, and without using the ActionManager.
There should be:
- a config file with all the problem setting, constraints/costs
- an execute file to set parameters, use the actionManager...
"""
urdf_path = rospkg.RosPack().get_path('mirror_urdf') + '/urdf/mirror.urdf'
urdf = open(urdf_path, 'r').read()
kd_frame = pycasadi_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
kd = pycasadi_kin_dyn.CasadiKinDyn(urdf)
rospy.set_param('/robot_description', urdf)
subprocess.Popen(['rosrun', 'robot_state_publisher', 'robot_state_publisher'])

ns = 50
tf = 8.0  # 10s
dt = tf / ns
problem_opts = {'ns': ns, 'tf': tf, 'dt': dt}

q_init = {}

for i in range(3):
    q_init[f'arm_{i + 1}_joint_2'] = -1.9
    q_init[f'arm_{i + 1}_joint_3'] = -2.30
    q_init[f'arm_{i + 1}_joint_5'] = -0.4

base_init = np.array([0, 0, 0.72, 0, 0, 0, 1])

# todo: this should not be in initialization
#  I should add contacts after the initialization, as forceTasks, no?

# set up model
prb = Problem(ns, receding=True)  # logging_level=logging.DEBUG
prb.setDt(dt)

model = FullModelInverseDynamics(problem=prb,
                                 kd=kd,
                                 q_init=q_init,
                                 base_init=base_init)

contacts = [f'arm_{i + 1}_TCP' for i in range(3)]

ti = TaskInterface(prb, model)

# register my plugin 'Contact'
# todo is this not required anymore?
# ti.loadPlugins(['horizon.rhc.plugins.contactTaskMirror'])

ptgt_final = base_init.copy()

task_base_x = {'type': 'Cartesian',
               'distal_link': 'base_link',
               'name': 'final_base_x',
               'indices': [0],
               'nodes': [ns]}
#
task_base_y = {'type': 'Cartesian',
               'distal_link': 'base_link',
               'name': 'final_base_y',
               'indices': [1],
               'nodes': [ns],
               'fun_type': 'residual',
               'weight': 1e3}

ti.setTaskFromDict(task_base_x)
ti.setTaskFromDict(task_base_y)

for contact in contacts:
    # subtask_force = {'type': 'Wrench',
    #                  'name': f'interaction_{contact}',
    #                  'frame': contact,
    #                  'fn_min': 10.0,
    #                  'enable_cop': False,
    #                  'dimensions': [0.2, 0.2]}

    subtask_force = {'type': 'VertexForce',
                     'name': f'interaction_{contact}',
                     'frame': contact,
                     'fn_min': 10.0,
                     'vertex_frames': [contact]
                     }

    subtask_cartesian = {'type': 'Cartesian',
                         'name': f'zero_velocity_{contact}',
                         'distal_link': contact,
                         'indices': [0, 1, 2, 3, 4, 5],
                         'cartesian_type': 'velocity',
                         'nodes': 'all'
                         }

    contact_task = {'type': 'Contact',
                    'subtask': [subtask_force, subtask_cartesian],
                    'name': 'contact_' + contact}

    ti.setTaskFromDict(contact_task)

task_base_x = ti.getTask('final_base_x')
task_base_x.setRef([0, 0, 0, ptgt_final[0], 0, 0, 1])

task_base_y = ti.getTask('final_base_y')
task_base_y.setRef([0, 0, 0, 0, ptgt_final[1], 0, 1])

# todo: next section to wrap up like the lines above

q = ti.prb.getVariables('q')
v = ti.prb.getVariables('v')
a = ti.prb.getVariables('a')
forces = [ti.prb.getVariables('f_' + c) for c in contacts]

q0 = model.q0
v0 = model.v0
f0 = np.array([0, 0, 300])

# final velocity
v.setBounds(v0, v0, nodes=ns)
# regularization costs

# base rotation
ti.prb.createResidual("min_rot", 1e-4 * (q[3:5] - q0[3:5]))

# joint posture
ti.prb.createResidual("min_q", 1e-1 * (q[7:] - q0[7:]))

# joint velocity
ti.prb.createResidual("min_v", 1e-2 * v)

# final posture
# todo: incredible, this is the problem. if it's FinalResidual, everything goes tho whores
ti.prb.createResidual("min_qf", 1e1 * (q[7:] - q0[7:]))

# regularize input
ti.prb.createIntermediateResidual("min_q_ddot", 1e0 * a)

# regularize forces
for f in forces:
    ti.prb.createIntermediateResidual(f"min_{f.getName()}", 1e-3 * (f - f0))

# costs and constraints implementing a gait schedule
# com_fn = kd.centerOfMass()

# contact velocity is zero, and normal force is positive
for i, frame in enumerate(contacts):
    # fk functions and evaluated vars
    fk = kd.fk(frame)

    ee_p = fk(q=q)['ee_pos']
    ee_rot = fk(q=q)['ee_rot']

    # vertical contact frame
    rot_err = cs.sumsqr(ee_rot[2, :2])
    ti.prb.createIntermediateCost(f'{frame}_rot', 1e4 * rot_err)

# set initial condition and initial guess
q.setBounds(q0, q0, nodes=0)
v.setBounds(v0, v0, nodes=0)

q.setInitialGuess(q0)

for f in forces:
    f.setInitialGuess(f0)

# ========================= set actions =====================================
c_0 = ti.getTask('contact_arm_1_TCP')
c_1 = ti.getTask('contact_arm_2_TCP')
c_2 = ti.getTask('contact_arm_3_TCP')

step = range(10, 30)
contact_c_0 = [c_n for c_n in list(range(ns + 1)) if c_n not in step]
c_0.setNodes(contact_c_0)
c_1.setNodes(range(ns + 1))
c_2.setNodes(range(ns + 1))

# create solver and solve initial seed
# print('===========executing ...========================')

opts = {'ipopt.tol': 0.001,
        'ipopt.constr_viol_tol': 1e-3,
        'ipopt.max_iter': 1000,
        'error_on_fail': True,
        'ilqr.max_iter': 200,
        'ilqr.alpha_min': 0.01,
        'ilqr.use_filter': False,
        'ilqr.hxx_reg': 0.0,
        'ilqr.integrator': 'RK4',
        'ilqr.merit_der_threshold': 1e-6,
        'ilqr.step_length_threshold': 1e-9,
        'ilqr.line_search_accept_ratio': 1e-4,
        'ilqr.kkt_decomp_type': 'qr',
        'ilqr.constr_decomp_type': 'qr',
        'ilqr.verbose': True,
        'ipopt.linear_solver': 'ma57',
        }

opts['type'] = 'ipopt'

opts_rti = opts.copy()
opts_rti['ilqr.enable_line_search'] = False
opts_rti['ilqr.max_iter'] = 4

ti.setSolverOptions(opts)
ti.finalize()

# print('VARIABLES:')
# for var_name, obj in ti.prb.getVariables().items():
#     print(var_name, ':', type(obj))
#     print(obj)
#     print(obj.getNodes().tolist())
#     print(obj.getBounds())

# print('CONSTRAINTS:')
# for cnsrt, obj in ti.prb.getConstraints().items():
#     print(cnsrt,':', type(obj))
#     print(obj.getFunction())
#     print(obj._fun_impl)
#     print(obj.getNodes())
#     print(obj.getBounds())
#
# print('COSTS:')
# for cnsrt, obj in ti.prb.getCosts().items():
#     print(cnsrt,':', obj.getNodes(), type(obj))
#

ti.bootstrap()
solution = ti.solution
# solver_bs = Solver.make_solver(solver_type, ti.prb, opts)
# solver_rti = Solver.make_solver(solver_type, ti.prb, opts_rti)

# solver_bs.solve()
# solution = solver_bs.getSolutionDict()

# os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
# subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=spot'])
# rospy.loginfo("'spot' visualization started.")

## single replay
# q_sol = solution['q']
# frame_force_mapping = {contacts[i]: solution[forces[i].getName()] for i in range(3)}
# repl = replay_trajectory.replay_trajectory(dt, kd.joint_names(), q_sol, frame_force_mapping, kd_frame, kd)
# repl.sleep(1.)
# repl.replay(is_floating_base=True)
# exit()

plot_flag = True
if plot_flag:
    import matplotlib.pyplot as plt
    import matplotlib

    plt.figure()
    for contact in contacts:
        FK = kd.fk(contact)
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'feet position - plane_xy')
        plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[1, :]).flatten(), linewidth=2.5)
        plt.scatter(np.array(pos[0, 0]), np.array(pos[1, 0]))
        plt.scatter(np.array(pos[0, -1]), np.array(pos[1, -1]), marker='x')

    plt.figure()
    for contact in contacts:
        FK = kd.fk(contact)
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'feet position - plane_xz')
        plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[2, :]).flatten(), linewidth=2.5)
        plt.scatter(np.array(pos[0, 0]), np.array(pos[2, 0]))
        plt.scatter(np.array(pos[0, -1]), np.array(pos[2, -1]), marker='x')

    hplt = plotter.PlotterHorizon(ti.prb, solution)
    hplt.plotVariables([elem.getName() for elem in forces], show_bounds=True, gather=2, legend=False)
    hplt.plotVariables(['q'], show_bounds=True, gather=2, legend=False)
    matplotlib.pyplot.show()

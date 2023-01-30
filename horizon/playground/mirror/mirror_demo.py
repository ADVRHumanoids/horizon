from horizon.utils import plotter
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.ros import replay_trajectory
from horizon.solvers.solver import Solver
from horizon.rhc.taskInterface import TaskInterface
from horizon.rhc.model_description import *
from horizon.utils.utils import barrier, barrier_1
from horizon.utils.patternGenerator import PatternGenerator
from horizon.utils.trajectoryGenerator import TrajectoryGenerator
import numpy as np
import rospkg
import casadi as cs
import rospy
import subprocess
import time
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

solver_type = 'ipopt'
transcription_method = 'multiple_shooting'
transcription_opts = dict(integrator='RK4')

init_nodes = 10
end_nodes = 20
cycle_nodes = 20
clearance = 0.1
duty_cycle = 1.
vertical_constraint_nodes = 2
# ns = 100
ns = init_nodes + cycle_nodes + end_nodes

print(ns)
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

for contact in contacts:
    model.setContactFrame(contact, 'vertex', dict(vertex_frames=[contact]))

ptgt_final = base_init.copy()

# base x position
base_pos_x_param = prb.createParameter('base_pos_x_param', 1)
prb.createFinalConstraint('base_link_x', model.q[0] - base_pos_x_param)

# base y position
base_pos_y_param = prb.createParameter('base_pos_y_param', 1)
prb.createFinalResidual('base_link_y', 1e3 * (model.q[1] - base_pos_y_param))

# ================================================================================
# ================================================================================

# gait_matrix = np.array([[0, 1, 0],
#                         [1, 0, 0],
#                         [0, 0, 1]]).astype(int)

gait_matrix = np.array([[0],
                        [1],
                        [0]]).astype(int)

flight_with_duty = int(cycle_nodes / gait_matrix.shape[1] * duty_cycle)

pg = PatternGenerator(cycle_nodes, contacts)
stance_nodes, swing_nodes = pg.generatePattern(gait_matrix, cycle_nodes, duty_cycle)

list_init_nodes = list(range(init_nodes))

list_move_nodes = dict()
for c in contacts:
    list_move_nodes[c] = [init_nodes + elem for elem in stance_nodes[c]]

list_end_nodes = [init_nodes+cycle_nodes + elem for elem in range(end_nodes+1)]

for c in contacts:
    stance_nodes[c] = list_init_nodes + list_move_nodes[c] + list_end_nodes

    swing_nodes[c] = [init_nodes + elem for elem in swing_nodes[c]]


for name, elem in stance_nodes.items():
    print(f'{name}: {elem}')

for name, elem in swing_nodes.items():
    print(f'{name}: {elem}')

# ================================================================================
# ================================================================================

contact_pos = dict()
z_des = dict()
clea = dict()
for contact in contacts:
    FK = kd.fk(contact)
    DFK = kd.frameVelocity(contact, kd_frame)
    DDFK = kd.frameAcceleration(contact, kd_frame)

    ee_pos = FK(q=model.q)['ee_pos']
    ee_rot = FK(q=model.q)['ee_rot']
    ee_v = DFK(q=model.q, qdot=model.v)['ee_vel_linear']
    ee_v_ang = DFK(q=model.q, qdot=model.v)['ee_vel_angular']
    a = DDFK(q=model.q, qdot=model.v)['ee_acc_linear']

    contact_pos[contact] = FK(q=model.q0)['ee_pos']

    # vertical contact frame
    rot_err = cs.sumsqr(ee_rot[2, :2])
    # prb.createIntermediateCost(f'{contact}_rot', 1e4 * rot_err)

    # barrier force
    fcost = barrier(model.fmap[contact][2] - 10.0)  # fz > 10
    prb.createResidual(f'{contact}_unil', 1e1 * fcost, nodes=stance_nodes[contact][:-1]) # [1:-1]

    # contact constraint
    prb.createConstraint(f"{contact}_vel", cs.vertcat(ee_v, ee_v_ang), nodes=stance_nodes[contact])


    if swing_nodes[contact]:
        # clearance
        contact_pos[contact] = FK(q=model.q0)['ee_pos']
        z_des[contact] = prb.createParameter(f'{contact}_z_des', 1)
        clea[contact] = prb.createConstraint(f"{contact}_clea", ee_pos[2] - z_des[contact], nodes=swing_nodes[contact])

        # zero force
        model.fmap[contact].setBounds(np.array([[0, 0, 0]] * len(swing_nodes[contact])).T,
                                    np.array([[0, 0, 0]] * len(swing_nodes[contact])).T,
                                    nodes=swing_nodes[contact])

        # vertical takeoff and touchdown
        lat_vel = cs.vertcat(ee_v[0:2], ee_v_ang)
        vert = prb.createConstraint(f"{contact}_vert", lat_vel,
                                    nodes=swing_nodes[contact][:vertical_constraint_nodes] + swing_nodes[contact][-vertical_constraint_nodes:])

tg = TrajectoryGenerator()

for contact, z_constr in z_des.items():
        pos_z = contact_pos[contact][2].elements()[0]
        z_trj = np.atleast_2d(tg.from_derivatives(flight_with_duty, pos_z, pos_z, clearance, [0, 0, 0]))
        rep_param = np.concatenate([z_trj] * int((len(swing_nodes[contact]) + 10) / z_trj.shape[1]), axis=1)

        z_des[contact].assign(rep_param[:, :len(swing_nodes[contact])], nodes=swing_nodes[contact])



base_pos_x_param.assign(ptgt_final[0])
base_pos_y_param.assign(ptgt_final[1])

q0 = model.q0
v0 = model.v0
f0 = np.array([0, 0, kd.mass() * 9.8 / 3])

# final velocity
model.v.setBounds(v0, v0, nodes=ns)

# base rotation
prb.createResidual("min_rot", 1e-4 * (model.q[3:5] - q0[3:5]))

# joint posture
prb.createResidual("min_q", 1e-1 * (model.q[7:] - q0[7:]))

# joint velocity
prb.createResidual("min_v", 1e-2 * model.v)

# final posture
# todo: incredible, this is the problem. if it's FinalResidual, everything goes tho whores
prb.createResidual("min_qf", 1e1 * (model.q[7:] - q0[7:]))
# prb.createFinalResidual("min_qf", 1e1 * (model.q[7:] - model.q0[7:]))

# regularize input
prb.createIntermediateResidual("min_q_ddot", 1e0 * model.a)

# regularize forces
for f in model.fmap.values():
    prb.createIntermediateResidual(f"min_{f.getName()}", 1e-3 * (f - f0))

# com_fn = kd.centerOfMass()


# set initial condition and initial guess
model.q.setBounds(q0, q0, nodes=0)
model.v.setBounds(v0, v0, nodes=0)

model.q.setInitialGuess(q0)

for f in model.fmap.values():
    f.setInitialGuess(f0)

model.setDynamics()

if solver_type != 'ilqr':
    Transcriptor.make_method(transcription_method, prb, transcription_opts)

# print('VARIABLES:')
# for var_name, obj in prb.getVariables().items():
#     print(var_name, ':', type(obj))
#     print(obj)
#     print(obj.getNodes().tolist())
#     print(obj.getBounds())

print('CONSTRAINTS:')
for cnsrt, obj in prb.getConstraints().items():
    print(cnsrt,':', type(obj))
    print(obj.getFunction())
    print(obj._fun_impl)
    print(obj.getNodes())
    print(obj.getBounds())
#
print('COSTS:')
for cnsrt, obj in prb.getCosts().items():
    print(cnsrt,':', obj.getNodes(), type(obj))


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

opts_rti = opts.copy()
opts_rti['ilqr.enable_line_search'] = False
opts_rti['ilqr.max_iter'] = 4

solver_bs = Solver.make_solver(solver_type, prb, opts)

try:
    solver_bs.set_iteration_callback()
except:
    pass

t = time.time()
solver_bs.solve()
elapsed = time.time() - t
print(f'bootstrap solved in {elapsed} s')

solution = solver_bs.getSolutionDict()



## single replay
q_sol = solution['q']
frame_force_mapping = {contacts[i]: solution[f'f_{contacts[i]}'] for i in range(3)}
repl = replay_trajectory.replay_trajectory(dt, kd.joint_names(), q_sol, frame_force_mapping, kd_frame, kd)
repl.sleep(1.)
repl.replay(is_floating_base=True)
exit()

plot_flag = True
if plot_flag:
    import matplotlib.pyplot as plt
    import matplotlib

    plt.figure()
    for contact in contacts:
        FK = cs.Function.deserialize(kd.fk(contact))
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'feet position - plane_xy')
        plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[1, :]).flatten(), linewidth=2.5)
        plt.scatter(np.array(pos[0, 0]), np.array(pos[1, 0]))
        plt.scatter(np.array(pos[0, -1]), np.array(pos[1, -1]), marker='x')

    plt.figure()
    for contact in contacts:
        FK = cs.Function.deserialize(kd.fk(contact))
        pos = FK(q=solution['q'])['ee_pos']

        plt.title(f'feet position - plane_xz')
        plt.plot(np.array(pos[0, :]).flatten(), np.array(pos[2, :]).flatten(), linewidth=2.5)
        plt.scatter(np.array(pos[0, 0]), np.array(pos[2, 0]))
        plt.scatter(np.array(pos[0, -1]), np.array(pos[2, -1]), marker='x')

    hplt = plotter.PlotterHorizon(ti.prb, solution)
    hplt.plotVariables([elem.getName() for elem in forces], show_bounds=True, gather=2, legend=False)
    hplt.plotVariables(['q'], show_bounds=True, gather=2, legend=False)
    matplotlib.pyplot.show()
